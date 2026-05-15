"""
benchmark/run_benchmark.py

Runs one approach on one or all shapes and saves renders in the layout
expected by benchmark/clip_score.py:

    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/view_<j>.png

Also saves a row image per prompt for quick visual inspection:

    <results_root>/<approach>_results/renders/<shape_id>/prompt_<i>/grid.png

Approaches:
    baseline         — standard TRELLIS, global prompt only (no SQ routing)
    approach5        — height-grouped semantic routing (3 groups: bottom / mid / top)
    approach6        — legacy per-SQ hard routing (one distinct prompt per SQ)
    local_sq         — older local_sq.py regional refine implementation
    decode_composite — current best: per-prompt SLAT → gaussian-level hard 3D merge
    approach7        — coupled diffusion (soft W + global coupling branch)

Usage:
    python benchmark/run_benchmark.py --approach baseline --shape-idx 3
    python benchmark/run_benchmark.py --approach baseline --shape-idx all
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# --- Path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "experiments"))
os.environ["SPCONV_ALGO"] = "native"

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from approach1_experiment import coords_to_world, compute_mesh_normalization, load_sq_params
from approach5_experiment import group_sqs_by_height, compute_hard_W, sample_slat_compositional
from approach6_experiment import compute_hard_W as compute_hard_W_6, sample_slat_regional_refine
from local_sq import (
    compute_hard_W as compute_hard_W_local_sq,
    make_contextual_local_prompts,
    sample_slat_regional_refine as sample_slat_local_sq,
)
from decode_composite import decode_composite_gaussian
from approach7_experiment import compute_soft_W as compute_soft_W_7, sample_slat_coupled


# SuperDec/ShapeNet benchmark assets are saved in the original ShapeNet frame
# (Y-up). TRELLIS cameras, spatial control, and decoded assets are used as Z-up.
SHAPENET_TO_TRELLIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def convert_sq_params_to_trellis_frame(sq_params):
    converted = []
    for sq in sq_params:
        out = dict(sq)
        out["translation"] = SHAPENET_TO_TRELLIS @ np.asarray(sq["translation"], dtype=np.float32)
        out["rotation"] = SHAPENET_TO_TRELLIS @ np.asarray(sq["rotation"], dtype=np.float32)
        converted.append(out)
    return converted


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_extrinsics_intrinsics():
    return render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2], [0.35] * 4, 10, 8
    )


def _superquadric_mesh_vf(scale, shape, translation, rotation, N=50):
    """Build (vertices, triangles) for one superquadric — same formula as the GUI."""
    A, B, C = scale[0], scale[1], scale[2]
    e1, e2 = shape[0], shape[1]
    def f(o, m): return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m
    def g(o, m): return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m
    u = np.linspace(-np.pi, np.pi, N, endpoint=True)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True)
    u = np.tile(u, N)
    v = np.repeat(v, N)
    if np.linalg.det(rotation) < 0:
        u = u[::-1]
    x = A * g(v, e1) * g(u, e2)
    y = B * g(v, e1) * f(u, e2)
    z = C * f(v, e1)
    x[:N] = 0.0
    x[-N:] = 0.0
    verts = np.stack([x, y, z], axis=1)
    verts = (rotation @ verts.T).T + translation
    tris = []
    for i in range(N - 1):
        for j in range(N - 1):
            tris.append([i * N + j, i * N + j + 1, (i + 1) * N + j])
            tris.append([(i + 1) * N + j, i * N + j + 1, (i + 1) * N + (j + 1)])
    for i in range(N - 1):
        tris.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
        tris.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])
    tris.append([(N - 1) * N + (N - 1), (N - 1) * N, (N - 1)])
    tris.append([(N - 1), (N - 1) * N, 0])
    return verts, np.array(tris)


def build_normalized_sq_mesh_o3d(sq_params, mesh_center, mesh_scale):
    """Return an open3d mesh of all SQs merged and normalized to the pipeline's coord
    space (centered at 0, scaled so AABB max is 1)."""
    import open3d as o3d
    from gui.utils import merge_meshes
    meshes = []
    for sq in sq_params:
        v, f = _superquadric_mesh_vf(sq['scale'], sq['shape'], sq['translation'], sq['rotation'], N=50)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(v)
        m.triangles = o3d.utility.Vector3iVector(f)
        meshes.append(m)
    merged = merge_meshes(meshes)
    merged.translate(-mesh_center)
    merged.scale(mesh_scale, (0, 0, 0))
    return merged


def render_sq_mesh(sq_mesh_o3d, extr, intr):
    """Render the merged+normalized SQ mesh from the same views as the gaussian renders.
    Returns a list of np.uint8 normal-shaded frames, or None on failure.
    """
    from trellis.representations.mesh.cube2mesh import MeshExtractResult
    verts = torch.tensor(np.asarray(sq_mesh_o3d.vertices), dtype=torch.float32, device="cuda")
    faces = torch.tensor(np.asarray(sq_mesh_o3d.triangles), dtype=torch.long, device="cuda")
    if verts.numel() == 0 or faces.numel() == 0:
        return None
    mesh = MeshExtractResult(vertices=verts, faces=faces, res=64)
    out = render_utils.render_frames(mesh, extr, intr, {"resolution": 512}, verbose=False)
    return out.get("normal")


def render_gs(gs, extr, intr, prompt=""):
    scales = gs.get_scaling
    if not torch.isfinite(scales).all() or scales.max().item() > 10.0:
        print(f"  WARNING: degenerate Gaussian scales (max={scales.max().item():.3g}) — skipping render")
        return None
    bg_color = (0, 0, 0) if "white" in prompt.lower() else (255, 255, 255)
    frames = render_utils.render_frames(
        gs, extr, intr, {"resolution": 512, "bg_color": bg_color}
    )["color"]
    return frames


def render_gaussian(pipeline, slat, extr, intr, prompt=""):
    gs = pipeline.decode_slat(slat, formats=["gaussian"])["gaussian"][0]
    frames = render_gs(gs, extr, intr, prompt)
    del gs
    return frames


def save_renders(frames, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"view_{j}.png")


def all_views_exist(out_dir: Path, n_views: int = 4) -> bool:
    return all((out_dir / f"view_{j}.png").exists() for j in range(n_views))


# ---------------------------------------------------------------------------
# Per-approach runners  (all return a raw slat tensor)
# ---------------------------------------------------------------------------

def run_baseline(pipeline, coords, global_prompt, steps, seed, cfg_strength):
    cond = pipeline.get_cond_text([global_prompt])
    torch.manual_seed(seed)
    return pipeline.sample_slat(cond, coords, sampler_params={"steps": steps})


def run_approach5(pipeline, coords, sq_params, mesh_center, mesh_scale,
                  global_prompt, local_prompts, steps, seed, cfg_strength):
    W = compute_hard_W(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    group_map = group_sqs_by_height(sq_params, mesh_center)

    # One prompt per height group — take the first SQ assigned to each group
    group_prompts = {}
    for sq_idx, grp in group_map.items():
        if grp not in group_prompts:
            group_prompts[grp] = local_prompts.get(sq_idx, global_prompt)
    for g in range(3):
        if g not in group_prompts:
            group_prompts[g] = global_prompt

    conds_local = {g: pipeline.get_cond_text([p]) for g, p in group_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_compositional(
        pipeline, coords, W, group_map, conds_local,
        steps=steps, cfg_strength=cfg_strength,
    )


def run_approach6(pipeline, coords, sq_params, mesh_center, mesh_scale,
                  global_prompt, local_prompts, steps, seed, cfg_strength):
    W = compute_hard_W_6(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_regional_refine(
        pipeline, coords, W, conds_local, cond_global,
        global_steps=steps, cfg_strength=cfg_strength,
    )


def run_local_sq(pipeline, coords, sq_params, mesh_center, mesh_scale,
                 global_prompt, local_prompts, steps, seed, cfg_strength):
    W = compute_hard_W_local_sq(coords_to_world(coords), sq_params, mesh_center, mesh_scale)
    cond_global = pipeline.get_cond_text([global_prompt])
    contextual_prompts = make_contextual_local_prompts(global_prompt, local_prompts)
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in contextual_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_local_sq(
        pipeline, coords, W, conds_local, cond_global,
        global_steps=steps, cfg_strength=cfg_strength,
    )


def run_decode_composite(pipeline, coords, sq_params, mesh_center, mesh_scale,
                          global_prompt, local_prompts, steps, seed, cfg_strength,
                          local_cfg=15.0):
    """New approach matching the GUI: independent SLAT per unique prompt → decode
    each to gaussians → hard-assign gaussians by SQ region in 3D → merge.
    Returns a Gaussian (already decoded), not a slat.
    """
    cond_global = pipeline.get_cond_text([global_prompt])
    # Build per-SQ cond mapping with global fallback, deduplicating identical prompts.
    n_sq = len(sq_params)
    prompt_to_cond = {global_prompt: cond_global}
    sq_prompt_str = {}
    for i in range(n_sq):
        p = local_prompts.get(i, '').strip() if isinstance(local_prompts.get(i, ''), str) else ''
        p = p or global_prompt
        sq_prompt_str[i] = p
        if p not in prompt_to_cond:
            prompt_to_cond[p] = pipeline.get_cond_text([p])
    conds_local = {i: prompt_to_cond[sq_prompt_str[i]] for i in range(n_sq)}

    torch.manual_seed(seed)
    merged_g, _mesh = decode_composite_gaussian(
        pipeline, coords, conds_local, cond_global, sq_params, mesh_center, mesh_scale,
        steps=steps, cfg_strength=cfg_strength, rescale_t=3.0,
        local_cfg_strength=local_cfg,
    )
    return merged_g


def run_approach7(pipeline, coords, sq_params, mesh_center, mesh_scale,
                  global_prompt, local_prompts, steps, seed, cfg_strength,
                  lam=0.3, tau=0.02):
    W = compute_soft_W_7(coords_to_world(coords), sq_params, mesh_center, mesh_scale, tau=tau)
    cond_global = pipeline.get_cond_text([global_prompt])
    conds_local = {k: pipeline.get_cond_text([v]) for k, v in local_prompts.items()}
    torch.manual_seed(seed)
    return sample_slat_coupled(
        pipeline, coords, W, conds_local, cond_global,
        steps=steps, cfg_strength=cfg_strength, lam=lam,
    )


# ---------------------------------------------------------------------------
# Per-shape runner
# ---------------------------------------------------------------------------

def run_shape(shape, approach, pipeline, extr, intr, results_root, steps, seed, cfg_strength, args):
    shape_id = shape["id"]
    n_prompts = len(shape["prompts"])
    output_root = Path(results_root) / f"{approach}_results" / "renders" / shape_id

    print(f"\n{'='*60}")
    print(f"Shape: {shape_id}  ({n_prompts} prompts)")

    if not args.force and all(all_views_exist(output_root / f"prompt_{i}") for i in range(n_prompts)):
        print("  All renders already exist — skipping.")
        return

    sq_params = convert_sq_params_to_trellis_frame(
        load_sq_params(os.path.join(project_root, shape["npz"]))
    )
    mesh_center, mesh_scale = compute_mesh_normalization(sq_params)

    # Build SQ mesh once per shape; render reference views to drop into each prompt folder.
    sq_mesh_shape = build_normalized_sq_mesh_o3d(sq_params, mesh_center, mesh_scale)
    sq_frames = render_sq_mesh(sq_mesh_shape, extr, intr)

    for prompt_idx in range(n_prompts):
        out_dir = output_root / f"prompt_{prompt_idx}"
        if not args.force and all_views_exist(out_dir):
            print(f"  prompt_{prompt_idx}: already rendered, skipping")
            continue

        global_prompt = shape["prompts"][prompt_idx]
        raw_local = shape.get("local_prompts", [{}] * n_prompts)[prompt_idx]
        local_prompts = {int(k): v for k, v in raw_local.items()}
        per_prompt_seed = (shape.get("seeds") or [None] * n_prompts)[prompt_idx]
        current_seed = per_prompt_seed if per_prompt_seed is not None else seed

        print(f"\n  prompt_{prompt_idx}: {global_prompt[:80]}")
        if per_prompt_seed is not None:
            print(f"  (using per-prompt seed={current_seed})")

        gs_direct = None
        slat = None
        coords = None
        cond_struct = None

        if approach in ("baseline", "decode_composite"):
            # GUI-exact path: write SQ mesh as spatial control, use it.
            import open3d as o3d
            spatial_path = str(Path(args.results_root) / f"{approach}_results" / "_spatial_control.ply")
            os.makedirs(os.path.dirname(spatial_path), exist_ok=True)
            o3d.io.write_triangle_mesh(spatial_path, sq_mesh_shape)

            if approach == "baseline":
                # Matches GUI generate(): pipeline.run with spatial_control_mesh_path.
                torch.manual_seed(current_seed)
                outputs = pipeline.run(
                    global_prompt, None, seed=current_seed,
                    sparse_structure_sampler_params={
                        "steps": args.gui_structure_steps,
                        "cfg_strength": cfg_strength,
                        "t0_idx_value": args.t0_idx,
                        "spatial_control_mesh_path": spatial_path,
                    },
                )
                gs_direct = outputs["gaussian"][0]
            else:
                # decode_composite: sample structure with control, then decode_composite.
                cond_global_for_struct = pipeline.get_cond_text([global_prompt])
                cond_struct = {**cond_global_for_struct,
                               'control': pipeline.encode_spatial_control(spatial_path)}
                torch.manual_seed(current_seed)
                coords = pipeline.sample_sparse_structure(
                    cond_struct, num_samples=1,
                    sampler_params={
                        "steps": args.gui_structure_steps,
                        "cfg_strength": cfg_strength,
                        "t0_idx_value": args.t0_idx,
                    },
                )
                gs_direct = run_decode_composite(
                    pipeline, coords, sq_params, mesh_center, mesh_scale,
                    global_prompt, local_prompts,
                    args.gui_slat_steps, current_seed, cfg_strength,
                    local_cfg=args.local_cfg,
                )
        else:
            # Other approaches: existing text-only structure sampling.
            cond_struct = pipeline.get_cond_text([global_prompt])
            torch.manual_seed(current_seed)
            coords = pipeline.sample_sparse_structure(
                cond_struct, num_samples=1, sampler_params={"steps": steps}
            )
            if approach == "approach5":
                slat = run_approach5(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                     global_prompt, local_prompts, steps, current_seed, cfg_strength)
            elif approach == "approach6":
                slat = run_approach6(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                     global_prompt, local_prompts, steps, current_seed, cfg_strength)
            elif approach == "local_sq":
                slat = run_local_sq(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                    global_prompt, local_prompts, steps, current_seed, cfg_strength)
            elif approach == "approach7":
                slat = run_approach7(pipeline, coords, sq_params, mesh_center, mesh_scale,
                                     global_prompt, local_prompts, steps, current_seed, cfg_strength,
                                     lam=args.lam, tau=args.tau)

        if gs_direct is not None:
            frames = render_gs(gs_direct, extr, intr, prompt=global_prompt)
            del gs_direct
        else:
            frames = render_gaussian(pipeline, slat, extr, intr, prompt=global_prompt)
            del slat
        if coords is not None:
            del coords
        if cond_struct is not None:
            del cond_struct
        if frames is None:
            print(f"    WARNING: degenerate Gaussian for prompt_{prompt_idx}, skipping")
        else:
            save_renders(frames, out_dir)
            print(f"    Saved {len(frames)} views → {out_dir}")

        # Also save SQ reference views (same camera params) into this prompt's folder.
        if sq_frames is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(sq_frames):
                Image.fromarray(frame).save(out_dir / f"sq_view_{j}.png")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Shape {shape_id} done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", required=True,
                        choices=["baseline", "approach5", "approach6", "local_sq", "decode_composite", "approach7"])
    parser.add_argument("--shape-idx", default="all",
                        help="Index into prompts JSON (0-based), or 'all' to run every shape")
    parser.add_argument("--prompts-file", default="benchmark/prompts_augmented.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-strength", type=float, default=7.5)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--local-cfg", type=float, default=15.0,
                        help="CFG strength for per-prompt local SLATs (decode_composite only)")
    parser.add_argument("--t0-idx", type=float, default=6.0,
                        help="Spatial-control strength (t0_idx_value); baseline + decode_composite")
    parser.add_argument("--gui-structure-steps", type=int, default=12,
                        help="Structure-sampler steps for baseline + decode_composite (GUI default)")
    parser.add_argument("--gui-slat-steps", type=int, default=25,
                        help="SLAT-sampler steps for decode_composite (GUI default)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate renders even when view PNGs already exist")
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        shapes = json.load(f)

    if args.shape_idx == "all":
        indices = list(range(len(shapes)))
    else:
        idx = int(args.shape_idx)
        if idx >= len(shapes):
            sys.exit(f"shape-idx {idx} out of range (have {len(shapes)} shapes)")
        indices = [idx]

    print(f"Approach : {args.approach}")
    print(f"Shapes   : {indices}")

    print("Loading pipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(os.path.join(project_root, "gui"))
    pipeline.cuda()
    extr, intr = get_extrinsics_intrinsics()

    for idx in indices:
        run_shape(shapes[idx], args.approach, pipeline, extr, intr,
                  args.results_root, args.steps, args.seed, args.cfg_strength, args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
