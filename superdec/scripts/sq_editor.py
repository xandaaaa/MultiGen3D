"""Standalone superquadric editor for MultiGen3D dataset_20 shapes.

A viser-based web editor. No TRELLIS / no GPU — purely edits the 4-field .npz
format (scales, shapes, rotations, translations) produced by SuperDec and
consumed by experiments/approach*.py.

Workflow
  1. Pick a shape from the `Shape → File` dropdown, click Load.
  2. Click any colored SQ in the 3D view to SELECT it — its gizmo and sliders
     appear in the "Selected SQ" panel at the top of the sidebar.
  3. Edit with the gizmo (drag arrows/rings) or the sliders (shape, scale).
  4. Duplicate / Delete the selected SQ from the buttons at the top.
  5. Save → writes `<stem>_edited.npz` next to the original (or overwrite).

Each SQ has a floating label in the 3D view showing its index, colored to
match scripts/preview_sqs.py — so `SQ 3` in the editor is the same `SQ 3` in
your approach6-style prompt files.

Usage
-----
  conda activate spacecontrol
  python superdec/scripts/sq_editor.py [--port 8080] [--root data/dataset_20]
"""
from __future__ import annotations

import argparse
import colorsys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import viser
import viser.transforms as vtf

SUPERDEC_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = SUPERDEC_DIR / "data" / "dataset_20"

# ---------- superquadric math ----------

def superquadric_mesh(scale, exponents, rotation, translation, N: int = 50):
    def f(o, m): return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m
    def g(o, m): return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m
    u = np.linspace(-np.pi, np.pi, N, endpoint=True)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True)
    u = np.tile(u, N); v = np.repeat(v, N)
    if np.linalg.det(rotation) < 0:
        u = u[::-1]
    x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
    y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
    z = scale[2] * f(v, exponents[0])
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
    return verts.astype(np.float32), np.array(tris, dtype=np.int32)


def sq_color(idx: int) -> tuple[int, int, int]:
    h = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


# ---------- .npz I/O ----------

def load_npz(path: Path) -> List[Dict]:
    d = np.load(path)
    P = d["scales"].shape[0]
    return [{
        "scale":       np.asarray(d["scales"][k],       dtype=np.float64).copy(),
        "shape":       np.asarray(d["shapes"][k],       dtype=np.float32).copy(),
        "rotation":    np.asarray(d["rotations"][k],    dtype=np.float32).copy(),
        "translation": np.asarray(d["translations"][k], dtype=np.float64).copy(),
    } for k in range(P)]


def save_npz(path: Path, sqs: List[Dict]) -> None:
    np.savez(path,
        scales=      np.stack([s["scale"]       for s in sqs]),
        shapes=      np.stack([s["shape"]       for s in sqs]),
        rotations=   np.stack([s["rotation"]    for s in sqs]),
        translations=np.stack([s["translation"] for s in sqs]))


# ---------- editor ----------

class Editor:
    def __init__(self, server: viser.ViserServer, root: Path):
        self.server = server
        self.root = root
        self.npz_dir = root / "npz"
        if not self.npz_dir.exists():
            raise FileNotFoundError(f"{self.npz_dir} not found")

        self.current_path: Path | None = None
        self.sqs: List[Dict] = []
        self.selected: Optional[int] = None

        self.mesh_handles: Dict[int, viser.MeshHandle] = {}
        self.label_handles: Dict[int, viser.LabelHandle] = {}
        self.gizmo_handles: Dict[int, viser.TransformControlsHandle] = {}

        # Per-SQ slider handles live here so we can rebuild them for the selected SQ only.
        self.sel_slider_handles: Dict[str, viser.GuiInputHandle] = {}
        self.sel_slider_folder: Optional[viser.GuiFolderHandle] = None

        self._build_toolbar()
        first = self._list_shapes()[0] if self._list_shapes() else None
        if first:
            self.shape_dropdown.value = first
            self._load_current()

    # ---- toolbar ----

    def _list_shapes(self) -> List[str]:
        return sorted(p.stem for p in self.npz_dir.glob("*.npz"))

    def _build_toolbar(self):
        shapes = self._list_shapes()
        with self.server.gui.add_folder("Shape", expand_by_default=True):
            self.shape_dropdown = self.server.gui.add_dropdown(
                "File", options=shapes, initial_value=shapes[0] if shapes else "")
            load_btn = self.server.gui.add_button("Load", icon=viser.Icon.FOLDER_OPEN)
            load_btn.on_click(lambda _: self._load_current())
            reload_btn = self.server.gui.add_button("Reload (discard edits)", icon=viser.Icon.REFRESH)
            reload_btn.on_click(lambda _: self._load_current())

        # Selected SQ panel lives up top so Delete/Duplicate + sliders are always visible.
        with self.server.gui.add_folder("Selected SQ", expand_by_default=True):
            self.sel_header = self.server.gui.add_markdown("_Click a SQ in the 3D view_")
            self.dup_btn = self.server.gui.add_button("Duplicate", icon=viser.Icon.COPY, disabled=True)
            self.del_btn = self.server.gui.add_button("Delete", color="red", icon=viser.Icon.TRASH, disabled=True)
            self.dup_btn.on_click(lambda _: self._duplicate_selected())
            self.del_btn.on_click(lambda _: self._delete_selected())
            # Sliders get added/removed into this inner folder.
            self.sel_sliders_folder_parent = self.server.gui.add_folder(
                "Edit parameters", expand_by_default=True)

        with self.server.gui.add_folder("Save", expand_by_default=True):
            self.save_suffix_box = self.server.gui.add_text("Suffix", initial_value="_edited")
            self.overwrite_cb = self.server.gui.add_checkbox("Overwrite original", initial_value=False)
            save_btn = self.server.gui.add_button("Save", color="green", icon=viser.Icon.DEVICE_FLOPPY)
            save_btn.on_click(lambda _: self._save_current())
            self.save_status = self.server.gui.add_markdown("_not saved_")

        with self.server.gui.add_folder("Add / Info", expand_by_default=True):
            self.info_md = self.server.gui.add_markdown("")
            self.legend_md = self.server.gui.add_markdown("")
            add_btn = self.server.gui.add_button("Add new SQ (unit box)", icon=viser.Icon.PLUS)
            add_btn.on_click(lambda _: self._add_sq())

    # ---- loading ----

    def _load_current(self):
        name = self.shape_dropdown.value
        if not name:
            return
        path = self.npz_dir / f"{name}.npz"
        self.current_path = path
        self.sqs = load_npz(path)
        self.selected = None
        self._rebuild_scene()
        self._refresh_selected_panel()
        self._refresh_info()
        self.save_status.content = f"_loaded `{path.name}` (P={len(self.sqs)})_"

    def _rebuild_scene(self):
        for d in (self.mesh_handles, self.label_handles, self.gizmo_handles):
            for h in list(d.values()):
                try: h.remove()
                except Exception: pass
            d.clear()

        for i in range(len(self.sqs)):
            self._add_sq_scene(i)

    def _add_sq_scene(self, i: int):
        sq = self.sqs[i]
        v, t = superquadric_mesh(sq["scale"], sq["shape"], sq["rotation"], sq["translation"], N=40)
        color = sq_color(i)
        mesh = self.server.scene.add_mesh_simple(
            f"/sq_{i}", vertices=v, faces=t,
            color=color, opacity=0.85, wireframe=False, flat_shading=False)
        mesh.on_click(lambda _, idx=i: self._select(idx))
        self.mesh_handles[i] = mesh

        # Floating 3D label — always visible, shows SQ index at the centroid.
        centroid = v.mean(0).astype(np.float32)
        self.label_handles[i] = self.server.scene.add_label(
            f"/label_{i}", text=f"SQ {i}", position=tuple(centroid.tolist()),
            font_screen_scale=1.3)

        # Gizmo: always created but hidden unless selected.
        rot_wxyz = vtf.SO3.from_matrix(sq["rotation"]).wxyz
        gz = self.server.scene.add_transform_controls(
            f"/gizmo_{i}", scale=0.18, line_width=2.5,
            position=tuple(sq["translation"]), wxyz=rot_wxyz, visible=(i == self.selected))
        gz.on_update(lambda _, idx=i: self._on_gizmo_update(idx))
        self.gizmo_handles[i] = gz

    def _on_gizmo_update(self, i: int):
        g = self.gizmo_handles[i]
        self.sqs[i]["translation"] = np.asarray(g.position, dtype=np.float64)
        self.sqs[i]["rotation"]    = vtf.SO3(wxyz=np.asarray(g.wxyz)).as_matrix().astype(np.float32)
        self._redraw_sq(i)

    def _redraw_sq(self, i: int):
        sq = self.sqs[i]
        v, t = superquadric_mesh(sq["scale"], sq["shape"], sq["rotation"], sq["translation"], N=40)
        # Replace mesh (same name overwrites) and re-wire click handler.
        mesh = self.server.scene.add_mesh_simple(
            f"/sq_{i}", vertices=v, faces=t,
            color=sq_color(i), opacity=0.85, wireframe=False, flat_shading=False)
        mesh.on_click(lambda _, idx=i: self._select(idx))
        self.mesh_handles[i] = mesh
        # Reposition label
        centroid = v.mean(0).astype(np.float32)
        self.label_handles[i].position = tuple(centroid.tolist())

    # ---- selection ----

    def _select(self, i: int):
        if i == self.selected:
            return
        # Hide previous gizmo, show new one.
        if self.selected is not None and self.selected in self.gizmo_handles:
            self.gizmo_handles[self.selected].visible = False
        self.selected = i
        if i in self.gizmo_handles:
            self.gizmo_handles[i].visible = True
        self._refresh_selected_panel()

    def _refresh_selected_panel(self):
        # Tear down existing sliders.
        if self.sel_slider_folder is not None:
            try: self.sel_slider_folder.remove()
            except Exception: pass
            self.sel_slider_folder = None
        self.sel_slider_handles.clear()

        if self.selected is None or self.selected >= len(self.sqs):
            self.sel_header.content = "_Click a SQ in the 3D view_"
            self.dup_btn.disabled = True
            self.del_btn.disabled = True
            return

        i = self.selected
        r, g, b = sq_color(i)
        self.sel_header.content = (
            f"### SQ {i}  "
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"background:rgb({r},{g},{b});border:1px solid #000;vertical-align:middle;'></span>"
        )
        self.dup_btn.disabled = False
        self.del_btn.disabled = False

        sq = self.sqs[i]
        # Re-add sliders under the Selected SQ folder. We stash them under a
        # fresh folder so we can remove them wholesale on selection change.
        self.sel_slider_folder = self.server.gui.add_folder(
            f"SQ {i} parameters", expand_by_default=True)
        with self.sel_slider_folder:
            sh0 = self.server.gui.add_slider(
                "shape_0 (e1)", min=0.01, max=2.0, step=0.01, initial_value=float(sq["shape"][0]))
            sh1 = self.server.gui.add_slider(
                "shape_1 (e2)", min=0.01, max=2.0, step=0.01, initial_value=float(sq["shape"][1]))
            sx = self.server.gui.add_slider("scale_x", min=0.001, max=1.0, step=0.002,
                                            initial_value=float(sq["scale"][0]))
            sy = self.server.gui.add_slider("scale_y", min=0.001, max=1.0, step=0.002,
                                            initial_value=float(sq["scale"][1]))
            sz = self.server.gui.add_slider("scale_z", min=0.001, max=1.0, step=0.002,
                                            initial_value=float(sq["scale"][2]))

        def on_shape_change(_):
            if self.selected is None or self.selected != i: return
            sq["shape"][0] = sh0.value
            sq["shape"][1] = sh1.value
            self._redraw_sq(i)
        def on_scale_change(_):
            if self.selected is None or self.selected != i: return
            sq["scale"][0] = sx.value
            sq["scale"][1] = sy.value
            sq["scale"][2] = sz.value
            self._redraw_sq(i)
        for h in (sh0, sh1): h.on_update(on_shape_change)
        for h in (sx, sy, sz): h.on_update(on_scale_change)
        self.sel_slider_handles = {"sh0": sh0, "sh1": sh1, "sx": sx, "sy": sy, "sz": sz}

    # ---- info / legend ----

    def _refresh_info(self):
        P = len(self.sqs)
        path = self.current_path.name if self.current_path else "(none)"
        self.info_md.content = f"**File:** `{path}`  \n**SQ count:** {P}"
        # Compact color-legend: one swatch per SQ, index under.
        rows = []
        for i in range(P):
            r, g, b = sq_color(i)
            rows.append(
                f"<span style='display:inline-block;width:18px;height:12px;"
                f"background:rgb({r},{g},{b});border:1px solid #000;margin-right:2px;'></span>"
                f"<small>{i}</small>"
            )
        self.legend_md.content = "**Legend:** " + "&nbsp;&nbsp;".join(rows) if rows else ""

    # ---- add / duplicate / delete ----

    def _add_sq(self):
        self.sqs.append({
            "scale":       np.array([0.1, 0.1, 0.1], dtype=np.float64),
            "shape":       np.array([1.0, 1.0], dtype=np.float32),
            "rotation":    np.eye(3, dtype=np.float32),
            "translation": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        })
        new_idx = len(self.sqs) - 1
        self.selected = new_idx
        self._rebuild_scene()
        self._refresh_selected_panel()
        self._refresh_info()

    def _duplicate_selected(self):
        if self.selected is None: return
        src = self.sqs[self.selected]
        dup = {k: v.copy() for k, v in src.items()}
        dup["translation"] = dup["translation"] + np.array([0.02, 0.02, 0.02])
        self.sqs.append(dup)
        self.selected = len(self.sqs) - 1
        self._rebuild_scene()
        self._refresh_selected_panel()
        self._refresh_info()

    def _delete_selected(self):
        if self.selected is None: return
        self.sqs.pop(self.selected)
        # After a delete, indices shift — simplest is full rebuild with selection cleared.
        self.selected = None
        self._rebuild_scene()
        self._refresh_selected_panel()
        self._refresh_info()

    # ---- save ----

    def _save_current(self):
        if self.current_path is None or not self.sqs:
            self.save_status.content = "_nothing to save_"
            return
        out = self.current_path if self.overwrite_cb.value else \
              self.current_path.with_name(self.current_path.stem + self.save_suffix_box.value + ".npz")
        save_npz(out, self.sqs)
        self.save_status.content = f"_saved to `{out.name}` (P={len(self.sqs)})_"
        print(f"saved {out}")
        self.shape_dropdown.options = tuple(self._list_shapes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port, up_axis=2)
    server.scene.set_up_direction([0.0, 0.0, 1.0])
    server.gui.configure_theme(dark_mode=True)

    editor = Editor(server, args.root)
    print(f"\n[editor] open http://localhost:{args.port}  (root: {args.root})")
    print("[editor] click a SQ to select; Delete/Duplicate live in 'Selected SQ' panel")
    print("[editor] ctrl-c to stop\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[editor] shutting down")


if __name__ == "__main__":
    main()
