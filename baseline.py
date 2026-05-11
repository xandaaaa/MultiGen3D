import os
import sys
import torch
import numpy as np
from PIL import Image

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "experiments"))

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils
from approach1_experiment import load_sq_params, compute_mesh_normalization

def run_baseline():
    # 1. Setup
    seed = 1024
    torch.manual_seed(seed)
    
    # Init Pipeline
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()
    
    # 2. Construct the "Naive" Long Prompt
    # We take all the local descriptions and jam them into one string
    global_part = "A Bauhaus chair"
    details = [
        "a yellow chair leg",
        "a green backrest",
        "a red seat cushion"
    ]
    naive_prompt = f"{global_part}, featuring: {', '.join(details)}"
    
    print(f"Running Baseline with prompt: {naive_prompt}")
    
    cond_naive = pipeline.get_cond_text([naive_prompt])
    
    # 3. Structure Sampling (Fixed Seed)
    # Using the same structure as your advanced experiments
    coords = pipeline.sample_sparse_structure(cond_naive, num_samples=1)
    
    # 4. Standard Sampling (No spatial fusion)
    # Note: We use the standard sample_slat method, not our custom advanced one
    slat_res = pipeline.sample_slat(cond_naive, coords, sampler_params={"steps": 15})
    
    # 5. Decode and Render
    gs_res = pipeline.decode_slat(slat_res, formats=['gaussian'])['gaussian'][0]
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0, np.pi/2, np.pi, 3*np.pi/2], [0.35]*4, 10, 8)
    frames = render_utils.render_frames(gs_res, extr, intr, {'resolution': 512, 'bg_color': (255,255,255)})['color']
    
    # Save
    output_dir = "/work/courses/3dv/team4/MultiGen3D/exp_6_outputs/seed1024"
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(np.concatenate(frames, axis=1)).save(os.path.join(output_dir, "baseline1024.png"))
    print("Baseline result saved.")

if __name__ == "__main__":
    run_baseline()