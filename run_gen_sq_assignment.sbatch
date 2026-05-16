#!/bin/bash
#SBATCH --job-name=gen_sq_viz
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/gen_sq_viz_%j.out
#SBATCH --error=logs/gen_sq_viz_%j.err

set -euo pipefail
echo "Job started: $(date) on $SLURMD_NODENAME"

module load cuda/12.8
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"
conda activate spacecontrol

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team4/MultiGen3D
python benchmark/gen_sq_assignment.py

echo "Job completed: $(date). Runtime: ${SECONDS}s"
