#!/bin/bash
#SBATCH --job-name=local_sq_bench
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/local_sq_bench_%j.out
#SBATCH --error=logs/local_sq_bench_%j.err

set -euo pipefail
echo "=========================================="
echo "Job started: $(date) on $SLURMD_NODENAME"
echo "=========================================="

module load cuda/12.8
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"
conda activate spacecontrol

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/courses/3dv/team4/MultiGen3D

# Full benchmark with the pre-91da279 extreme_v1 sampler.
# Overwrites prior local_sq_results renders for fair CLIP comparison.
python benchmark/run_benchmark.py \
    --approach local_sq \
    --shape-idx all \
    --sampler-variant extreme_v1 \
    --force

echo "=========================================="
echo "Job completed: $(date). Runtime: ${SECONDS}s"
echo "=========================================="
