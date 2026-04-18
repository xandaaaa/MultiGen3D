#!/bin/bash
#SBATCH --job-name=superdec_cand
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/superdec_cand_%j.out
#SBATCH --error=logs/superdec_cand_%j.err

set -euo pipefail
echo "=========================================="
echo "Job started: $(date) on $SLURMD_NODENAME"
echo "=========================================="

module load cuda/12.8
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"
conda activate superdec

cd /work/courses/3dv/team4/MultiGen3D/superdec

python scripts/run_candidates.py

echo "=========================================="
echo "Job completed: $(date). Runtime: ${SECONDS}s"
echo "=========================================="
