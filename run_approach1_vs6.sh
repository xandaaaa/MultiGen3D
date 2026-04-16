#!/bin/bash
#SBATCH --job-name=a1_vs_a6
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/a1_vs_a6_%j.out
#SBATCH --error=logs/a1_vs_a6_%j.err

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "=========================================="

module load eth_proxy cuda/12.8
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"
conda activate spacecontrol
which python
python -c "import torch; print(torch.__version__)"

python experiments/approach1_vs_approach6.py

echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
