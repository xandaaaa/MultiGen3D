#!/bin/bash
#SBATCH --job-name=hard_cases
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/hard_cases_%j.out
#SBATCH --error=logs/hard_cases_%j.err

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

python experiments/hard_cases_experiment.py

echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="