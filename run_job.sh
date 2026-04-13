#!/bin/bash
#SBATCH --job-name=openfrontier_mem
#SBATCH --account=ls_polle
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=4090:1
#SBATCH --output=logs/openfrontier_mem_%A_%a.out
#SBATCH --error=logs/openfrontier_mem_%A_%a.err


echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID  Array task: $SUBSET_INDEX / $NUM_SUBSETS"
echo "Running on node: $SLURMD_NODENAME"
echo "=========================================="

module load eth_proxy
export HF_HOME=/cluster/project/cvg/students/xanyap/.cache/huggingface
conda activate spacecontrol
which python
python -c "import torch; print(torch.__version__)"

# Run script
python experiments/approach1_experiment.py

echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
