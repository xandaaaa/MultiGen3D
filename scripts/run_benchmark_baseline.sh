#!/bin/bash
#SBATCH --job-name=bench_baseline
#SBATCH --account=ls_polle
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=4090:1
#SBATCH --output=logs/bench_baseline_%A_%a.out
#SBATCH --error=logs/bench_baseline_%A_%a.err

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID  Shape: $SLURM_ARRAY_TASK_ID / 19"
echo "Running on node: $SLURMD_NODENAME"
echo "=========================================="

module load eth_proxy
export HF_HOME=/cluster/project/cvg/students/xanyap/.cache/huggingface
source /cluster/project/cvg/students/xanyap/miniconda3/bin/activate base
conda activate spacecontrol
which python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python benchmark/run_benchmark.py \
    --approach baseline \
    --shape-idx all \
    --prompts-file benchmark/prompts_augmented.json \
    --results-root results \
    --steps 15 \
    --seed 42

echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
