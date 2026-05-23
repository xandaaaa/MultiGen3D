#!/bin/bash
#SBATCH --job-name=decode_composite
#SBATCH --account=ls_polle
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=4090:1
#SBATCH --output=logs/decode_composite_%j.out
#SBATCH --error=logs/decode_composite_%j.err

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "=========================================="

module load eth_proxy
export HF_HOME=/cluster/project/cvg/students/xanyap/.cache/huggingface
source /cluster/project/cvg/students/xanyap/miniconda3/bin/activate base
conda activate spacecontrol
which python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /cluster/scratch/xanyap/MultiGen3D
python benchmark/run_benchmark.py \
    --approach decode_composite \
    --prompts-file benchmark/prompts_augmented.json \
    --results-root results \
    --resolution 512 \
    --seed 42

echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
