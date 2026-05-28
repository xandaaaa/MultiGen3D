#!/bin/bash
#SBATCH --job-name=clip_score
#SBATCH --account=ls_polle
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=4090:1
#SBATCH --output=logs/clip_%j.out
#SBATCH --error=logs/clip_%j.err

module load eth_proxy
export HF_HOME=/cluster/project/cvg/students/xanyap/.cache/huggingface
source /cluster/project/cvg/students/xanyap/miniconda3/bin/activate base
conda activate spacecontrol

cd /cluster/scratch/xanyap/MultiGen3D
python benchmark/clip_score.py \
    --benchmark benchmark/prompts_augmented.json \
    --results-root results \
    --approaches baseline multigen \
    --output results/clip_scores.json
