#!/bin/bash
#SBATCH --job-name=sdec_3new
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/sdec_3new_%j.out
#SBATCH --error=logs/sdec_3new_%j.err

set -euo pipefail
echo "Job started: $(date) on $SLURMD_NODENAME"

module load cuda/12.8
eval "$(/work/courses/3dv/team4/env_root/miniconda3/bin/conda shell.bash hook)"
conda activate superdec

cd /work/courses/3dv/team4/MultiGen3D/superdec

# telephone, display, loudspeaker
export ONLY_SYNSETS="04401088,03211117,03691459"
python scripts/run_candidates.py

echo "Done: $(date). Runtime: ${SECONDS}s"
