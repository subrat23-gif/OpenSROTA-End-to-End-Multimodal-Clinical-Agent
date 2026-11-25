#!/bin/bash
#SBATCH --job-name=extractor_infer
#SBATCH --account=vivek2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /nas/vivek2/subrat/clinical_agent/

# Activate virtual environment
source .venv39/bin/activate

# Verify environment
echo "Job started on $(hostname)"   # <--- FIXED HERE
echo "Job ID: ${SLURM_JOB_ID}"
nvidia-smi

# Run with unbuffered output
python -u scripts/run_extractor_on_new_audio.py

echo "Inference completed"