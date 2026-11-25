#!/bin/bash
#SBATCH --job-name=conformal
#SBATCH --account=vivek2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /nas/vivek2/subrat/clinical_agent/
source .venv39/bin/activate

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# 1. Format the data (NO --manifest flag!)
python -u scripts/format_conformal_data.py \
    --pred outputs/suite/val/lora_clean_v4.pred.jsonl \
    --out outputs/phase0/conformal_data_final.jsonl

# 2. Run the Controller
python -u scripts/conformal_risk_hybrid.py \
    --data outputs/phase0/conformal_data_final.jsonl \
    --whisper-dir data/transcripts/val_conf_clean \
    --omi-dir data/transcripts/val_omi_clean

echo "Conformal Controller completed"