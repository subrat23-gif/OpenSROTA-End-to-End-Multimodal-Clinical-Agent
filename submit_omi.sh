#!/bin/bash
#SBATCH --job-name=omi_asr
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
source .venv39/bin/activate

echo "Job started on ${hostname}"
echo "Job ID: ${SLURM_JOB_ID}"
nvidia-smi

# Run the Omi extraction script
python scripts/extract_asr_omi_cli.py \
    --audio-dir data/audio_ift/val_clean \
    --out-dir data/transcripts/val_omi_clean \
    --model-path omi-med-stt-v1-q8_0.gguf \
    --runtime cpp

echo "Omi Extraction completed"