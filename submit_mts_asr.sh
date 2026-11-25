#!/bin/bash
#SBATCH --job-name=mts_asr
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

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# 1. Whisper Confidence
python -u scripts/extract_asr_conf.py \
    --audio-dir data/audio_mts \
    --out-dir data/transcripts/mts_whisper \
    --manifest data/mts_dialog.jsonl \
    --asr large-v3-turbo

# 2. Omi Transcripts
python -u scripts/extract_asr_omi_cli.py \
    --audio-dir data/audio_mts \
    --out-dir data/transcripts/mts_omi \
    --model-path omi-med-stt-v1-q8_0.gguf \
    --runtime cpp

echo "ASR completed"