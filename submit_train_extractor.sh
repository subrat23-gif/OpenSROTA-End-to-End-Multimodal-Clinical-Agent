#!/bin/bash
#SBATCH --job-name=train_extractor
#SBATCH --account=<YOUR_ACCOUNT_NAME>
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /nas/vivek2/subrat/clinical_agent/
source .venv39/bin/activate

echo "Job started on ${hostname}"
echo "Job ID: ${SLURM_JOB_ID}"
nvidia-smi

# Train Qwen on the clean data
python scripts/train_ift_sft.py \
    --sft-dir outputs/ift_sft_data_clean_v3 \
    --out outputs/ift_lora_clean_v3 \
    --base Qwen/Qwen3-4B-Instruct-2507 \
    --epochs 2

echo "Training completed"