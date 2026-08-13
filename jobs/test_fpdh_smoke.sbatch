#!/bin/bash
#SBATCH --job-name=dmboa_fpdh_smoke
#SBATCH --partition=dev_accelerated-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa/logs/fpdh_smoke_%J.out
#SBATCH --error=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa/logs/fpdh_smoke_%J.err

set -euo pipefail

PROJECT_DIR="/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa"

cd "${PROJECT_DIR}"
mkdir -p logs

source /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/etc/profile.d/conda.sh
conda activate /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/envs/dmboa

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Début: $(date)"

nvidia-smi

# Enregistrement périodique de l'utilisation mémoire GPU.
GPU_LOG="${PROJECT_DIR}/logs/fpdh_gpu_${SLURM_JOB_ID}.csv"

nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv \
    --loop=5 \
    > "${GPU_LOG}" &

MONITOR_PID=$!

cleanup() {
    kill "${MONITOR_PID}" 2>/dev/null || true
}

trap cleanup EXIT

python -u train.py \
    -c config/sr_sr3_fpdh_smoke.json \
    -p train \
    -gpu 0

echo "Fin: $(date)"
nvidia-smi