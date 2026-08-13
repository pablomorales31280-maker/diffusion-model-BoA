#!/bin/bash
#SBATCH --job-name=build_dataset_test
#SBATCH --partition=dev_cpuonly
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa/logs/build_dataset_test_%J.out
#SBATCH --error=/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa/logs/build_dataset_test_%J.err

set -euo pipefail

PROJECT_DIR="/hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa"

cd "${PROJECT_DIR}"
mkdir -p logs

source /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/etc/profile.d/conda.sh
conda activate /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/software/miniconda3/envs/dmboa

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -u data/create_gopro_lmdb.py \
    --dataset-root /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa/dataset \
    --split test \
    --output /hkfs/work/workspace/scratch/ma_pmorales-nafnet_try/dm_boa/dataset_lmdb/gopro_test.lmdb \
    --crop-size 128 \
    --overwrite