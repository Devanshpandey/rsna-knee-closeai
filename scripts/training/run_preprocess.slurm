#!/bin/bash
#SBATCH -J rsna_preprocess
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A MLL
#SBATCH -o /work/07880/devansh/lonestar/rsna_knee/logs/%x_%j.log

set -euo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
cd /work/07880/devansh/lonestar/rsna_knee

python scripts/training/preprocess_volumes.py \
  --dicom-root data/raw/train_series \
  --series-csv data/csv/train_series.csv \
  --out-dir /scratch/07880/devansh/rsna_knee/data/proc/train \
  --size 256 --workers 64
