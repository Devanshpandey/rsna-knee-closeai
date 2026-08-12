#!/bin/bash
# Launch 4 training folds, one per H100 GPU.
# Usage: bash launch_folds.sh <run_prefix> [extra train_series.py args...]
set -euo pipefail
cd /scratch/09059/xliaoyi/kaggle
PREFIX=${1:?usage: launch_folds.sh RUN_PREFIX [extra args]}
shift
for F in 0 1 2 3; do
  OUT="runs/${PREFIX}_f${F}"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=$F nohup python3 src/train/train_series.py \
    --npz-root data/processed/npz320_study \
    --series-csv data/raw/train_series.csv \
    --labels data/processed/labels_train_30b_v5_aligned.parquet \
    --train-csv data/raw/train.csv \
    --fingerprints data/processed/fingerprints.csv \
    --fold $F --n-folds 4 \
    "$@" \
    --out "$OUT" > "$OUT/train.log" 2>&1 &
  echo "fold $F -> $OUT (pid $!)"
done
