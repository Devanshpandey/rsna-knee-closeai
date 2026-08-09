#!/usr/bin/env bash
# Third backbone (seresnext50_32x4d) on r2 targets: folds 0-2 parallel, then fold 3.
set -uo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache
W=/work/07880/devansh/lonestar/rsna_knee
cd $W/scripts/training
CS="--folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_r2.csv --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv"
run() { CUDA_VISIBLE_DEVICES=$1 python train.py --fold $2 --backbone seresnext50_32x4d --out-dir $W/results/r2_v3 $CS > $W/logs/train_r2_v3_fold$2.log 2>&1; }
run 0 0 & run 1 1 & run 2 2 & wait
run 0 3 & wait
echo "V3 ALL DONE"
