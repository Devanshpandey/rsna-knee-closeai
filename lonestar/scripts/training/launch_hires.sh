#!/usr/bin/env bash
# Round-3 hi-res: wait for 320px preprocessing + v3 -> EfficientNetV2-S at crop 288, 4 folds.
set -uo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache
W=/work/07880/devansh/lonestar/rsna_knee
cd $W
while ! grep -q "^done:" logs/preprocess_320.log 2>/dev/null; do sleep 120; done
while ! grep -q "V3 ALL DONE" logs/launch_v3.log 2>/dev/null; do sleep 300; done
echo "$(date) prerequisites met -> hi-res round"
cd scripts/training
CS="--folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_r2.csv --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv --proc-dir /scratch/07880/devansh/rsna_knee/data/proc320/train"
run() { CUDA_VISIBLE_DEVICES=$1 python train.py --fold $2 --backbone tf_efficientnetv2_s --crop 288 --batch 2 --accum 4 --out-dir $W/results/r3_hires $CS > $W/logs/train_hires_fold$2.log 2>&1; }
run 0 0 & run 1 1 & run 2 2 & wait
run 0 3 & wait
echo "HIRES ALL DONE"
