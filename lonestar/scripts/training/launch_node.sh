#!/usr/bin/env bash
# Launch training on the interactive GPU node (run ON the node, e.g. c317-013).
# Waits for preprocessing to finish, then runs folds 0-2 in parallel on GPUs 0-2.
# Fold 3 runs on GPU 0 after fold 0 completes.
set -uo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
cd /work/07880/devansh/lonestar/rsna_knee
LOG=logs

# wait for preprocessing to complete (marker: "done:" line in its log)
while ! grep -q "^done:" $LOG/preprocess_node_20260808.log 2>/dev/null; do
  echo "$(date) waiting for preprocessing..."; sleep 300
done
echo "$(date) preprocessing complete, starting training"

cd scripts/training
for f in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$f nohup python train.py --fold $f --out-dir /work/07880/devansh/lonestar/rsna_knee/results/img_v1 \
    --folds-csv /work/07880/devansh/lonestar/rsna_knee/data/csv/train_folds.csv \
    --series-csv /work/07880/devansh/lonestar/rsna_knee/data/csv/train_series.csv \
    > ../../$LOG/train_fold${f}.log 2>&1 &
  echo "fold $f -> GPU $f (pid $!)"
done
wait -n  # when the first fold finishes, start fold 3 on its GPU
CUDA_VISIBLE_DEVICES=0 nohup python train.py --fold 3 --out-dir /work/07880/devansh/lonestar/rsna_knee/results/img_v1 \
  --folds-csv /work/07880/devansh/lonestar/rsna_knee/data/csv/train_folds.csv \
  --series-csv /work/07880/devansh/lonestar/rsna_knee/data/csv/train_series.csv \
  > ../../$LOG/train_fold3.log 2>&1 &
echo "fold 3 -> GPU 0 (pid $!)"
wait
echo "ALL FOLDS DONE"
