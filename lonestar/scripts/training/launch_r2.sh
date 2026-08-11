#!/usr/bin/env bash
# Round-2 chain: wait for v2 -> build blended targets -> retrain both archs on 3 GPUs.
set -uo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache
W=/work/07880/devansh/lonestar/rsna_knee
cd $W

while ! grep -q "V2 ALL FOLDS DONE" logs/launch_v2.log 2>/dev/null; do sleep 300; done
echo "$(date) v2 done -> building round-2 targets"

python scripts/training/make_round2_targets.py \
  --model-dirs results/img_v1 results/img_v2 \
  --folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds.csv \
  --train-csv /scratch/07880/devansh/rsna_knee/data/csv/train.csv \
  --expert-csv results/expert_review.csv \
  --out /scratch/07880/devansh/rsna_knee/data/csv/train_folds_r2.csv || { echo "target build FAILED"; exit 1; }

cd scripts/training
CS="--folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_r2.csv --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv"
run() { CUDA_VISIBLE_DEVICES=$1 python train.py --fold $2 --backbone $3 --out-dir $W/results/$4 $CS > $W/logs/train_$4_fold$2.log 2>&1; }

# wave 1
run 0 0 convnext_tiny r2_v1 & run 1 1 convnext_tiny r2_v1 & run 2 2 convnext_tiny r2_v1 & wait
# wave 2
run 0 3 convnext_tiny r2_v1 & run 1 0 tf_efficientnetv2_s r2_v2 & run 2 1 tf_efficientnetv2_s r2_v2 & wait
# wave 3
run 0 2 tf_efficientnetv2_s r2_v2 & run 1 3 tf_efficientnetv2_s r2_v2 & wait
echo "R2 ALL DONE"
