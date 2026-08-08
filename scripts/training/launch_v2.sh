#!/usr/bin/env bash
# Second architecture (tf_efficientnetv2_s) on GPUs 1-2: folds 0,1 then 2,3.
set -uo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache
cd /work/07880/devansh/lonestar/rsna_knee/scripts/training
R=/work/07880/devansh/lonestar/rsna_knee/results/img_v2
CS="--folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds.csv --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv"
L=/work/07880/devansh/lonestar/rsna_knee/logs

CUDA_VISIBLE_DEVICES=1 python train.py --fold 0 --backbone tf_efficientnetv2_s --out-dir $R $CS > $L/train_v2_fold0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --fold 1 --backbone tf_efficientnetv2_s --out-dir $R $CS > $L/train_v2_fold1.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 python train.py --fold 2 --backbone tf_efficientnetv2_s --out-dir $R $CS > $L/train_v2_fold2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --fold 3 --backbone tf_efficientnetv2_s --out-dir $R $CS > $L/train_v2_fold3.log 2>&1 &
wait
echo "V2 ALL FOLDS DONE"
