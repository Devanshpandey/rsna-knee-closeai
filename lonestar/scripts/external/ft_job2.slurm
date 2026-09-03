#!/bin/bash
#SBATCH -J rsna_nsft
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 07:00:00
#SBATCH -A MLL
#SBATCH -o /work/07880/devansh/lonestar/rsna_knee/logs/%x_%j.log
set -euo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/07880/devansh/lonestar/rsna_knee/scripts/training
python train.py --fold $FOLD --backbone $BB --epochs 8 --lr 1e-4 --k-slices 20 \
  --batch $BATCH --accum $ACCUM --crop $CROP \
  --init-ckpt /work/07880/devansh/lonestar/rsna_knee/results/nspre_$TAG/fold0/best.pt \
  --proc-dir /scratch/07880/devansh/rsna_knee/data/proc384f140/train \
  --folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_v7.csv \
  --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv \
  --out-dir /work/07880/devansh/lonestar/rsna_knee/results/ns_$TAG
