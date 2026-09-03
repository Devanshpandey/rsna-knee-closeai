#!/bin/bash
#SBATCH -J rsna_nspre
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A MLL
#SBATCH -o /work/07880/devansh/lonestar/rsna_knee/logs/%x_%j.log
set -euo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /work/07880/devansh/lonestar/rsna_knee/scripts/training
python train.py --fold 0 --backbone $BB --epochs 4 --lr 1.5e-4 --k-slices 20 \
  --batch $BATCH --accum $ACCUM --crop $CROP \
  --proc-dir /scratch/07880/devansh/rsna_knee/data/fastmri_proc384f140 \
  --folds-csv /scratch/07880/devansh/rsna_knee/data/csv/fastmri_folds_v2.csv \
  --series-csv /scratch/07880/devansh/rsna_knee/data/csv/fastmri_series.csv \
  --out-dir /work/07880/devansh/lonestar/rsna_knee/results/nspre_$TAG
# chain own fine-tunes (keeps queue under QOS submit limit)
for f in 0 1 2 3; do
  sbatch -d afterok:$SLURM_JOB_ID -J rsna_nsft_${TAG}_f$f     --export=ALL,FOLD=$f,BB=$BB,CROP=$CROP,TAG=$TAG,BATCH=$BATCH,ACCUM=$ACCUM     /work/07880/devansh/lonestar/rsna_knee/scripts/external/ft_job2.slurm
done
