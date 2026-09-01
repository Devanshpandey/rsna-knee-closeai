#!/usr/bin/env bash
# Noisy-student campaign (fully chained). Run once on the LS6 login node.
# prep(3406484) -> pseudo-label(GPU job) -> 3 pretrains -> 12 fine-tunes.
set -uo pipefail
W=/work/07880/devansh/lonestar/rsna_knee
S=/scratch/07880/devansh/rsna_knee
cd $W

# 1) GPU job: pseudo-label fastMRI with the two v7@384 families
cat > scripts/external/pl_job.slurm <<'SL'
#!/bin/bash
#SBATCH -J rsna_pslabel
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A MLL
#SBATCH -o /work/07880/devansh/lonestar/rsna_knee/logs/%x_%j.log
set -euo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
cd /work/07880/devansh/lonestar/rsna_knee
python scripts/external/pseudolabel_fastmri.py \
  --series-csv /scratch/07880/devansh/rsna_knee/data/csv/fastmri_series.csv \
  --proc-dir /scratch/07880/devansh/rsna_knee/data/fastmri_proc384f140 \
  --ckpt-dirs results/v7c_effs384 results/v7b_srx384 \
  --out /scratch/07880/devansh/rsna_knee/data/csv/fastmri_folds.csv
SL

# 2) pretrain job template (fastMRI pseudo-labels, fold 0 = 5% sanity val)
cat > scripts/external/pre_job.slurm <<'SL'
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
  --batch 2 --accum 4 --crop $CROP \
  --proc-dir /scratch/07880/devansh/rsna_knee/data/fastmri_proc384f140 \
  --folds-csv /scratch/07880/devansh/rsna_knee/data/csv/fastmri_folds.csv \
  --series-csv /scratch/07880/devansh/rsna_knee/data/csv/fastmri_series.csv \
  --out-dir /work/07880/devansh/lonestar/rsna_knee/results/nspre_$TAG
SL

# 3) fine-tune job template (v7 targets, warm-started)
cat > scripts/external/ft_job.slurm <<'SL'
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
  --batch 2 --accum 4 --crop $CROP \
  --init-ckpt /work/07880/devansh/lonestar/rsna_knee/results/nspre_$TAG/fold0/best.pt \
  --proc-dir /scratch/07880/devansh/rsna_knee/data/proc384f140/train \
  --folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_v7.csv \
  --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv \
  --out-dir /work/07880/devansh/lonestar/rsna_knee/results/ns_$TAG
SL

PREP=3406484
PL=$(sbatch --parsable -d afterok:$PREP scripts/external/pl_job.slurm | tail -1)
echo "pseudo-label job: $PL (after prep $PREP)"
declare -A CROPS=( [effs384]=320 [srx384]=352 [cnxts384]=352 )
declare -A BBS=( [effs384]=tf_efficientnetv2_s [srx384]=seresnext50_32x4d [cnxts384]=convnext_small.fb_in22k_ft_in1k )
for tag in effs384 srx384 cnxts384; do
  PRE=$(sbatch --parsable -d afterok:$PL -J rsna_nspre_$tag \
        --export=ALL,BB=${BBS[$tag]},CROP=${CROPS[$tag]},TAG=$tag scripts/external/pre_job.slurm | tail -1)
  echo "pretrain $tag: $PRE"
  for f in 0 1 2 3; do
    sbatch --parsable -d afterok:$PRE -J rsna_nsft_${tag}_f$f \
      --export=ALL,FOLD=$f,BB=${BBS[$tag]},CROP=${CROPS[$tag]},TAG=$tag scripts/external/ft_job.slurm | tail -1
  done
done
echo "CAMPAIGN CHAINED"
