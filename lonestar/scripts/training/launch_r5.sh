#!/usr/bin/env bash
# r5 diversity families on the jupyter node: cnxt_small@288 (GPUs 1,2), maxvit@224 (GPU 0 after TTA eval).
set -uo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/work/07880/devansh/lonestar/rsna_knee
cd $W/scripts/training
CS="--folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_v4_r2.csv --series-csv /scratch/07880/devansh/rsna_knee/data/csv/train_series.csv --proc-dir /scratch/07880/devansh/rsna_knee/data/proc320f140/train"
run() { CUDA_VISIBLE_DEVICES=$1 python train.py --fold $2 --backbone $3 --crop $4 --k-slices 20 --lr 1.5e-4 --batch 2 --accum 4 --epochs 8 --out-dir $W/results/$5 $CS > $W/logs/train_$5_fold$2.log 2>&1; }

case "${1:?gpu-role}" in
  g1) run 1 0 convnext_small.fb_in22k_ft_in1k 288 r5_cnxts; run 1 1 convnext_small.fb_in22k_ft_in1k 288 r5_cnxts ;;
  g2) run 2 2 convnext_small.fb_in22k_ft_in1k 288 r5_cnxts; run 2 3 convnext_small.fb_in22k_ft_in1k 288 r5_cnxts ;;
  g0) while ! grep -q "combined" $W/logs/eval_tta.log 2>/dev/null; do sleep 300; done
      for f in 0 1 2 3; do run 0 $f maxvit_tiny_tf_224.in1k 224 r5_maxvit; done ;;
esac
echo "R5 ${1} DONE"
