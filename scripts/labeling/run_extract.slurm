#!/bin/bash
#SBATCH -J rsna_label_extract
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A MLL
#SBATCH -o /work/07880/devansh/lonestar/rsna_knee/logs/%x_%j.log

set -euo pipefail
source /work/07880/devansh/anaconda3/etc/profile.d/conda.sh
conda activate rsna_env
export HF_HOME=/work/07880/devansh/hf_cache
cd /work/07880/devansh/lonestar/rsna_knee

MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
TAG=${TAG:-14b}

python scripts/labeling/extract_labels_llm.py \
  --model "$MODEL" \
  --train-csv data/csv/train.csv \
  --output results/labels_llm_${TAG}.csv

python scripts/labeling/validate_extraction.py \
  --pred results/labels_llm_${TAG}.csv \
  --train-csv data/csv/train.csv \
  --out results/validation_llm_${TAG}.csv
