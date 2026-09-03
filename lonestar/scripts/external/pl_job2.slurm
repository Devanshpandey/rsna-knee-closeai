#!/bin/bash
#SBATCH -J rsna_pslabel2
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
python scripts/external/pseudolabel_fastmri.py   --series-csv /scratch/07880/devansh/rsna_knee/data/csv/fastmri_series.csv   --proc-dir /scratch/07880/devansh/rsna_knee/data/fastmri_proc384f140   --ckpt-dirs results/ns_effs384 results/ns_srx384   --out /scratch/07880/devansh/rsna_knee/data/csv/fastmri_folds_v2.csv
