#!/bin/bash
# One-shot setup for the RSNA knee project on a TACC GPU node.
# Usage: bash tacc_bootstrap.sh /path/to/workdir KGAT_yourtoken
set -euo pipefail
WORK=${1:?usage: tacc_bootstrap.sh WORKDIR KAGGLE_TOKEN}
TOKEN=${2:?usage: tacc_bootstrap.sh WORKDIR KAGGLE_TOKEN}

mkdir -p "$WORK" && cd "$WORK"
mkdir -p ~/.kaggle && printf '%s' "$TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN="$TOKEN"

python -m pip install -q --user kaggle timm lightgbm pandas scikit-learn pyarrow

# code + labels + csvs (small)
kaggle datasets download liaoyixu/rsna-knee-code -p code_ds --unzip
mkdir -p src/train src/reports data/raw data/processed runs
cp code_ds/{dataset,model,train_series,stage2}.py src/train/ 2>/dev/null || \
  for f in dataset model train_series stage2; do cp code_ds/$f.py src/train/ 2>/dev/null || true; done
cp code_ds/{rubric,extract_labels_server,make_labels,validate_extractor}.py src/reports/ 2>/dev/null || true
cp code_ds/train.csv code_ds/train_series.csv data/raw/
cp code_ds/labels_train_30b.parquet code_ds/fingerprints.csv data/processed/
cp code_ds/labels_train_30b_v5.parquet data/processed/ 2>/dev/null || echo "(v5 labels not published yet — use v4/30b)"

# volumes (46 GB)
kaggle datasets download liaoyixu/rsna-knee-npz320 -p npz_zip --unzip
mkdir -p data/processed/npz320_study
find npz_zip -name "*.npz" -exec ln -s {} data/processed/npz320_study/ \;
cp npz_zip/series_meta.parquet data/processed/npz320_study/

echo "=== ready. Example: ==="
echo "python src/train/train_series.py --npz-root data/processed/npz320_study \\"
echo "  --series-csv data/raw/train_series.csv --labels data/processed/labels_train_30b.parquet \\"
echo "  --train-csv data/raw/train.csv --fingerprints data/processed/fingerprints.csv \\"
echo "  --fold 0 --epochs 4 --bs 16 --k-slices 16 --workers 8 --out runs/v2_f0"
