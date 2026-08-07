#!/usr/bin/env bash
# Download RSNA Knee Abnormality Detection data.
# Prereqs: (1) joined the competition on kaggle.com (accepted rules),
#          (2) ~/.kaggle/kaggle.json API token with chmod 600.
# On TACC: run inside $SCRATCH (dataset is large); use a login node or idev.
set -euo pipefail

COMP=rsna-knee-abnormality-detection
DEST=${1:-./data/raw}
mkdir -p "$DEST"

# CSVs first (small) so EDA can start while DICOMs stream down
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  kaggle competitions download -c "$COMP" -f "$f" -p "$DEST" || true
done
cd "$DEST" && for z in *.zip; do [ -e "$z" ] && unzip -o "$z" && rm "$z"; done; cd - >/dev/null

# Full bundle (includes train_series/ + test_series/ DICOMs) — large, resumable via -o
kaggle competitions download -c "$COMP" -p "$DEST"
echo "Unzip manually once complete: unzip -n ${DEST}/${COMP}.zip -d ${DEST}"
