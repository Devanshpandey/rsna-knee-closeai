#!/usr/bin/env bash
# Package best checkpoints + code into a Kaggle dataset for the offline submission notebook.
# Run on TACC login node (needs ~/.kaggle/kaggle.json and internet).
# Usage: bash package_weights.sh r2                  # packages results/r2_v1 + r2_v2
set -euo pipefail
TAG=${1:-r2}
W=/work/07880/devansh/lonestar/rsna_knee
PKG=/scratch/07880/devansh/rsna_knee/kaggle_pkg_${TAG}
rm -rf "$PKG" && mkdir -p "$PKG/ckpts" "$PKG/src"

for d in "$W"/results/${TAG}_v1 "$W"/results/${TAG}_v2; do
  name=$(basename "$d")
  for f in "$d"/fold*/best.pt; do
    fold=$(basename "$(dirname "$f")")
    cp "$f" "$PKG/ckpts/${name}_${fold}.pt"
  done
done
cp "$W"/scripts/training/model.py "$W"/scripts/inference/predict.py "$PKG/src/"
du -sh "$PKG"

cat > "$PKG/dataset-metadata.json" <<EOF
{
  "title": "rsna-knee-weights-${TAG}",
  "id": "$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")/rsna-knee-weights-${TAG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

export PATH=$HOME/.local/bin:$PATH
if kaggle datasets status "$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")/rsna-knee-weights-${TAG}" >/dev/null 2>&1; then
  kaggle datasets version -p "$PKG" -m "update $(date +%Y%m%d-%H%M)" --dir-mode zip
else
  kaggle datasets create -p "$PKG" --dir-mode zip
fi
echo "PACKAGED ${TAG}"
