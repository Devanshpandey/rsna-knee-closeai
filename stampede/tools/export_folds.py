#!/usr/bin/env python
"""Export the canonical team fold assignment to shared/folds/.

Writes folds_v<N>.csv (StudyInstanceUID, fold), folds_v<N>.hash (the
fold-hash), and copies fingerprints.csv alongside so the assignment is
independently reproducible.

The fold-hash is:
    pd.util.hash_pandas_object(
        folds.sort_values("StudyInstanceUID").reset_index(drop=True)
    ).sum()
computed over the two columns (StudyInstanceUID, fold).

IMPORTANT: recomputing folds from scratch proved environment-sensitive
(GroupKFold tie-breaking shifted 30/1088 studies when a library changed,
2026-08-11) — the exported CSV is therefore the AUTHORITATIVE assignment;
liaoyixu's make_folds() loads a pinned copy of it and only recomputes as a
fallback. Align OOF to the CSV, never to a recomputation.

These are the union-find folds: GroupKFold over scanner-fingerprint groups
merged with byte-identical-report clusters, seed 42 — treat as a versioned
snapshot; if the labels parquet row order or study set ever changes, export
the next version rather than overwriting.

Usage (from liaoyixu's working dir, where data/ lives):
  python stampede/tools/export_folds.py \
      --labels data/processed/labels_train_30b.parquet \
      --train-csv data/raw/train.csv \
      --fingerprints data/processed/fingerprints.csv \
      --out-dir <repo>/shared/folds --version v4
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))
from train_series import LABELS, make_folds

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True, help="labels parquet defining the study set")
ap.add_argument("--train-csv", required=True)
ap.add_argument("--fingerprints", required=True)
ap.add_argument("--n-folds", type=int, default=4)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--version", default="v4")
args = ap.parse_args()

labels = pd.read_parquet(args.labels).dropna(subset=LABELS)
gold_uids = set(pd.read_csv(args.train_csv).dropna(
    subset=[LABELS[0]]).StudyInstanceUID)
labels = labels[~labels.StudyInstanceUID.isin(gold_uids)].reset_index(drop=True)
folds = make_folds(labels, args.n_folds, args.fingerprints, args.train_csv)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
csv_p = out / f"folds_{args.version}.csv"
if csv_p.exists():
    raise SystemExit(f"{csv_p} exists — shared/ files are versioned, never "
                     "overwritten. Bump --version.")
folds[["StudyInstanceUID", "fold"]].to_csv(csv_p, index=False)

h = pd.util.hash_pandas_object(
    folds[["StudyInstanceUID", "fold"]].sort_values("StudyInstanceUID")
    .reset_index(drop=True)).sum()
(out / f"folds_{args.version}.hash").write_text(str(h) + "\n")
shutil.copy(args.fingerprints, out / f"fingerprints_{args.version}.csv")

print(f"wrote {csv_p} ({len(folds)} studies, "
      f"sizes {folds.fold.value_counts().sort_index().tolist()})")
print(f"fold-hash {h}")
