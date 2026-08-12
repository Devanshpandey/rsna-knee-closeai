#!/usr/bin/env python
"""v6 pseudo-label targets: blend extractor v5 targets with ensemble OOF preds.

The 16-model OOF ensemble scores gold-58 ~0.915 vs the extractor's ~0.84, so
model predictions carry signal the report extraction misses. Blend in
probability space: target = w*extractor + (1-w)*oof. Studies without OOF
coverage keep their v5 targets. Row order (and thus make_folds) is preserved
from the v5-aligned parquet. Per-label _w confidence columns pass through.
"""
import argparse

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

ap = argparse.ArgumentParser()
ap.add_argument("--v5", default="data/processed/labels_train_30b_v5_aligned.parquet")
ap.add_argument("--oof", default="preds/all16_tta/oof_study_scores.csv")
ap.add_argument("--w-extractor", type=float, default=0.5)
ap.add_argument("--out", default="data/processed/labels_train_v6_pl.parquet")
args = ap.parse_args()

v5 = pd.read_parquet(args.v5)
oof = pd.read_csv(args.oof).set_index("StudyInstanceUID")
w = args.w_extractor

out = v5.copy()
covered = v5.StudyInstanceUID.isin(oof.index)
print(f"OOF coverage: {covered.sum()}/{len(v5)} studies; w_extractor={w}")
sub = out.loc[covered, "StudyInstanceUID"]
for c in LABELS:
    blended = w * out.loc[covered, c].values + (1 - w) * oof.loc[sub, c].values
    out.loc[covered, c] = blended.astype(np.float32)

# row order must match v5-aligned exactly (fold definition depends on it)
assert (out.StudyInstanceUID.values == v5.StudyInstanceUID.values).all()
out.to_parquet(args.out)
corr = np.mean([np.corrcoef(out[c], v5[c])[0, 1] for c in LABELS])
print(f"wrote {args.out}; mean corr(v6, v5) = {corr:.3f}")
for c in LABELS[:3] + LABELS[-2:]:
    print(f"  {c:18s} v5 mean {v5[c].mean():.3f} -> v6 mean {out[c].mean():.3f}")
