#!/usr/bin/env python
"""Learned study-level aggregation over per-series ensemble preds.

Baseline: study = 0.5*(mean+max) over series probs (current kernel).
Candidates: per-label LightGBM / logistic regression over per-study features
built from series preds + acquisition metadata. Honest eval: per-fold models
(train folds != f, predict fold f), pooled OOF macro-AUC vs baseline.

Targets are the EXTRACTOR labels (v5-aligned), never pseudo-labels.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_series import LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--series-preds", default="preds/all16_series/oof_series_preds.parquet")
ap.add_argument("--labels", default="data/processed/labels_train_30b_v5_aligned.parquet")
ap.add_argument("--out", default="")
args = ap.parse_args()

sp = pd.read_parquet(args.series_preds)
lab = pd.read_parquet(args.labels).set_index("StudyInstanceUID")

FEATS = ["mean", "max", "min", "std", "top2", "n", "fluid_mean", "nonfluid_mean",
         "sag_mean", "sag_max", "cor_mean", "cor_max", "axi_mean", "axi_max"]

def study_features_all(g):
    """All 12 labels at once -> [12, n_feats]."""
    p = g[LABELS].values                                    # [n_series, 12]
    fs = g["Fluid_Sensitive"].values.astype(bool)
    srt = np.sort(p, axis=0)
    out = [p.mean(0), p.max(0), p.min(0),
           p.std(0) if len(p) > 1 else np.zeros(12),
           srt[-2:].mean(0), np.full(12, len(p)),
           p[fs].mean(0) if fs.any() else p.mean(0),
           p[~fs].mean(0) if (~fs).any() else p.mean(0)]
    for plane in ("Sagittal", "Coronal", "Axial"):
        m = (g["Anatomical_Plane"] == plane).values
        out.append(p[m].mean(0) if m.any() else p.mean(0))
        out.append(p[m].max(0) if m.any() else p.max(0))
    return np.stack(out, axis=1)                            # [12, n_feats]

groups = list(sp.groupby("StudyInstanceUID", sort=True))
uids = [u for u, _ in groups]
folds = np.array([g["fold"].iloc[0] for _, g in groups])
Y = (lab.loc[uids, LABELS].values >= 0.5).astype(int)
FA = np.stack([study_features_all(g) for _, g in groups])   # [N, 12, n_feats]
print(f"features built: {FA.shape}")

base_aucs, lgb_aucs, lr_aucs = [], [], []
try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False
    print("lightgbm unavailable — skipping LGBM candidate")

for li, c in enumerate(LABELS):
    X = FA[:, li, :]
    y = Y[:, li]
    base = 0.5 * (X[:, FEATS.index("mean")] + X[:, FEATS.index("max")])
    base_aucs.append(roc_auc_score(y, base))
    oof_lgb = np.zeros(len(y))
    oof_lr = np.zeros(len(y))
    for f in sorted(set(folds)):
        tr, va = folds != f, folds == f
        if HAVE_LGB:
            m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15,
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, verbose=-1, n_jobs=4)
            m.fit(X[tr], y[tr])
            oof_lgb[va] = m.predict_proba(X[va])[:, 1]
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(X[tr], y[tr])
        oof_lr[va] = lr.predict_proba(X[va])[:, 1]
    if HAVE_LGB:
        lgb_aucs.append(roc_auc_score(y, oof_lgb))
    lr_aucs.append(roc_auc_score(y, oof_lr))
    msg = f"{c:18s} base {base_aucs[-1]:.4f} lr {lr_aucs[-1]:.4f}"
    if HAVE_LGB:
        msg += f" lgb {lgb_aucs[-1]:.4f}"
    print(msg)

print(f"\nmacro: base {np.mean(base_aucs):.4f} | lr {np.mean(lr_aucs):.4f}"
      + (f" | lgb {np.mean(lgb_aucs):.4f}" if HAVE_LGB else ""))
