#!/usr/bin/env python
"""Per-label blend-weight optimization over family study-score OOF CSVs.

Each family CSV (from eval_ensemble.py --out) holds OOF study scores per label.
For every label we search simplex weights over families maximizing AUC on the
weak-label OOF pool. Nested check: fit weights on 3 folds, score the held-out
fold, rotate — the pooled held-out macro-AUC is an unbiased estimate of the
gain over uniform. Final weights are then fit on the full OOF pool.

Usage:
  python blend_opt.py --families cnxt=preds/new_cnxt_tta/oof_study_scores.csv \
      v2m=preds/v2m_tta/oof_study_scores.csv cnxb=preds/cnxb_tta/oof_study_scores.csv \
      --labels data/processed/labels_train_30b_v5_aligned.parquet \
      --train-csv data/raw/train.csv --fingerprints data/processed/fingerprints.csv \
      --out blend_weights.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_series import LABELS, make_folds


def simplex_grid(k, step=0.1):
    n = round(1 / step)
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True, help="name=csv pairs")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--fingerprints", required=True)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    names, mats = [], []
    for spec in args.families:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path).set_index("StudyInstanceUID").sort_index()
        names.append(name)
        mats.append(df)
    idx = mats[0].index
    for m in mats[1:]:
        assert m.index.equals(idx), "family CSVs cover different studies"
    S = np.stack([m[LABELS].values for m in mats])          # [F, N, 12]

    labels = pd.read_parquet(args.labels).dropna(subset=LABELS)
    gold_uids = set(pd.read_csv(args.train_csv).dropna(
        subset=[LABELS[0]]).StudyInstanceUID)
    labels = labels[~labels.StudyInstanceUID.isin(gold_uids)].reset_index(drop=True)
    folds = make_folds(labels, args.n_folds, args.fingerprints, args.train_csv)
    lab = labels.merge(folds, on="StudyInstanceUID").set_index("StudyInstanceUID")
    lab = lab.loc[idx]
    Y = (lab[LABELS].values >= 0.5).astype(int)             # [N, 12]
    fold_arr = lab["fold"].values
    F = len(names)
    grid = list(simplex_grid(F, args.step))
    print(f"{F} families {names}, {len(idx)} studies, {len(grid)} grid points")

    def fit_weights(mask):
        """Best per-label weights on studies where mask is True."""
        W = np.zeros((12, F))
        for li in range(12):
            y = Y[mask, li]
            if not 0 < y.sum() < len(y):
                W[li] = np.ones(F) / F
                continue
            best, bw = -1, None
            for w in grid:
                p = np.tensordot(w, S[:, mask, li], axes=1)
                a = roc_auc_score(y, p)
                if a > best:
                    best, bw = a, w
            W[li] = bw
        return W

    # ---- nested estimate: fit on 3 folds, score held-out fold ----
    held_uniform, held_weighted = [], []
    for f in range(args.n_folds):
        tr_m, va_m = fold_arr != f, fold_arr == f
        W = fit_weights(tr_m)
        for li in range(12):
            y = Y[va_m, li]
            if not 0 < y.sum() < len(y):
                continue
            pu = S[:, va_m, li].mean(0)
            pw = np.tensordot(W[li], S[:, va_m, li], axes=1)
            held_uniform.append(roc_auc_score(y, pu))
            held_weighted.append(roc_auc_score(y, pw))
    print(f"nested held-out macro-AUC: uniform {np.mean(held_uniform):.4f} "
          f"-> weighted {np.mean(held_weighted):.4f} "
          f"(gain {np.mean(held_weighted) - np.mean(held_uniform):+.4f})")

    # ---- final weights on full OOF ----
    W = fit_weights(np.ones(len(idx), bool))
    aucs_u, aucs_w = [], []
    for li in range(12):
        y = Y[:, li]
        aucs_u.append(roc_auc_score(y, S[:, :, li].mean(0)))
        aucs_w.append(roc_auc_score(y, np.tensordot(W[li], S[:, :, li], axes=1)))
    print(f"in-sample macro-AUC: uniform {np.mean(aucs_u):.4f} -> weighted "
          f"{np.mean(aucs_w):.4f}")
    for li, lname in enumerate(LABELS):
        print(f"  {lname:18s} {aucs_u[li]:.4f} -> {aucs_w[li]:.4f} | "
              + " ".join(f"{n}:{w:.2f}" for n, w in zip(names, W[li])))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"families": names, "weights": W.tolist()}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
