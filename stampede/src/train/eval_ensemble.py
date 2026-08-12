#!/usr/bin/env python
"""Offline ensemble evaluation on weak-val OOF and gold-58, mirroring the
submission kernel's aggregation exactly:

  per-series p = mean over models of sigmoid(logits); mirror TTA averages the
  original with the mirrored pass (medial<->lateral columns swapped back);
  study score = 0.5 * (mean + max) over series.

Weak-val OOF: each checkpoint predicts only its own fold's val studies
(fold read from ckpt args), pooled over folds -> macro-AUC vs weak labels.
Checkpoints sharing a fold are ensembled. Gold-58: every model predicts all
gold studies, ensembled.

Usage:
  python eval_ensemble.py --ckpts 'runs/cnxt_v5_f*/best.pt' weights_v1/best_f0.pt ... \
      --npz-root data/processed/npz320_study --series-csv data/raw/train_series.csv \
      --labels data/processed/labels_train_30b_v5_aligned.parquet \
      --train-csv data/raw/train.csv --fingerprints data/processed/fingerprints.csv \
      [--tta] [--out preds_dir]
"""
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import PLANES, slice_indices
from model import SeriesModel
from train_series import LABELS, make_folds, macro_auc

SWAP = list(range(12))
SWAP[2], SWAP[3], SWAP[4], SWAP[5] = 3, 2, 5, 4


class InferDataset(Dataset):
    """Deterministic per-series inference items: original + mirrored triplets."""

    def __init__(self, series_df, npz_root, k, tta):
        self.df = series_df.reset_index(drop=True)
        self.root = Path(npz_root)
        self.k = k
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def _triplets(self, vol):
        n = vol.shape[0]
        centers = slice_indices(n, self.k, jitter=False)
        lo = np.clip(centers - 1, 0, n - 1)
        hi = np.clip(centers + 1, 0, n - 1)
        x = np.stack([vol[lo], vol[centers], vol[hi]], axis=1).astype(np.float32) / 255.0
        pos = (centers / max(n - 1, 1)).astype(np.float32)
        return x, pos

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = self.root / r.StudyInstanceUID / f"{r.SeriesInstanceUID}.npz"
        if p.exists():
            with np.load(p) as z:
                vol = z["vol"]
        else:
            with np.load(self.root / f"{r.StudyInstanceUID}.npz") as z:
                vol = z[f"vol_{r.SeriesInstanceUID}"]
        plane = PLANES.get(r.Anatomical_Plane, 0)
        x0, pos = self._triplets(vol)
        views = [x0]
        if self.tta:
            mvol = np.ascontiguousarray(vol[::-1] if plane == 0 else vol[:, :, ::-1])
            x1, _ = self._triplets(mvol)
            views.append(x1)
        meta = np.array([plane, int(r.Fluid_Sensitive), int(r.Fat_Suppression)],
                        dtype=np.int64)
        return (torch.from_numpy(np.stack(views)), torch.from_numpy(meta),
                torch.from_numpy(pos), i)


@torch.no_grad()
def predict_series(models_k, series_df, npz_root, dev, tta, bs=8, workers=8):
    """models_k: list of (model, k) or (model, k, root). Returns ensemble probs."""
    out = np.zeros((len(series_df), 12), dtype=np.float64)
    by_k = defaultdict(list)
    for mk in models_k:
        m, k, root = mk if len(mk) == 3 else (*mk, npz_root)
        by_k[(root, k)].append(m)
    for (root, k), models in by_k.items():
        ds = InferDataset(series_df, root, k, tta)
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
        for xv, meta, pos, idx in dl:
            B, V = xv.shape[:2]
            x = xv.flatten(0, 1).to(dev)               # [B*V, K, 3, H, W]
            mt = meta.repeat_interleave(V, 0).to(dev)
            ps = pos.repeat_interleave(V, 0).to(dev)
            acc = np.zeros((B, 12), dtype=np.float64)
            for m in models:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(dev == "cuda")):
                    lg, _ = m(x, mt, ps)
                p = torch.sigmoid(lg).float().cpu().numpy().reshape(B, V, 12)
                pv = p[:, 0]
                if V == 2:
                    pv = 0.5 * (p[:, 0] + p[:, 1][:, SWAP])
                acc += pv
            out[idx.numpy()] += acc          # sum over models; normalized below
    return out / len(models_k)               # plain mean over ALL models (kernel semantics)


def study_agg(series_df, probs):
    df = series_df.reset_index(drop=True).copy()
    for i, c in enumerate(LABELS):
        df[c] = probs[:, i]
    gp = df.groupby("StudyInstanceUID")[LABELS]
    return 0.5 * (gp.mean() + gp.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--npz-root-384", default="",
                    help="root for ckpts trained on 384px npz (detected via args.npz_root)")
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--fingerprints", required=True)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--k-override", type=int, default=0,
                    help="use this K at inference instead of each ckpt's k_slices")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save-series", action="store_true",
                    help="dump per-series OOF ensemble preds to the out dir")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    paths = []
    for pat in args.ckpts:
        paths += sorted(glob.glob(pat))
    fold_models, all_models = defaultdict(list), []
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        a = sd.get("args") or {}
        bb = a.get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k")
        k = args.k_override or int(a.get("k_slices", 16))
        fold = int(a.get("fold", -1))
        root = (args.npz_root_384 if ("384" in str(a.get("npz_root", ""))
                                      and args.npz_root_384) else args.npz_root)
        m = SeriesModel(bb, pretrained=False)
        m.load_state_dict(sd["model"])
        m.to(dev).eval()
        fold_models[fold].append((m, k, root))
        all_models.append((m, k, root))
        print(f"{p}: fold={fold} {bb} K={k} root={Path(root).name} "
              f"weak-val={sd.get('auc', float('nan')):.4f} "
              f"gold={sd.get('gold_auc', float('nan')):.4f}")

    labels = pd.read_parquet(args.labels).dropna(subset=LABELS)
    tcsv = pd.read_csv(args.train_csv)
    gold = tcsv.dropna(subset=[LABELS[0]])[["StudyInstanceUID"] + LABELS]
    gold_uids = set(gold.StudyInstanceUID)
    labels = labels[~labels.StudyInstanceUID.isin(gold_uids)].reset_index(drop=True)
    folds = make_folds(labels, args.n_folds, args.fingerprints, args.train_csv)
    labels = labels.merge(folds, on="StudyInstanceUID")
    series = pd.read_csv(args.series_csv)
    roots = [args.npz_root] + ([args.npz_root_384] if args.npz_root_384 else [])
    for r in roots:
        meta_p = Path(r) / "series_meta.parquet"
        if meta_p.exists():
            sm = pd.read_parquet(meta_p)
            have = set(zip(sm.StudyInstanceUID, sm.SeriesInstanceUID))
            series = series[[(s, u) in have for s, u in
                             zip(series.StudyInstanceUID, series.SeriesInstanceUID)]]

    # ---- weak-val OOF: per fold, its models predict its val studies ----
    oof_scores, oof_uids, series_dumps = [], [], []
    for f in sorted(k for k in fold_models if k >= 0):
        va = labels[labels.fold == f]
        sva = series[series.StudyInstanceUID.isin(set(va.StudyInstanceUID))]
        probs = predict_series(fold_models[f], sva, args.npz_root, dev, args.tta,
                               args.bs, args.workers)
        if args.save_series:
            d = sva.reset_index(drop=True).copy()
            for i, c in enumerate(LABELS):
                d[c] = probs[:, i]
            d["fold"] = f
            series_dumps.append(d)
        st = study_agg(sva, probs)
        oof_scores.append(st)
        oof_uids.append(va.set_index("StudyInstanceUID")[LABELS])
        y = (va.set_index("StudyInstanceUID").loc[st.index][LABELS].values >= 0.5)
        auc, _ = macro_auc(y.astype(float), st.values)
        print(f"fold {f}: weak-val macro-AUC {auc:.4f} ({len(st)} studies)")
    if oof_scores:
        st_all = pd.concat(oof_scores)
        y_all = pd.concat(oof_uids).loc[st_all.index].values >= 0.5
        auc, per = macro_auc(y_all.astype(float), st_all.values)
        print(f"pooled OOF macro-AUC {auc:.4f} ({len(st_all)} studies)")
        print("  per-label:", " ".join(f"{LABELS[i][:9]}:{a:.3f}" for i, a in per.items()))
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            st_all.to_csv(Path(args.out) / "oof_study_scores.csv")
            if series_dumps:
                pd.concat(series_dumps).to_parquet(
                    Path(args.out) / "oof_series_preds.parquet")

    # ---- gold-58: all models ensemble ----
    sg = series[series.StudyInstanceUID.isin(gold_uids)]
    probs = predict_series(all_models, sg, args.npz_root, dev, args.tta,
                           args.bs, args.workers)
    st = study_agg(sg, probs)
    y = gold.set_index("StudyInstanceUID").loc[st.index][LABELS].values
    auc, per = macro_auc(y.astype(float), st.values)
    print(f"gold-58 ensemble macro-AUC {auc:.4f} (sigma~0.0125)")
    print("  per-label:", " ".join(f"{LABELS[i][:9]}:{a:.3f}" for i, a in per.items()))
    if args.out:
        st.to_csv(Path(args.out) / "gold_study_scores.csv")


if __name__ == "__main__":
    main()
