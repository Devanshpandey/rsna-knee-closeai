#!/usr/bin/env python3
"""Mirror-TTA evaluation: score gold + OOF for checkpoint families with
original + horizontally-mirrored inputs (medial/lateral output columns swapped back).

Usage (GPU node):
  CUDA_VISIBLE_DEVICES=0 python eval_tta.py --dirs r4_v2 r4_v3 \
      --proc-dir /scratch/07880/devansh/rsna_knee/data/proc320f140/train \
      --folds-csv /scratch/07880/devansh/rsna_knee/data/csv/train_folds_v4.csv --crop 288 --k-slices 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import LABELS, SWAP_IDX, KneeStudyDataset, collate
from model import KneeStudyModel

W = Path("/work/07880/devansh/lonestar/rsna_knee")


def predict(model, loader, device):
    outs, uids = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, meta, mask, _, uid in loader:
            x, meta, mask = x.to(device), meta.to(device), mask.to(device)
            p1 = model(x, meta, mask).float().sigmoid()
            p2 = model(torch.flip(x, dims=[-1]), meta, mask).float().sigmoid()[:, SWAP_IDX]
            outs.append(((p1 + p2) / 2).cpu().numpy())
            uids += uid
    return np.concatenate(outs), uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--folds-csv", required=True)
    ap.add_argument("--series-csv", default="/scratch/07880/devansh/rsna_knee/data/csv/train_series.csv")
    ap.add_argument("--train-csv", default="/scratch/07880/devansh/rsna_knee/data/csv/train.csv")
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--k-slices", type=int, default=20)
    ap.add_argument("--out-dir", default=str(W / "results/team_oof"))
    args = ap.parse_args()
    device = "cuda"
    train = pd.read_csv(args.train_csv).set_index("StudyInstanceUID")
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    gold_all, oof_all = {}, {}
    for d in args.dirs:
        gold_parts, oof_parts = [], []
        for f in range(4):
            ck = torch.load(W / f"results/{d}/fold{f}/best.pt", map_location="cpu", weights_only=False)
            model = KneeStudyModel(ck["backbone"], pretrained=False)
            model.load_state_dict(ck["model"]); model.eval().to(device)
            for split, parts in [("gold", gold_parts), ("val", oof_parts)]:
                ds = KneeStudyDataset(args.folds_csv, args.series_csv, args.proc_dir, f, split,
                                      k_slices=args.k_slices, crop=args.crop, train_aug=False,
                                      target_cols=[f"g_{l}" for l in LABELS] if split == "gold" else None)
                dl = DataLoader(ds, batch_size=2, num_workers=8, collate_fn=collate)
                p, uids = predict(model, dl, device)
                parts.append(pd.DataFrame(p, columns=LABELS).assign(StudyInstanceUID=uids).set_index("StudyInstanceUID"))
            print(f"{d} fold{f} done", flush=True)
        gold_m = sum(g[LABELS] for g in gold_parts) / len(gold_parts)
        oof_m = pd.concat(oof_parts)
        gold_all[d], oof_all[d] = gold_m, oof_m
        y = train.loc[gold_m.index, LABELS].astype(int)
        auc = np.mean([roc_auc_score(y[l], gold_m[l]) for l in LABELS])
        print(f"== {d} 4-fold + mirror-TTA gold AUC: {auc:.4f}", flush=True)
        gold_m.reset_index().to_csv(out / f"gold_tta_{d}.csv", index=False)
        oof_m.reset_index().to_csv(out / f"oof_tta_{d}.csv", index=False)

    if len(args.dirs) > 1:
        gm = sum(gold_all.values()) / len(gold_all)
        y = train.loc[gm.index, LABELS].astype(int)
        auc = np.mean([roc_auc_score(y[l], gm[l]) for l in LABELS])
        print(f"== combined {'+'.join(args.dirs)} + mirror-TTA gold AUC: {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
