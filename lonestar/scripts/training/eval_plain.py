#!/usr/bin/env python3
"""No-TTA gold + OOF eval for checkpoint families with per-family crops.

Usage (GPU node):
  python eval_plain.py --dirs ns_effs384 ns_srx384 ns_cnxts384 --crops 320 352 320 \
      --folds-csv .../train_folds_v7.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import LABELS, KneeStudyDataset, collate
from model import KneeStudyModel

W = Path("/work/07880/devansh/lonestar/rsna_knee")


def predict(model, loader, device):
    outs, uids = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, meta, mask, _, uid in loader:
            x, meta, mask = x.to(device), meta.to(device), mask.to(device)
            outs.append(model(x, meta, mask).float().sigmoid().cpu().numpy())
            uids += uid
    return np.concatenate(outs), uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--crops", nargs="+", type=int, required=True)
    ap.add_argument("--proc-dir", default="/scratch/07880/devansh/rsna_knee/data/proc384f140/train")
    ap.add_argument("--folds-csv", default="/scratch/07880/devansh/rsna_knee/data/csv/train_folds_v7.csv")
    ap.add_argument("--series-csv", default="/scratch/07880/devansh/rsna_knee/data/csv/train_series.csv")
    ap.add_argument("--train-csv", default="/scratch/07880/devansh/rsna_knee/data/csv/train.csv")
    ap.add_argument("--k-slices", type=int, default=20)
    ap.add_argument("--out-dir", default=str(W / "results/team_oof"))
    args = ap.parse_args()
    assert len(args.dirs) == len(args.crops)
    device = "cuda"
    train = pd.read_csv(args.train_csv).set_index("StudyInstanceUID")
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    gold_all = {}
    for d, crop in zip(args.dirs, args.crops):
        gold_parts, oof_parts = [], []
        for f in range(4):
            ck = torch.load(W / f"results/{d}/fold{f}/best.pt", map_location="cpu", weights_only=False)
            model = KneeStudyModel(ck["backbone"], pretrained=False)
            model.load_state_dict(ck["model"]); model.eval().to(device)
            for split, parts in [("gold", gold_parts), ("val", oof_parts)]:
                ds = KneeStudyDataset(args.folds_csv, args.series_csv, args.proc_dir, f, split,
                                      k_slices=args.k_slices, crop=crop, train_aug=False,
                                      target_cols=[f"g_{l}" for l in LABELS] if split == "gold" else None)
                dl = DataLoader(ds, batch_size=2, num_workers=8, collate_fn=collate)
                p, uids = predict(model, dl, device)
                parts.append(pd.DataFrame(p, columns=LABELS).assign(StudyInstanceUID=uids).set_index("StudyInstanceUID"))
            print(f"{d} fold{f} done", flush=True)
        gold_m = sum(g[LABELS] for g in gold_parts) / len(gold_parts)
        gold_all[d] = gold_m
        y = train.loc[gold_m.index, LABELS].astype(int)
        auc = np.mean([roc_auc_score(y[l], gold_m[l]) for l in LABELS])
        print(f"== {d} 4-fold gold AUC: {auc:.4f}", flush=True)
        gold_m.reset_index().to_csv(out / f"gold_{d}.csv", index=False)
        pd.concat(oof_parts).reset_index().to_csv(out / f"oof_{d}.csv", index=False)

    if len(args.dirs) > 1:
        gm = sum(gold_all.values()) / len(gold_all)
        y = train.loc[gm.index, LABELS].astype(int)
        auc = np.mean([roc_auc_score(y[l], gm[l]) for l in LABELS])
        print(f"== combined {'+'.join(args.dirs)} gold AUC: {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
