#!/usr/bin/env python3
"""Pseudo-label preprocessed fastMRI studies with an ensemble of trained checkpoints.

Writes a train.py-compatible folds CSV: t_<label> soft targets, fold assignment
(95% fold=1 train pool, 5% fold=0 sanity val), is_gold=0.

Usage: CUDA_VISIBLE_DEVICES=0 python pseudolabel_fastmri.py \
    --series-csv .../fastmri_series.csv --proc-dir .../fastmri_proc384f140 \
    --ckpt-dirs results/v7c_effs384 results/v7b_srx384 --out .../fastmri_folds.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from dataset import LABELS, KneeStudyDataset, collate  # noqa: E402
from model import KneeStudyModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--k-slices", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    series = pd.read_csv(args.series_csv)
    studies = sorted(series.StudyInstanceUID.astype(str).unique())
    stub = pd.DataFrame({"StudyInstanceUID": studies})
    stub["fold"] = 0          # dataset 'val' split of fold 0 = every study
    stub["is_gold"] = 0
    for l in LABELS:
        stub[f"t_{l}"] = 0.5
        stub[f"g_{l}"] = np.nan
    stub_path = Path(args.out).with_suffix(".stub.csv")
    stub.to_csv(stub_path, index=False)

    ds = KneeStudyDataset(str(stub_path), args.series_csv, args.proc_dir, 0, "val",
                          k_slices=args.k_slices, crop=args.crop, train_aug=False)
    dl = DataLoader(ds, batch_size=2, num_workers=10, collate_fn=collate)
    print(f"pseudo-labeling {len(ds)} studies", flush=True)

    models = []
    for d in args.ckpt_dirs:
        for p in sorted(Path(d).glob("fold*/best.pt")):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = KneeStudyModel(ck["backbone"], pretrained=False)
            m.load_state_dict(ck["model"]); m.eval().to(device)
            models.append(m)
    print(f"{len(models)} checkpoints loaded", flush=True)

    preds, uids = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for bi, (x, meta, mask, _, uid) in enumerate(dl):
            x, meta, mask = x.to(device), meta.to(device), mask.to(device)
            p = torch.stack([m(x, meta, mask).float().sigmoid() for m in models]).mean(0)
            preds.append(p.cpu().numpy()); uids += uid
            if (bi + 1) % 200 == 0:
                print(f"{(bi+1)*2}/{len(ds)}", flush=True)
    P = np.concatenate(preds)

    out = pd.DataFrame(P, columns=[f"t_{l}" for l in LABELS])
    out.insert(0, "StudyInstanceUID", uids)
    rng = np.random.RandomState(42)
    out["fold"] = np.where(rng.rand(len(out)) < 0.05, 0, 1)
    out["is_gold"] = 0
    for l in LABELS:
        out[f"g_{l}"] = np.nan
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} studies | mean t_ACL {out.t_ACL.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
