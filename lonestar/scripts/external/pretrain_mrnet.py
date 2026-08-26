#!/usr/bin/env python3
"""Pretrain the KneeStudyModel trunk on MRNet (abnormal/ACL/meniscus heads).

Saves a checkpoint whose backbone + attention pools warm-start the 12-label
competition model via train.py --init-ckpt.

Usage:
  CUDA_VISIBLE_DEVICES=0 python pretrain_mrnet.py --mrnet-root /scratch/.../MRNet-v1.0 \
      --backbone convnext_small.fb_in22k_ft_in1k --out mrnet_cnxts.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from dataset import collate  # noqa: E402
from model import KneeStudyModel  # noqa: E402
from mrnet_dataset import MRNET_TASKS, MRNetDataset  # noqa: E402

torch.manual_seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrnet-root", required=True)
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--k-slices", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    ds_tr = MRNetDataset(args.mrnet_root, "train", args.k_slices, args.crop)
    ds_va = MRNetDataset(args.mrnet_root, "valid", args.k_slices, args.crop, train_aug=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=10,
                       collate_fn=collate, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, num_workers=4, collate_fn=collate)
    print(f"MRNet train {len(ds_tr)} valid {len(ds_va)}", flush=True)

    model = KneeStudyModel(args.backbone, n_labels=len(MRNET_TASKS)).to(device)
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda a, c, n: 0.999 * a + 0.001 * c, use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=len(dl_tr) * args.epochs, pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()

    best = -1
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, meta, mask, y, _ in dl_tr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = bce(model(x.to(device), meta.to(device), mask.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            ema.update_parameters(model)
            tot += loss.item()
        net = ema.module if ep >= 1 else model
        net.eval(); preds, ys = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, meta, mask, y, _ in dl_va:
                preds.append(net(x.to(device), meta.to(device), mask.to(device)).float().sigmoid().cpu().numpy())
                ys.append(y.numpy())
        p, yv = np.concatenate(preds), np.concatenate(ys)
        aucs = {t: roc_auc_score(yv[:, i], p[:, i]) for i, t in enumerate(MRNET_TASKS)}
        mean_auc = float(np.mean(list(aucs.values())))
        print(f"ep{ep}: loss {tot/len(dl_tr):.4f} | valid AUC {aucs} mean {mean_auc:.4f} "
              f"| {(time.time()-t0)/60:.1f} min", flush=True)
        if mean_auc > best:
            best = mean_auc
            torch.save({"model": net.state_dict(), "backbone": args.backbone,
                        "mrnet_auc": aucs, "epoch": ep}, args.out)
    print(f"DONE best mean AUC {best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
