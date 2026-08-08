#!/usr/bin/env python3
"""Train one CV fold of the knee study model on teacher soft labels.

Model selection on val-fold AUC vs hard teacher labels; also reports AUC on the
58 gold studies (never trained on by any fold).

Usage (on a GPU node):
  CUDA_VISIBLE_DEVICES=0 python train.py --fold 0 --out-dir results/img_v1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import LABELS, KneeStudyDataset, collate
from model import KneeStudyModel

torch.manual_seed(42)
np.random.seed(42)

# per-label loss weights from teacher ensemble AUC on the gold 58: w = clip(2*auc-1, .2, 1)
TEACHER_AUC = {"ACL": .979, "MCL": .968, "Medial Meniscus": .919, "Lateral Meniscus": .872,
               "Medial OA": .960, "Lateral OA": .884, "PF OA": .793, "Effusion": .857,
               "Synovitis": .728, "Baker's": .909, "Contusion": .795, "Fracture": .846}
LABEL_W = torch.tensor([min(1.0, max(0.2, 2 * TEACHER_AUC[l] - 1)) for l in LABELS])


def evaluate(model, loader, device, gold_cols=False):
    model.eval()
    preds, targs, uids = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, meta, mask, y, uid in loader:
            out = model(x.to(device), meta.to(device), mask.to(device))
            preds.append(out.float().sigmoid().cpu().numpy())
            targs.append(y.numpy())
            uids += uid
    preds, targs = np.concatenate(preds), np.concatenate(targs)
    aucs = {}
    for i, lab in enumerate(LABELS):
        t = targs[:, i]
        th = (t >= 0.5).astype(int)
        if 0 < th.mean() < 1:
            aucs[lab] = roc_auc_score(th, preds[:, i])
    return preds, targs, uids, aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--folds-csv", default="data/csv/train_folds.csv")
    ap.add_argument("--series-csv", default="data/csv/train_series.csv")
    ap.add_argument("--proc-dir", default="/scratch/07880/devansh/rsna_knee/data/proc/train")
    ap.add_argument("--out-dir", default="results/img_v1")
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--k-slices", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate plumbing")
    args = ap.parse_args()

    device = "cuda"
    out = Path(args.out_dir) / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    ds_tr = KneeStudyDataset(args.folds_csv, args.series_csv, args.proc_dir,
                             args.fold, "train", k_slices=args.k_slices)
    ds_va = KneeStudyDataset(args.folds_csv, args.series_csv, args.proc_dir,
                             args.fold, "val", k_slices=args.k_slices, train_aug=False)
    ds_gold = KneeStudyDataset(args.folds_csv, args.series_csv, args.proc_dir,
                               args.fold, "gold", k_slices=args.k_slices, train_aug=False,
                               target_cols=[f"g_{l}" for l in LABELS])
    if args.smoke:
        ds_tr.meta = ds_tr.meta.head(24); ds_va.meta = ds_va.meta.head(12)
        args.epochs = 1

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=max(2, args.workers // 2), collate_fn=collate)
    dl_gold = DataLoader(ds_gold, batch_size=args.batch, shuffle=False,
                         num_workers=2, collate_fn=collate)
    print(f"fold {args.fold}: train {len(ds_tr)} | val {len(ds_va)} | gold {len(ds_gold)}", flush=True)

    model = KneeStudyModel(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = max(1, len(dl_tr) // args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    lw = LABEL_W.to(device)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    best = -1.0
    log = []
    for ep in range(args.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad()
        for i, (x, meta, mask, y, _) in enumerate(dl_tr):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to(device), meta.to(device), mask.to(device))
                loss = (bce(logits, y.to(device)) * lw).mean() / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * args.accum; nb += 1
            if nb % 50 == 0:
                print(f"  ep{ep} {nb}/{len(dl_tr)} loss {tot/nb:.4f} "
                      f"({(time.time()-t0)/nb:.2f}s/b)", flush=True)

        _, _, _, va_aucs = evaluate(model, dl_va, device)
        gp, gt, gu, gold_aucs = evaluate(model, dl_gold, device)
        va_mean = float(np.mean(list(va_aucs.values())))
        gold_mean = float(np.mean(list(gold_aucs.values()))) if gold_aucs else float("nan")
        print(f"ep{ep}: loss {tot/max(nb,1):.4f} | val-teacher AUC {va_mean:.4f} | "
              f"gold AUC {gold_mean:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
        log.append(dict(epoch=ep, loss=tot / max(nb, 1), val_auc=va_mean, gold_auc=gold_mean,
                        val_aucs=va_aucs, gold_aucs=gold_aucs))
        if va_mean > best:
            best = va_mean
            torch.save({"model": model.state_dict(), "backbone": args.backbone,
                        "epoch": ep, "val_auc": va_mean, "gold_auc": gold_mean},
                       out / "best.pt")

    # final: dump val + gold predictions from best checkpoint
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    vp, vt, vu, va_aucs = evaluate(model, dl_va, device)
    gp, gt, gu, gold_aucs = evaluate(model, dl_gold, device)
    pd.DataFrame(vp, columns=LABELS).assign(StudyInstanceUID=vu).to_csv(out / "val_preds.csv", index=False)
    pd.DataFrame(gp, columns=LABELS).assign(StudyInstanceUID=gu).to_csv(out / "gold_preds.csv", index=False)
    with open(out / "log.json", "w") as f:
        json.dump(dict(args=vars(args), best_val_auc=ck["val_auc"], history=log), f, indent=2, default=float)
    print(f"DONE fold {args.fold}: best val AUC {ck['val_auc']:.4f} "
          f"(gold {ck['gold_auc']:.4f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
