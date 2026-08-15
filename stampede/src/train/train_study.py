#!/usr/bin/env python
"""Train the study-level MIL model (devansh architecture x our recipe).

Same harness discipline as train_series.py: pinned folds, gold-58 excluded
from training and evaluated per epoch, EMA, bf16, per-label weighted loss,
best.pt on held-out weak-label macro-AUC. Predictions are study-level
directly — no mean/max aggregation.

Usage (one fold, 384px):
  python train_study.py --npz-root data/processed/npz384_study \
      --series-csv data/raw/train_series.csv \
      --labels data/processed/labels_train_v7_pl.parquet \
      --train-csv data/raw/train.csv \
      --fingerprints data/processed/fingerprints.csv \
      --fold 0 --epochs 6 --bs 2 --lr 6e-5 --k-slices 16 --max-series 6 \
      --out runs/mil384_v7_f0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_dataset import StudyDataset, collate
from study_model import KneeStudyModel
from train_series import LABELS, macro_auc, make_folds


@torch.no_grad()
def predict(model, dl, dev, amp_dtype):
    model.eval()
    preds, ys, uids = [], [], []
    for x, meta, mask, y, w, u in dl:
        with torch.amp.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
            logits = model(x.to(dev), meta.to(dev), mask.to(dev))
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy()); uids.extend(u)
    p = np.concatenate(preds); yv = np.concatenate(ys)
    out = pd.DataFrame({"StudyInstanceUID": uids})
    for i, c in enumerate(LABELS):
        out[c + "_p"] = p[:, i]; out[c + "_y"] = yv[:, i]
    return out, p, yv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--fingerprints", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2, help="grad accumulation steps")
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--k-slices", type=int, default=16)
    ap.add_argument("--max-series", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    labels = pd.read_parquet(args.labels)
    labels = labels.dropna(subset=LABELS).reset_index(drop=True)
    assert np.isfinite(labels[LABELS].to_numpy(dtype=np.float32)).all()
    weight_cols = [c + "_w" for c in LABELS] if (LABELS[0] + "_w") in labels.columns else None

    tcsv = pd.read_csv(args.train_csv)
    gold = tcsv.dropna(subset=[LABELS[0]])[["StudyInstanceUID"] + LABELS].reset_index(drop=True)
    gold_uids = set(gold.StudyInstanceUID)
    labels = labels[~labels.StudyInstanceUID.isin(gold_uids)].reset_index(drop=True)
    print(f"{len(labels)} weakly-labeled studies (gold {len(gold)} held out)")

    folds = make_folds(labels, args.n_folds, args.fingerprints, args.train_csv)
    labels = labels.merge(folds, on="StudyInstanceUID")
    series = pd.read_csv(args.series_csv)
    meta_p = Path(args.npz_root) / "series_meta.parquet"
    sm = pd.read_parquet(meta_p)
    have = set(zip(sm.StudyInstanceUID, sm.SeriesInstanceUID))
    series = series[[(s, u) in have for s, u in
                     zip(series.StudyInstanceUID, series.SeriesInstanceUID)]]
    if args.limit:
        keep = set(labels.StudyInstanceUID.drop_duplicates().head(args.limit)) | gold_uids
        series = series[series.StudyInstanceUID.isin(keep)]

    tr, va = labels[labels.fold != args.fold], labels[labels.fold == args.fold]
    kw = dict(label_cols=LABELS, npz_root=args.npz_root, k_slices=args.k_slices,
              max_series=args.max_series)
    ds_tr = StudyDataset(series, tr, train=True, weight_cols=weight_cols, **kw)
    ds_va = StudyDataset(series, va, train=False, **kw)
    ds_gold = StudyDataset(series, gold.copy(), train=False, **kw)
    print(f"train studies {len(ds_tr)}, val studies {len(ds_va)}, "
          f"gold studies {len(ds_gold)}, device {dev}")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=(dev == "cuda"), drop_last=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False,
                       num_workers=args.workers, collate_fn=collate)
    dl_gold = DataLoader(ds_gold, batch_size=args.bs, shuffle=False,
                         num_workers=args.workers, collate_fn=collate)

    yva = (va[LABELS].values >= 0.5).astype(int)
    valid_mask = [(0 < yva[:, i].sum() < len(yva)) for i in range(len(LABELS))]
    excluded = [LABELS[i] for i, v in enumerate(valid_mask) if not v]
    if excluded:
        print(f"labels excluded from fold-{args.fold} weak-val AUC: {excluded}")

    model = KneeStudyModel(args.backbone).to(dev)
    ema = (ModelEmaV2(model, decay=args.ema_decay) if args.ema_decay > 0 else None)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(dl_tr) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda" and amp_dtype == torch.float16))
    crit = torch.nn.BCEWithLogitsLoss(reduction="none")

    def per_label_str(per):
        return " ".join(f"{LABELS[i][:9]}:{a:.3f}" for i, a in per.items())

    best, history = -1.0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for bi, (x, meta, mask, y, w, _) in enumerate(dl_tr):
            x, meta, mask = x.to(dev), meta.to(dev), mask.to(dev)
            y, w = y.to(dev), w.to(dev)
            with torch.amp.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
                logits = model(x, meta, mask)
                loss = (crit(logits, y) * w).sum() / w.sum().clamp(min=1e-6)
            scaler.scale(loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            tot += loss.item()
            if (bi + 1) % 100 == 0:
                print(f"ep{ep} {bi+1}/{len(dl_tr)} loss {tot/(bi+1):.4f} "
                      f"{(time.time()-t0)/(bi+1):.2f}s/it", flush=True)

        eval_model = ema.module if ema is not None else model
        vdf, p_va, y_va = predict(eval_model, dl_va, dev, amp_dtype)
        auc, per = macro_auc(y_va, p_va, valid_mask)
        gdf, p_g, y_g = predict(eval_model, dl_gold, dev, amp_dtype)
        gauc, gper = macro_auc(y_g, p_g)
        print(f"ep{ep}: weak-val macro-AUC {auc:.4f} | gold-58 macro-AUC {gauc:.4f} "
              f"(sanity only, sigma~0.0125)", flush=True)
        print(f"  val per-label: {per_label_str(per)}", flush=True)
        print(f"  gold per-label: {per_label_str(gper)}", flush=True)
        state = {"model": eval_model.state_dict(), "auc": auc, "gold_auc": gauc,
                 "args": vars(args)}
        torch.save(state, out / f"ep{ep}.pt")
        if auc > best:
            best = auc
            torch.save(state, out / "best.pt")
        vdf.to_csv(out / f"oof_ep{ep}.csv", index=False)
        gdf.to_csv(out / f"gold_ep{ep}.csv", index=False)
        history.append({"epoch": ep, "weak_val_auc": auc, "gold_auc": gauc,
                        "per_label_val": {LABELS[i]: a for i, a in per.items()},
                        "per_label_gold": {LABELS[i]: a for i, a in gper.items()}})
        (out / "metrics.json").write_text(json.dumps(
            {"best_weak_val_auc": best, "history": history}, indent=1))
    print(f"best weak-val macro-AUC {best:.4f}")


if __name__ == "__main__":
    main()
