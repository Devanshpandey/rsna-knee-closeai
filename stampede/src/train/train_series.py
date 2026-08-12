#!/usr/bin/env python
"""Train the stage-1 per-series model on weak labels.

Review fixes: NaN-label drop+assert; gold studies excluded from training and
evaluated separately each epoch (sanity signal only — best.pt selection stays
on held-out weak-label macro-AUC); explicit >=0.5 thresholding (no banker's
rounding); fixed valid-label mask per fold; per-label weighted loss;
study score = 0.5*(mean+max) over series predictions; folds grouped by
scanner fingerprint unioned with duplicate-report clusters.

Usage:
  python train_series.py --npz-root data/processed/npz320 \
      --series-csv data/raw/train_series.csv \
      --labels data/processed/labels_train.parquet \
      --train-csv data/raw/train.csv \
      --fingerprints data/processed/fingerprints.csv \
      --fold 0 --epochs 4 --bs 8 --out runs/v1_f0
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SeriesDataset
from model import SeriesModel

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def union_find_groups(labels_df, fingerprints_csv, train_csv):
    """Group studies by scanner fingerprint, merging groups linked by
    byte-identical reports (template normals reused across studies)."""
    studies = labels_df[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)
    fp = pd.read_csv(fingerprints_csv)[["StudyInstanceUID", "fingerprint"]]
    studies = studies.merge(fp, on="StudyInstanceUID", how="left")
    studies["fingerprint"] = studies["fingerprint"].fillna("unk_fp")
    t = pd.read_csv(train_csv)[["StudyInstanceUID", "Report"]]
    t["rhash"] = t["Report"].fillna("").map(
        lambda s: hashlib.md5(s.encode()).hexdigest()[:12])
    studies = studies.merge(t[["StudyInstanceUID", "rhash"]], on="StudyInstanceUID", how="left")

    parent = {}
    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for r in studies.itertuples():
        union("fp:" + r.fingerprint, "rh:" + r.rhash)
    studies["group"] = ["g" + find("fp:" + r.fingerprint) for r in studies.itertuples()]
    return studies[["StudyInstanceUID", "group"]]


def make_folds(labels_df, n_folds, fingerprints_csv, train_csv, seed=42,
               pinned_csv="data/processed/folds_pinned.csv"):
    """GroupKFold tie-breaking proved environment-sensitive (fold assignments
    silently shifted mid-2026-08-11), so when a pinned assignment file exists
    it is authoritative and the computation below is skipped."""
    import os
    if pinned_csv and os.path.exists(pinned_csv):
        pf = pd.read_csv(pinned_csv)
        merged = labels_df[["StudyInstanceUID"]].merge(pf, on="StudyInstanceUID",
                                                       how="left")
        if not merged.fold.isna().any():
            return merged[["StudyInstanceUID", "fold"]]
        print(f"pinned folds missing {int(merged.fold.isna().sum())} studies — "
              "recomputing")
    g = union_find_groups(labels_df, fingerprints_csv, train_csv)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(g))
    g = g.iloc[order].reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_folds)
    g["fold"] = -1
    for f, (_, va) in enumerate(gkf.split(g, groups=g["group"].values)):
        g.loc[va, "fold"] = f
    return g[["StudyInstanceUID", "fold"]]


def study_scores(df, label_cols):
    """Per-study score = 0.5 * (mean + max) over series predictions."""
    gp = df.groupby("StudyInstanceUID")
    mean = gp[[c + "_p" for c in label_cols]].mean()
    mx = gp[[c + "_p" for c in label_cols]].max()
    ys = gp[[c + "_y" for c in label_cols]].first()
    return 0.5 * (mean.values + mx.values), ys.values, mean.index


def macro_auc(y_soft, p, valid_mask=None):
    y = (y_soft >= 0.5).astype(int)
    aucs, used = [], []
    for i in range(y.shape[1]):
        if valid_mask is not None and not valid_mask[i]:
            continue
        yi = y[:, i]
        if 0 < yi.sum() < len(yi):
            aucs.append(roc_auc_score(yi, p[:, i])); used.append(i)
    return (float(np.mean(aucs)) if aucs else float("nan")), dict(zip(used, aucs))


@torch.no_grad()
def predict(model, dl, dev, df_index, amp_dtype=torch.float16):
    model.eval()
    preds, ys = [], []
    for x, meta, pos, y, w in dl:
        with torch.amp.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
            logits, _ = model(x.to(dev), meta.to(dev), pos.to(dev))
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    p = np.concatenate(preds); yv = np.concatenate(ys)
    out = df_index[["StudyInstanceUID"]].copy()
    for i, c in enumerate(LABELS):
        out[c + "_p"] = p[:, i]; out[c + "_y"] = yv[:, i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--labels", required=True,
                    help="parquet: StudyInstanceUID + 12 soft labels (+ optional *_w weight cols)")
    ap.add_argument("--train-csv", required=True, help="for gold labels + report hashes")
    ap.add_argument("--fingerprints", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--k-slices", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="0 disables EMA; best.pt then tracks the raw model")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    labels = pd.read_parquet(args.labels)
    n0 = len(labels)
    labels = labels.dropna(subset=LABELS).reset_index(drop=True)
    if len(labels) != n0:
        print(f"dropped {n0 - len(labels)} rows with missing weak labels")
    assert np.isfinite(labels[LABELS].to_numpy(dtype=np.float32)).all()
    weight_cols = [c + "_w" for c in LABELS] if (LABELS[0] + "_w") in labels.columns else None

    # gold studies: excluded from training/weak-val, used as per-epoch sanity eval
    tcsv = pd.read_csv(args.train_csv)
    gold = tcsv.dropna(subset=[LABELS[0]])[["StudyInstanceUID"] + LABELS].reset_index(drop=True)
    gold_uids = set(gold.StudyInstanceUID)
    labels = labels[~labels.StudyInstanceUID.isin(gold_uids)].reset_index(drop=True)
    print(f"{len(labels)} weakly-labeled studies (gold {len(gold)} held out)")

    folds = make_folds(labels, args.n_folds, args.fingerprints, args.train_csv)
    labels = labels.merge(folds, on="StudyInstanceUID")
    series = pd.read_csv(args.series_csv)
    meta_p = Path(args.npz_root) / "series_meta.parquet"
    if meta_p.exists():  # authoritative list of successfully preprocessed series
        sm = pd.read_parquet(meta_p)
        have = set(zip(sm.StudyInstanceUID, sm.SeriesInstanceUID))
    else:
        have = {(p.parent.name, p.stem) for p in Path(args.npz_root).glob("*/*.npz")}
    series = series[[(s, u) in have for s, u in
                     zip(series.StudyInstanceUID, series.SeriesInstanceUID)]]
    if args.limit:
        keep = set(labels.StudyInstanceUID.drop_duplicates().head(args.limit)) | gold_uids
        series = series[series.StudyInstanceUID.isin(keep)]

    tr, va = labels[labels.fold != args.fold], labels[labels.fold == args.fold]
    ds_tr = SeriesDataset(series, tr, LABELS, args.npz_root, args.k_slices,
                          train=True, weight_cols=weight_cols)
    ds_va = SeriesDataset(series, va, LABELS, args.npz_root, args.k_slices, train=False)
    gold_soft = gold.copy()
    ds_gold = SeriesDataset(series, gold_soft, LABELS, args.npz_root, args.k_slices, train=False)
    print(f"train series {len(ds_tr)}, val series {len(ds_va)}, "
          f"gold series {len(ds_gold)}, device {dev}")
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=(dev == "cuda"), drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=args.workers)
    dl_gold = DataLoader(ds_gold, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    # fixed valid-label mask for this fold (which labels have both classes)
    yva = (va[LABELS].values >= 0.5).astype(int)
    valid_mask = [(0 < yva[:, i].sum() < len(yva)) for i in range(len(LABELS))]
    excluded = [LABELS[i] for i, v in enumerate(valid_mask) if not v]
    if excluded:
        print(f"labels excluded from fold-{args.fold} weak-val AUC: {excluded}")

    model = SeriesModel(args.backbone).to(dev)
    model = model.to(memory_format=torch.channels_last)
    ema = (ModelEmaV2(model, decay=args.ema_decay) if args.ema_decay > 0 else None)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda" and amp_dtype == torch.float16))
    crit = torch.nn.BCEWithLogitsLoss(reduction="none")

    def per_label_str(per):
        return " ".join(f"{LABELS[i][:9]}:{a:.3f}" for i, a in per.items())

    best, history = -1.0, []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for bi, (x, meta, pos, y, w) in enumerate(dl_tr):
            x, meta, pos = x.to(dev), meta.to(dev), pos.to(dev)
            y, w = y.to(dev), w.to(dev)
            with torch.amp.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
                logits, _ = model(x, meta, pos)
                loss = (crit(logits, y) * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item()
            if (bi + 1) % 100 == 0:
                print(f"ep{ep} {bi+1}/{len(dl_tr)} loss {tot/(bi+1):.4f} "
                      f"{(time.time()-t0)/(bi+1):.2f}s/it", flush=True)

        # evaluate the model we would ship: EMA if enabled, else raw
        eval_model = ema.module if ema is not None else model
        vdf = predict(eval_model, dl_va, dev, ds_va.df, amp_dtype)
        p_va, y_va, _ = study_scores(vdf, LABELS)
        auc, per = macro_auc(y_va, p_va, valid_mask)
        gdf = predict(eval_model, dl_gold, dev, ds_gold.df, amp_dtype)
        p_g, y_g, _ = study_scores(gdf, LABELS)
        gauc, gper = macro_auc(y_g, p_g)
        raw_auc = float("nan")
        if ema is not None:  # raw-model val for EMA diagnostics
            rdf = predict(model, dl_va, dev, ds_va.df, amp_dtype)
            p_r, y_r, _ = study_scores(rdf, LABELS)
            raw_auc, _ = macro_auc(y_r, p_r, valid_mask)
        print(f"ep{ep}: weak-val macro-AUC {auc:.4f} (raw {raw_auc:.4f}) | "
              f"gold-58 macro-AUC {gauc:.4f} (sanity only, sigma~0.0125)", flush=True)
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
        history.append({"epoch": ep, "weak_val_auc": auc, "raw_val_auc": raw_auc,
                        "gold_auc": gauc,
                        "per_label_val": {LABELS[i]: a for i, a in per.items()},
                        "per_label_gold": {LABELS[i]: a for i, a in gper.items()}})
        (out / "metrics.json").write_text(json.dumps(
            {"best_weak_val_auc": best, "history": history}, indent=1))
    print(f"best weak-val macro-AUC {best:.4f}")


if __name__ == "__main__":
    main()
