#!/usr/bin/env python3
"""Round-2 (noisy-student) targets: per-label blend of teacher scores and model OOF predictions.

Blend weight w per label chosen on the gold-58 by maximizing AUC of
  w * model + (1-w) * teacher
where "model" = average of fold-model predictions on gold (models never saw gold),
and for training studies the model score is strictly out-of-fold.
w is capped at 0.75 so every label keeps some teacher signal, except labels where
teacher AUC on gold is below 0.75, where w may reach 1.0.

Usage (TACC):
  python make_round2_targets.py --model-dirs results/img_v1 results/img_v2 \
      --folds-csv data/csv/train_folds.csv --train-csv data/csv/train.csv \
      --out data/csv/train_folds_r2.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--folds-csv", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--expert-csv", default=None, help="optional expert_review.csv overrides")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folds = pd.read_csv(args.folds_csv)
    train = pd.read_csv(args.train_csv).set_index("StudyInstanceUID")

    # OOF model predictions (avg across model dirs), gold predictions (avg folds x dirs)
    oof_parts, gold_parts = [], []
    for d in args.model_dirs:
        oof = []
        golds = []
        for f in range(4):
            vp = Path(d) / f"fold{f}/val_preds.csv"
            gp = Path(d) / f"fold{f}/gold_preds.csv"
            if vp.exists():
                oof.append(pd.read_csv(vp))
            if gp.exists():
                golds.append(pd.read_csv(gp).set_index("StudyInstanceUID")[LABELS])
        if oof:
            oof_parts.append(pd.concat(oof).set_index("StudyInstanceUID")[LABELS])
        if golds:
            gold_parts.append(sum(golds) / len(golds))
    oof_m = sum(o.loc[oof_parts[0].index] for o in oof_parts) / len(oof_parts)
    gold_m = sum(gold_parts) / len(gold_parts)
    print(f"OOF coverage: {len(oof_m)} studies from {len(oof_parts)} model dirs")

    gold_lab = train.loc[gold_m.index, LABELS].astype(int)
    tset = folds.set_index("StudyInstanceUID")
    teacher_gold = tset.loc[gold_m.index, [f"t_{l}" for l in LABELS]]

    weights, report = {}, []
    for lab in LABELS:
        y = gold_lab[lab].values
        tm = teacher_gold[f"t_{lab}"].values
        mm = gold_m[lab].values
        auc_t = roc_auc_score(y, tm)
        cap = 1.0 if auc_t < 0.75 else 0.75
        best_w, best_auc = 0.0, auc_t
        for w in np.arange(0, 1.01, 0.25):
            if w > cap:
                continue
            a = roc_auc_score(y, w * mm + (1 - w) * tm)
            if a > best_auc:
                best_w, best_auc = w, a
        weights[lab] = best_w
        report.append(dict(label=lab, teacher_auc=auc_t,
                           model_auc=roc_auc_score(y, mm),
                           blend_w_model=best_w, blend_auc=best_auc))
    rep = pd.DataFrame(report)
    print(rep.to_string(float_format="%.3f"))

    out = folds.copy().set_index("StudyInstanceUID")
    for lab in LABELS:
        w = weights[lab]
        t = out[f"t_{lab}"].copy()
        m = oof_m[lab].reindex(out.index)
        blend = w * m + (1 - w) * t
        out[f"t_{lab}"] = blend.fillna(t)      # gold studies / missing OOF keep teacher
    # optional expert overrides (hard labels from manual review)
    if args.expert_csv and Path(args.expert_csv).exists():
        ex = pd.read_csv(args.expert_csv)
        n = 0
        for _, r in ex.iterrows():
            if r.StudyInstanceUID in out.index and f"t_{r.label}" in out:
                out.loc[r.StudyInstanceUID, f"t_{r.label}"] = float(r.call)
                n += 1
        print(f"expert overrides applied: {n}")

    out.reset_index().to_csv(args.out, index=False)
    rep.to_csv(Path(args.out).parent / "round2_blend_report.csv", index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
