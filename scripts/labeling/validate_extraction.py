#!/usr/bin/env python3
"""Score LLM-extracted labels against the human-labeled subset (n=58).

Usage: python validate_extraction.py --pred results/labels_llm_14b.csv --train-csv data/raw/train.csv
"""
import argparse

import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--out", default=None, help="optional CSV for per-label metrics")
    args = ap.parse_args()

    truth = pd.read_csv(args.train_csv).dropna(subset=LABELS)
    pred = pd.read_csv(args.pred)
    m = truth.merge(pred, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    print(f"labeled studies matched: {len(m)} | parse_ok: {int(m['parse_ok'].sum())}")

    rows = []
    for lab in LABELS:
        t = m[f"{lab}_true"].astype(int)
        p = m[f"{lab}_pred"].astype("Int64")
        ok = p.notna()
        t2, p2 = t[ok], p[ok].astype(int)
        tp = int(((t2 == 1) & (p2 == 1)).sum())
        tn = int(((t2 == 0) & (p2 == 0)).sum())
        fp = int(((t2 == 0) & (p2 == 1)).sum())
        fn = int(((t2 == 1) & (p2 == 0)).sum())
        acc = (tp + tn) / max(len(t2), 1)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rows.append(dict(label=lab, n=len(t2), prev=float(t2.mean()), acc=acc,
                         sens=sens, spec=spec, f1=f1, fp=fp, fn=fn))
    res = pd.DataFrame(rows)
    res.loc["mean"] = res.mean(numeric_only=True)
    print(res.to_string(float_format="%.3f"))
    if args.out:
        res.to_csv(args.out, index=False)

    # disagreements for manual review
    dis = []
    for lab in LABELS:
        bad = m[m[f"{lab}_true"].astype(int) != m[f"{lab}_pred"].fillna(-1).astype(int)]
        for _, r in bad.iterrows():
            dis.append(dict(label=lab, uid=r["StudyInstanceUID"],
                            true=int(r[f"{lab}_true"]), pred=r[f"{lab}_pred"]))
    if dis:
        print(f"\n{len(dis)} disagreements:")
        print(pd.DataFrame(dis).to_string())


if __name__ == "__main__":
    main()
