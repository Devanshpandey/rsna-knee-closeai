#!/usr/bin/env python3
"""Phase 0 EDA: label prevalence, labeled-subset size, report languages, series inventory.

Usage: python eda_labels.py --data data/raw --out results/
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

random.seed(42)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")

    labeled = train.dropna(subset=LABELS, how="all")
    summary = {
        "n_studies": len(train),
        "n_labeled_studies": len(labeled),
        "n_series": len(series),
        "series_per_study": series.groupby("StudyInstanceUID").size().describe().to_dict(),
        "plane_counts": series["Anatomical_Plane"].value_counts().to_dict(),
        "fluid_sensitive_frac": float(series["Fluid_Sensitive"].mean()),
        "fat_suppression_frac": float(series["Fat_Suppression"].mean()),
        "label_prevalence_in_labeled": {c: float(labeled[c].mean()) for c in LABELS if c in labeled},
        "report_present_frac": float(train["Report"].notna().mean()),
        "report_len_chars": train["Report"].str.len().describe().to_dict(),
    }
    with open(out / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

    # Rough language guess from a sample of reports (for the labeling pipeline design)
    try:
        from langdetect import detect  # optional dep
        sample = train["Report"].dropna().sample(min(500, train["Report"].notna().sum()), random_state=42)
        langs = sample.map(lambda t: detect(str(t)[:1000])).value_counts()
        langs.to_csv(out / "report_languages.csv")
        print(langs)
    except ImportError:
        print("langdetect not installed — skipping language breakdown (pip install langdetect)")


if __name__ == "__main__":
    main()
