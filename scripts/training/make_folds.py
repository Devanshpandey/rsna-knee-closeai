#!/usr/bin/env python3
"""Build train_folds.csv: teacher soft labels (14B+7B ensemble), report language,
language-stratified folds, gold-58 quarantine flag.

Run locally: python scripts/training/make_folds.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
N_FOLDS = 4


def main() -> None:
    train = pd.read_csv("data/raw/train.csv")
    p14 = pd.read_csv("results/labels_llm_14b_v2.csv").set_index("StudyInstanceUID")
    p7 = pd.read_csv("results/labels_llm_7b_v2.csv").set_index("StudyInstanceUID")

    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    langs = train["Report"].fillna("").map(lambda s: detect(str(s)[:800]) if s else "unk")

    df = train[["StudyInstanceUID"]].copy()
    df["lang"] = langs.values
    gold = train.dropna(subset=LABELS)
    df["is_gold"] = df["StudyInstanceUID"].isin(gold["StudyInstanceUID"]).astype(int)

    idx = df["StudyInstanceUID"]
    for lab in LABELS:
        a = p14.loc[idx, lab].astype(float)
        b = p7.loc[idx, lab].astype(float)
        ens = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1).mean(axis=1, skipna=True)
        df[f"t_{lab}"] = ens.values                # teacher soft target (14B-only where 7B unparsed)
    for lab in LABELS:                             # gold labels where they exist
        df[f"g_{lab}"] = train.set_index("StudyInstanceUID").loc[idx, lab].values

    # stratify on language x coarse label profile (ACL + effusion bins)
    strat = df["lang"] + "_" + (df["t_ACL"] > 0.5).astype(str) + "_" + (df["t_Effusion"] > 0.5).astype(str)
    # rare strata -> collapse to language only
    vc = strat.value_counts()
    strat = strat.where(strat.map(vc) >= N_FOLDS, df["lang"])

    # group by byte-identical report cluster so duplicated reports never span folds
    rep_norm = train.set_index("StudyInstanceUID").loc[idx, "Report"].fillna("").str.strip()
    df["rep_group"] = rep_norm.factorize()[0]
    df["fold"] = -1
    nong = df[df.is_gold == 0]
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for f, (_, va) in enumerate(sgkf.split(nong, strat.loc[nong.index], groups=nong["rep_group"])):
        df.loc[nong.index[va], "fold"] = f
    # gold studies: fold = -1 (never trained on; used as global validation set)

    df.to_csv("data/csv/train_folds.csv", index=False)
    print(df.groupby("fold")["lang"].value_counts().unstack(fill_value=0).to_string())
    print(f"\ngold quarantined: {int(df.is_gold.sum())} | per-fold sizes:\n{df.fold.value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
