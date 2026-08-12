# RSNA Knee Abnormality Detection — Team CloseAI

Rank-1 pipeline (public LB 0.944, 2026-08-12) for the Kaggle competition
`rsna-knee-abnormality-detection`. Code only — competition data, extracted
labels, fold files, and trained weights are intentionally excluded per
competition rules; they live in the team's private Kaggle datasets
(rsna-knee-code / rsna-knee-npz320 / rsna-knee-npz384 / rsna-knee-weights).

## Pipeline
1. **Preprocessing** (`src/train/dicom_to_npz.py`): DICOM series -> canonical
   left-knee uint8 volumes, 140mm physical crop, <=48 slices, 320/384 px.
2. **Weak labels**: 12 findings extracted from radiology reports by a local
   LLM (rubric-driven, severity-graded). Gold-58 studies held out as sanity.
3. **Stage-1 model** (`src/train/model.py`): 2.5D — K=24 slice triplets ->
   timm backbone -> Fourier slice-position encoding -> attention pooling
   (+ plane/fluid/fat-sat embeddings) -> 12 logits.
4. **Training** (`src/train/train_series.py`): per-label-weighted BCE on soft
   targets, mirror aug with medial<->lateral label swap, EMA 0.999, bf16,
   OneCycle, GroupKFold folds PINNED to a file (see HANDOFF).
5. **v6/v7 pseudo-labels** (`src/train/make_labels_v6.py`): blend extractor
   targets with the ensemble's out-of-fold predictions — the decisive lever
   (+0.6 LB pts) once models outgrew the report extractor.
6. **Inference kernel** (`kernel/rsna-submit-v1.py`): offline Kaggle notebook;
   per-checkpoint arch/K/img-size, mirror TTA, study = 0.5*(mean+max),
   defensive fallbacks. Push with `push_kernel.py` (T4 machine shape required).

See HANDOFF.md for the full operational playbook and hard-won infrastructure
notes (P100 ban, cu118 wheel bundling, dataset-version races, fold pinning).
