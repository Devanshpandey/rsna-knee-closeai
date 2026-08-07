# RSNA Knee Abnormality Detection — Kaggle Competition

Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## Task

Predict per-study probability of **12 knee findings** from multi-sequence knee MRI (DICOM):
`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

- **Metric:** macro-averaged AUC-ROC across the 12 labels.
- **Type:** research **code competition** — submit via Kaggle Notebooks, **internet disabled**, ≤ 9 h GPU (or CPU) runtime, output `submission.csv`.
- **Efficiency track:** second prize pool scored by `AUC/(Benchmark − maxAUC) + Runtime/32400` (minimize) — a small/fast model is separately valuable.

## Timeline (2026)

| Date | Milestone |
|---|---|
| Jul 30 | Start |
| Oct 15 | Entry + team-merger deadline (must accept rules by then) |
| **Oct 22** | **Final submission deadline** |
| Nov 5 | Winners' code/video deadline |

## Data — the key structural facts

1. `train.csv`: one row per study — `StudyInstanceUID`, `PatientSex`, `Report` (free text, **multiple languages**), 12 binary labels.
2. **Only a small subset of training studies carry per-condition labels.** The radiology report is provided for all training studies → derive weak labels from reports for the rest.
3. **Test set (~1,300 studies) has NO reports** — images only. So text can only be a *teacher*, never an input at inference.
4. `train_series.csv`: per-series metadata — `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane` (Sagittal/Coronal/Axial). Series = one MRI acquisition; ~20–45 slices median 30, tail to few hundred.
5. DICOMs: mixed transfer syntaxes (Explicit/Implicit VR LE, JPEG Lossless, JPEG 2000), heterogeneous intensity/orientation/resolution, international multi-site (domain shift!). 86 allowlisted tags.
6. Prevalence differs between train / public LB / private LB — don't overfit thresholds or per-label priors to public LB.

## Strategy (agent-style phases)

### Phase 0 — Setup & EDA
- Download data, inventory series counts/planes/sequences per study, label prevalence, report languages.
- Grouped CV split by StudyInstanceUID (and site proxy if derivable from DICOM tags) — stratify by rare labels (Fracture, Lateral OA likely rare).

### Phase 1 — Report → labels (teacher)
- Parse the labeled subset to calibrate; then LLM/rule-hybrid extraction of the 12 findings from multilingual reports (translate-then-extract or multilingual model).
- Validate extraction against the human-labeled subset (per-label F1/AUC); iterate until extraction ≥ ~0.95 agreement on labeled subset.
- Output: soft labels (probabilities) for ALL training studies.

### Phase 2 — Image models (student)
- Per-series 2.5D/3D encoders (MRNet lineage; modern choice: 2D backbone + slice attention, or 3D CNN / video transformer). Route by sequence type (plane × fluid-sensitivity) — e.g., sagittal fluid-sensitive → ACL/meniscus/effusion; axial → PF OA/effusion; coronal → MCL/collaterals/OA compartments.
- Study-level head: attention/pooling over series embeddings → 12 sigmoid outputs.
- Train on report-derived soft labels + human labels (higher weight); knowledge distillation from a text-informed teacher head optional.
- Pretrained inits allowed (public external data/models OK): RadImageNet, SAM/MedSAM features, MRNet weights, video models.

### Phase 3 — Ensembling & submission engineering
- K-fold ensemble, TTA (flips — mind left/right knee laterality), per-label calibration.
- Offline notebook: package weights as a Kaggle Model/Dataset; 9 h budget for ~1,300 studies ⇒ ~25 s/study for full ensemble — plan compute budget early.
- Separate lightweight single-model branch for the Efficiency track.

## Compute plan

- **Mac** — EDA, report labeling, prototyping on a data subsample.
- **TACC LS6 GPU (MLL allocation)** — full training runs. Download data directly on TACC (`kaggle competitions download -c rsna-knee-abnormality-detection` in `$SCRATCH`).
- **Kaggle notebooks** — inference/submission only (no internet; weights uploaded as Kaggle datasets/models).

## Project structure

```
RSNA_knee/
├── CLAUDE.md            ← this file
├── data/                ← local subsample only (full data on TACC $SCRATCH)
├── scripts/
│   ├── labeling/        ← report → label extraction + validation vs labeled subset
│   ├── training/        ← per-series encoders, study head, folds
│   └── inference/       ← offline notebook build, weight packaging
├── notebooks/           ← Kaggle submission notebooks (committed copies)
├── results/             ← CV metrics, LB log
└── logs/
```

## Conventions

- Seed 42 everywhere; log every run to `logs/`.
- Track every submission in `results/lb_log.csv`: date, CV per-label AUC, macro AUC, public LB, notes.
- Trust CV over public LB (prevalence shift is explicit in the rules).
- Never train on anything derived from test images (no pseudo-labeling test — check rules before any test-time adaptation).
