# HANDOFF — RSNA Knee Abnormality Detection (for a fresh Claude Code session, e.g. on TACC)

You are joining an active Kaggle competition effort. Read this fully before acting.
STATUS 2026-08-12: **RANK 1 at 0.944 public** (next: 0.943/0.942/0.942/0.941).
Winning config: 8 models = 4-fold convnext_small@384 + 4-fold effnetv2_m@384,
both trained on **v6 pseudo-labels** (0.5·v5-extractor + 0.5·16-model-OOF,
make_labels_v6.py; labels_train_v6_pl.parquet in rsna-knee-code). OOF 0.9123,
gold-58 0.9228. GOAL NOW: hold #1 through 2026-10-22 (leaders will respond).
Next levers: extractor v6+ (lab server — biggest), careful v7 pseudo-label
iteration (LB-validate each round), longer 384 schedules.
Deadline: 2026-10-22. Submissions: 5/day, 2 finals. Code competition (offline
9-h Kaggle notebook produces submission.csv).

## TACC session progress 2026-08-11 (READ FIRST — supersedes parts below)
- Trained on 4xH100 with EMA(0.999)+bf16+per-label-AUC train_series.py (in
  rsna-knee-code dataset now): per-family 4-fold OOF-TTA numbers —
  cnxt_small@320 0.8908, effnetv2_m@320 0.8929, cnxt_base@320 0.8916,
  cnxt_small@384 0.8907 (gold 0.9179), 8-model 0.8953→LB 0.938,
  12-model 0.8961, 16-model 0.8963 (SATURATED — more 6-ep families ≈ no gain;
  weak-label OOF ceiling ~0.896). effnetv2_m@384 training now.
- LB mapping so far: OOF 0.8908→0.932, 0.8953→0.938 (slope ≈ +1.3 LB pt/OOF pt).
- Per-label blend-weight optimization: nested-CV gain +0.0002 — NOT worth it.
- K=32 inference on K24-trained models: +0.02pt — not worth kernel runtime.
- CRITICAL fold gotcha: make_folds() depends on labels parquet ROW ORDER.
  v5 parquet row order differs from v4! Use labels_train_30b_v5_aligned.parquet
  (v5 values in v4 row order, in code dataset) — fold hash then matches v4.
- CRITICAL kernel gotchas: competition FORBIDS P100; CLI kernels push cannot
  set accelerator and lands on P100 (image torch also dropped sm_60). Push with
  push_kernel.py (kagglesdk 0.1.28, machine_shape="NvidiaTeslaT4"). Kaggle
  strips '+' from wheel filenames (kernel restores it). rsna-torch-cu118
  dataset carries cu118 torch+torchvision installed pre-import (works on T4 and
  P100). Wait for `kaggle datasets files` to show new files before kernels
  push (status says "ready" during processing — version race attaches stale data).
- Kernel now reads backbone/K/img-size per checkpoint (size via args.npz_root
  "384" substring); preprocesses each series once per needed size. Weights
  dataset naming: best_f{K}.pt (cnxt320), best_f{K}_v2m.pt, best_f{K}_cnxb.pt.
- Next levers (in rough value order): better labels (v6+ extractor — lab
  server); study-level aggregator over per-series preds (untested; mean/max
  heuristic is the ceiling suspect); longer 384px schedules; greedy family
  subset selection on OOF.

## Split of responsibilities
- **Lab-server session** (this file's author) orchestrates: report labeling with
  local LLMs, Kaggle dataset/kernels/submissions, preprocessing from the 247 GB zip.
- **TACC session (you)**: heavy GPU training only. Produce fold checkpoints;
  hand them back as files (see "returning weights").

## Context you need
- Task: 12 binary findings per knee-MRI study; macro-AUC; hidden test ~1300
  studies, images only. Only 58/4407 train studies have gold labels — training
  targets are LLM-extracted from multilingual radiology reports (rubric-driven,
  severity-graded; gold macro-AUC of the extractor ≈ 0.84).
- Full strategy/history: docs/STRATEGY.md, docs/rubric_v4_notes.md, README.md.
- Data (preprocessed): canonical LEFT-knee uint8 volumes, physical 140 mm crop,
  320px, ≤48 slices/series, per-study npz bundles (`vol_<SeriesUID>` arrays).
  Available as private Kaggle dataset `liaoyixu/rsna-knee-npz320`
  (shard_*/ dirs of study .npz) — download with the Kaggle token, or rsync
  data/processed/npz320_study/ from the lab server.
- Labels: data/processed/labels_train_30b.parquet (binary v4) and — once the
  overnight run finishes — labels_train_30b_v5.parquet (severity-graded:
  targets 0.03–0.985). Also in Kaggle dataset `liaoyixu/rsna-knee-code`.
- Code: src/train/{dataset,model,train_series,stage2}.py. dataset.py handles
  both per-series and study-bundle npz layouts. Folds: GroupKFold over scanner
  fingerprints unioned with duplicate-report clusters (fingerprints.csv).
  Gold studies are EXCLUDED from training and evaluated per-epoch as sanity.

## Reference training command (single fold)
```bash
python src/train/train_series.py \
  --npz-root <dir with study .npz files + series_meta.parquet> \
  --series-csv data/raw/train_series.csv \
  --labels data/processed/labels_train_30b_v5.parquet \
  --train-csv data/raw/train.csv \
  --fingerprints data/processed/fingerprints.csv \
  --fold 0 --n-folds 4 --epochs 3 --bs 4 --lr 1.5e-4 --k-slices 16 \
  --workers 8 --out runs/v2_f0
```
v1 recipe results (T4, 320px, bs4, 3 epochs): weak-val 0.866–0.889,
gold-58 0.876–0.884 per fold; 4-fold ensemble gold 0.889.

## What to improve on TACC (A100/H100 headroom)
1. Bigger backbone: `--backbone convnext_small.fb_in22k_ft_in1k` or
   tf_efficientnetv2_m; bs 16–32 at 320px fits easily on A100.
2. Longer schedules with EMA (add EMA to train_series.py — not yet implemented).
3. K=24–32 slices; 384px once the lab server re-preprocesses (npz384 pending).
4. Log per-label AUC; menisci/OA are the weak labels.
5. Keep the fold definition EXACTLY as make_folds() produces (comparability).

## Returning weights
Copy best.pt per fold back to the lab server
(/stor/scratch/Kirkpatrick/xliaoyi/rsna-knee/kaggle_weights_ds/best_f<K>.pt), or
upload directly: `kaggle datasets version -p kaggle_weights_ds -m "..."` with the
user's token (~/.kaggle/access_token pattern). The submission kernel
(notebooks/submit_v1) automatically ensembles every best_f*.pt it finds.

## Rules constraints (do not violate)
- Competition data must not go to hosted LLM APIs; local/open-weights only.
- Do not publish competition data publicly; private Kaggle datasets are fine.
- All code used for winning submissions must be open-sourceable (CC-BY-NC OK).

## Environment hints (TACC)
- python>=3.11, torch+cu12x, timm, pandas, scikit-learn, pydicom (only if
  preprocessing), lightgbm (stage-2). No DICOM work needed if you use npz320.
- The npz dataset download: `kaggle datasets download liaoyixu/rsna-knee-npz320`
  (46 GB; the shard dirs contain the study npz files directly).

## Session end-state 2026-08-12 ~21:40 UTC (node expiry) — READ BEFORE RESUMING
- LB: **CloseAI 0.946 = RANK 1** (tied MKhlystun 0.946, ahead on tiebreak;
  then 0.943/0.942). Scored config = v7 pseudo-label pair (cnxt-s@384 +
  v2m@384, 8 models, kernel v15, weights dataset current). v7 iteration
  transferred (+0.2 LB) despite flat offline gold — offline yardsticks
  underestimate pseudo-label generations; LB-probe them.
- KERNEL RUNTIME CEILING: 12x384 models + TTA EXCEEDS 9h rerun (one burned
  submission). Max 8x384 models.
- Task 1 (T_dev = devansh extractor): NEGATIVE. cnxt384_tdev OOF 0.8867 /
  gold 0.9140 vs v5 baseline 0.8907/0.9179. v2m384_tdev fold golds
  0.889-0.903 (also below v5's 0.9036-0.9146). Extractor accuracy does NOT
  transfer to students; severity grading + pseudo-labels matter more.
  shared/oof/ has cnxt384_tdev; v2m384_tdev eval was racing node expiry —
  checkpoints in stampede runs/ on /scratch (survive the node).
- v8 targets (per-label trust map §1.3 over T_dev + v7-pair OOF,
  labels_train_v8_pl.parquet): folds 0 done (gold 0.9192 — between v6 and
  v7), f1/f2 mid-train at expiry, f3 not started. Verdict incomplete;
  pseudo-label iterations (v6->v7->v8) all land gold 0.92-0.926 — the
  target-engineering lane is saturating. Next levers: team stacking (Task 3,
  devansh OOF on folds_v4), longer schedules, bigger 384 backbones.
- All fold assignments PINNED: shared/folds/folds_v4.csv (hash
  17161916260226758772). NEVER recompute (env-sensitive; see export_folds.py).
- Checkpoint inventory on /scratch/09059/xliaoyi/kaggle/runs/: v5-era
  (cnxt,v2m,cnxb@320; cnxt384,v2m384@384), v6/v7 pairs @384 (LB 0.944 =
  v6 pair), tdev + partial v8. Weights dataset currently holds the v7 pair.
