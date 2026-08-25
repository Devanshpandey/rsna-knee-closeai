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
- LATE UPDATE (node T-5min): v2m384_tdev completed ALL 4 folds before expiry
  (fold golds 0.889/0.895/0.903/0.899 — uniformly below the v5 baseline,
  confirming the Task-1 negative). TTA family eval + shared/oof export did
  NOT finish — checkpoints in /scratch/09059/xliaoyi/kaggle/runs/v2m384_tdev_f*/;
  next GPU allocation should run eval_ensemble on them and version the OOF
  files into shared/oof/ if the stacking work wants them.

## Session 2026-08-14 (fresh 48h allocation, ends 08-15 18:28 UTC-5)
- Cleared the expiry backlog: v2m384_tdev 4-fold TTA eval done (pooled OOF
  0.8867 / gold 0.9113 — Task-1 negative stands); full 4-fold OOF+gold
  replaced the 3-fold files in shared/oof.
- shared/oof now has ALL 11 Stampede3 families for the Task-3 stack (v5-era
  x5, v6 pair, v7 pair, tdev pair, cnxt384_v8) — 4-fold TTA, v5-aligned
  yardstick, folds_v4, manifest updated. v2m384_v7 per-family eval was
  missing and is now computed (OOF 0.9109 / gold 0.9225).
- v8 VERDICT (cnxt384_v8 4-fold complete): gold 0.9273 vs v7 0.9258
  (+0.0015, within sigma 0.0125); v5-yardstick OOF 0.9045 (mismatched
  yardstick for pseudo-label targets — expected to read low). Offline FLAT,
  consistent with target-lane saturation; per the v7 lesson the deciding
  test is an LB probe of the v8 PAIR.
- v2m384_v8 COMPLETE (4 folds, exact v2m384_v7 recipe on v8 targets):
  pooled OOF 0.9042 / gold 0.9246. shared/oof has all 12 families.
- v8 PAIR (8 models, TTA): OOF 0.9064 / gold 0.9263 vs v7 pair 0.9118 /
  0.9234 (LB 0.946). Gold +0.0029 (noise), v5-yardstick OOF lower (target
  drift artifact). OFFLINE CANNOT SEPARATE v8 FROM v7 — same as the v7-vs-v6
  call, which transferred +0.2 LB. Recommend ONE probe of the v8 pair when
  the submission budget allows; if flat, the target-engineering lane is
  confirmed closed and effort goes to Task-3 stacking + bigger backbones.
  Checkpoints: runs/cnxt384_v8_f*/best.pt + runs/v2m384_v8_f*/best.pt
  (lab server decides; weights_upload/ still stages the v7 pair, untouched).
- NODE-SHARING NOTE: the user runs cell-tracking trainings on the same tap_jupy
  node (up to 49 GB on one GPU). Check nvidia-smi per-GPU headroom before
  launching; a 384px bs6-8 fold needs ~50-55 GB.
- NEW FAMILY cnxb384_v7 (convnext_base@384 on v7_pl, 6 ep, bs 6, lr 9e-5,
  K=24, EMA): pooled OOF 0.9126 = BEST single family (cnxt384_v7 0.9101,
  v2m384_v7 0.9109); gold 0.9255. Weak labels improved: LatMen 0.921,
  PF OA 0.870, Synovitis 0.877 (OOF). shared/oof has all 13 families.
  Checkpoints runs/cnxb384_v7_f*/best.pt. Kernel is capped at 8x384 models —
  swapping cnxb384_v7 in (e.g. for v2m384) is a lab-server call; greedy
  subset selection over the 13 exported OOF families can decide offline.
- Session stopped launching new lanes with ~15h walltime left: 6-ep family
  space is saturated offline; remaining levers (v8-pair LB probe, kernel
  family swap, Task-3 stack once devansh's OOF lands) are lab-server-side.
  GPUs freed for the user's cell-tracking work.

## Session 2026-08-15 — LB response (we fell to #3: Brandon Low 0.949, MKhlystun 0.946 tiebreak)
- 10-agent fold-honest stacking study over all 18 shared/oof families
  (doc: shared/oof/STACKING_RESULTS.md). Verified winner: uniform prob-avg
  {cnxb384_v7, v2m384_v6} OOF 0.9141 / gold 0.9254 (+0.0020/+0.0010 vs v7
  pair, 4/4 folds, zero selection optimism by nested greedy). Learned
  stackers NO-GO (tie at 7x kernel budget). D-mix: devansh's 2:1-4:1
  sweet spot REFUTED — best D weight <=2.5-7.5%, gains <=0.0002.
- DEPLOYED from Stampede3: weights dataset v9 (best_f{0-3}.pt = cnxb384_v7,
  best_f{0-3}_v2m384v6.pt = v2m384_v6), kernel v16 (code unchanged,
  auto-adapts), SUBMITTED 2026-08-15 15:32 UTC as "v10" — PENDING.
  Expected LB ~0.9486 +/- 0.001. Stub inference window 56s vs 47s for the
  old pair (+19%, safe vs the 12-model timeout). 4/5 probes left today.
- If v10 >= 0.948: probe candidates in STACKING_RESULTS.md sec 5 (rank-avg
  variant, 2.5% D-mix) are method probes, not score probes. If v10 < 0.947:
  winner's-curse hit; fall back config #5 {cnxt384_v7, v2m384_v6} (+0.0009,
  zero runtime risk) and investigate.
- AGGREGATION STUDY (shared/oof/AGGR_RESULTS.md): top-4-mean per label
  beats 0.5(mean+max) by +0.0009 (4/4 folds, k=3-4 plateau, verified,
  label-free). ALL fitted aggregators (LR/LGBM, 14-28 features) lose by
  0.003-0.009 — that lever is closed. Kernel v17 = pair + top-4 mean.
- SECOND PROBE SUBMITTED 16:07 UTC: "v11" = kernel v17 (pair + top-4 agg),
  combined OOF 0.9147 (+0.0027 vs old config), expected LB ~0.949.
  v10 (pair only, kernel v16) still pending -> clean A/B on aggregation.
  3/5 probes left today. Series dumps for 7 modern families in preds/
  <fam>_series/ (also enables future aggregation work on other configs).
- v10 RESULT: 0.946 — flat vs old config at 3-decimal LB resolution.
  The +1.3 LB/OOF mapping did NOT hold for this step (+0.0020 OOF -> ~0 LB).
  Read: our weak-label OOF yardstick has decoupled from the hidden test at
  this margin; sub-0.001-LB increments are also invisible on a 3-decimal
  board. STOP probing incremental ensemble variants. Meanwhile leaders
  moved: Brandon Low 0.950 (+0.004 in 2 days), MKhlystun 0.947. The gap
  (~0.004) is beyond everything left in our saturated ensemble space —
  next gains must come from a qualitatively different lever: architecture
  (3D/cross-series attention at scale), pretraining (video/MRI weights),
  external data (check rules thread), or resolution >384 with the physical
  crop. Keep v10/v11 for the conservative final pick evaluation.
- READY TO LAUNCH (next allocation): study-MIL@384 lane — devansh's
  architecture (slice-attn -> series-attn, stampede/src/train/study_model.py)
  x our full recipe (plane-aware mirror+label-swap, [0,1] norm, pinned
  folds, EMA, bf16; study_dataset.py + train_study.py). Smoke-tested
  end-to-end on npz384_study. Architecturally diverse from everything in
  the kernel AND from devansh's 320px retrains on LS6. Launch per fold:
    python3 src/train/train_study.py --npz-root data/processed/npz384_study \
      --series-csv data/raw/train_series.csv --labels data/processed/labels_train_v7_pl.parquet \
      --train-csv data/raw/train.csv --fingerprints data/processed/fingerprints.csv \
      --fold K --epochs 6 --bs 2 --accum 2 --lr 6e-5 --k-slices 16 \
      --max-series 6 --workers 12 --out runs/mil384_v7_fK
  (~50GB/GPU at bs2; est 5-7h/fold on a free H100 — start all 4 folds at
  allocation start, not mid-walltime.)

## Session 2026-08-24/25 (fresh 48h allocation, ends 08-26 14:28 UTC-5)
- LB check 08-24: we fell to ~#8 at 0.946 (Brandon Low 0.952, MKhlystun 0.948,
  2x 0.947, pack at 0.946). v11 RESULT (unrecorded before): 0.945 — top-4
  aggregation LOST to 0.5(mean+max) on LB despite +0.0009 OOF. Kernel reverted
  to mean+max. OOF yardstick decoupling confirmed twice now.
- mil384_v7 lane COMPLETE (4 folds, ~1.5h/fold on H100 — NOT 5-7h as
  estimated): per-fold best weak-val 0.9458/0.9453/0.9347/0.9406 all at ep5
  (still rising). Family export in shared/oof (oof_/gold_mil384_v7.csv):
  solo OOF(v5-yard) 0.8926 / gold-57 0.9118 — weak solo, but corr vs v10 pair
  0.937 and STACKS: pair+MIL gold 0.9254->0.9272 (+0.0018, flat 1/5-1/3 wt).
- PRESET STACK GRID (scratch, no search — gold-57 prob-avg):
  pair 0.9254 | +mil384 0.9272 | +D3 0.9269 (devansh repro'd) |
  pair+Lc7+D3 0.9274 | pair+D3+mil384 0.9287 (BEST) | +Lc7 adds nothing
  once mil384 in (0.9286) -> Lc7's 4 series ckpts NOT needed, saves runtime.
- KERNEL v18 STAGED (kernel_submit_meta/rsna-submit-v1.py): MIL inference
  path added (loads mil_f*.pt, KneeStudyModel, K16 S6 no-TTA, bit-exact vs
  trainer eval — tested), aggregation reverted to 0.5(mean+max) (v10-proven),
  SERIES_TTA + W_MIL=0.2 constants. Needs study_model.py added to
  rsna-knee-code dataset before push.
- RUNTIME BUDGET: pair+TTA ~= 56s stub ~= 8h hidden (tight). pair+TTA+mil384
  ~= 67s stub — RISKY. Options: drop series TTA (halves series cost; penalty
  UNMEASURED — measuring next on freed GPUs), or 2-fold MIL. D3 add-on
  (+12 ckpt @288) only fits with no-TTA series.
- @devansh ACTION NEEDED: share Kaggle dataset devansh26/rsna-knee-weights-r2
  with user liaoyixu (403 for us now) — required to attach D families to the
  submission kernel. Also: repo lonestar predict.py decodes WITHOUT the 140mm
  physical crop / canonicalization — confirm the dataset src/predict.py is the
  authoritative decode for r4/r5 ckpts, and whether our canonical 384->resize
  320 volumes can feed them (would save a second DICOM decode pass in-kernel).
- IN FLIGHT: mil384e10_v7 (same recipe, 10 ep) 4 folds on 4xH100, ETA ~2.5h
  from 01:00 UTC-5 08-25. Then: no-TTA pair eval, final config pick, weights
  ds v10 (v9 files + mil_f*.pt), kernel push, stub check, LB probe.
- 08-25 02:30-03:00 UTC-5: mil384e10_v7 (10 ep) DONE — big step up (solo gold
  0.9205, stack gold 0.9291-0.9297 @1/3). TTA measured ~worthless on the pair
  (see shared/oof MANIFEST.md) -> dropped. DEPLOYED: weights v10 (pair +
  mil_f0-3), code ds + study_model.py (NOTE: 'kaggle datasets version' SKIPS
  folders silently — wheels/ was dropped once, restored with --dir-mode zip;
  always verify file list), kernel v18 (no-TTA pair + MIL@1/3, mean+max agg),
  stub 42s/3 studies 0 fails. SUBMITTED v12 07:33 UTC — PENDING, exp ~0.948+.
  4/5 subs left today. IN FLIGHT: milb384e10_v7 (convnext_base MIL, 10 ep,
  bs1 accum4) 4 folds on H100s, ETA ~5h — candidate 3rd stack member.

## devansh sync 2026-08-25
- Back online (LS6 idle, MLL 160k SUs, EXPIRES SEP 30 — front-load big training).
- v12 monitored from our side; will analyze on score.
- ASK RECEIVED (weights-r2 share): being done via Kaggle UI (API can't add collaborators).
- ASK TO liaoyixu: drop labels_train_v7_pl.parquet (and v6 if handy) into shared/labels/
  — LS6 will train 2-3 more study-MIL@384 diversity families (seresnext50, effnetv2_s,
  maxvit @384) on v7 targets for the stack. 384px/140mm preprocessing already launching
  on LS6 so data is ready the moment targets land.
