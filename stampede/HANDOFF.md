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
- 08-25 ~05:30 UTC-5: milb384e10_v7 (cnxb MIL) DONE — best solo MIL (weak-val
  0.9520-0.9551, OOF 0.9062, gold 0.9185) but corr 0.966 vs mil384e10 ->
  REDUNDANT: no combo beats v12 config (gold 0.9291). MIL family space closed
  at one member. Series 10-ep retrain ruled out (pair curves flat at ep5).
  shared/labels now has v7/v6 parquets for LS6 lanes (devansh ask).
  NEW LANE: sw3d224e10_v7 — swin3d_t (Kinetics-400 video pretrain) per-series
  3D encoder + our series-attn head (src/train/study3d_{model,dataset}.py +
  train_study3d.py, smoke-tested); T=32 slices @224, 10 ep, 4 folds on H100s.
  The 'video pretraining' pivot lever — structurally new vs everything else.
- 08-25 ~07:00 UTC-5: sw3d224e10_v7 (swin3d_t Kinetics) DONE — most diverse
  family yet (corr 0.935 vs pair AND vs MIL) but solo gold 0.9136: every
  v12+sw3d combo is flat-to-negative (0.9286-0.9290 vs 0.9291). 6-ep MIL
  sibling also flat. VERDICT: Stampede-side family space is SATURATED at
  v12 = pair(noTTA) + mil384e10@1/3, offline gold 0.9291. GPUs idled for
  the user's other work. Waiting on: v12 LB score (>8h in), devansh's
  weights-r2 share (D-fam kernel integration; note offline D3 added ~0
  over e10-MIL: 0.9292), and LS6's new MIL@384 families (the main hope
  for a real stack step — our own diversity attempts all landed redundant).
- v12 RESULT: **0.945** (< v10 0.946). The gold-57 yardstick (+0.0037) ALSO
  failed to predict LB direction — both offline yardsticks are now confirmed
  uninformative at this margin (all our "gains" are inside gold sigma 0.0125).
  v12 changed 2 vars (noTTA + MIL); decomposition probe v13 = kernel v19
  (pair noTTA, W_MIL=0) launched — isolates TTA. If v13=0.946: MIL hurt, TTA
  free (bigger runtime budget stands). If 0.945: mirror TTA is load-bearing
  on hidden test; revert to TTA in all future kernels. BEST KNOWN CONFIG
  remains v10 = kernel VERSION 16 + current weights ds (v16 ignores mil_f*;
  submitting -v 16 reproduces v10 exactly). Keep for finals.
- FLAG for lab-server/devansh: external-data rules check (MRNet/fastMRI knee
  pretraining is the next structural lever; forum thread must confirm
  external data policy before we invest).
- v13 RESULT: **0.945** = v12. DECOMPOSITION ANSWER: mirror TTA is worth
  ~+0.001 LB (v10 0.946 with TTA vs v13 0.945 without) though offline read
  it as worthless; MIL@1/3 was LB-NEUTRAL (v12=v13). Rules going forward:
  (a) SERIES_TTA stays ON in every kernel (runtime budget: pair+TTA=56s stub);
  (b) hidden test rewards robustness/averaging beyond what weak-val OR gold
  can see — TTA-like variance reduction is a live LB lever, ensemble
  diversity is not. 2/5 subs left today. LB config unchanged: v10/kernel-v16.
- IN FLIGHT: cnxl384_v7 (convnext_large, exact cnxb recipe, bs4) — ep2 golds
  0.919-0.927 ALREADY at cnxb-final levels, weak-val 0.957-0.961. If the
  4-fold family beats cnxb offline, candidate component-upgrade submission;
  NOTE runtime: cnxl inference ~2x cnxb -> cnxl+v2m+TTA ~84s stub = TIMEOUT
  territory; viable shapes: cnxl-4-only+TTA (~56s) or cnxl+v2m TTA-on-cnxl-only.
- cnxl384_v7 COMPLETE (convnext_large, cnxb recipe, bs4, ~5h/4folds): per-fold
  weak-val 0.9595-0.9638 (all > cnxb). Family TTA eval: OOF 0.9141 = ENTIRE
  v10 pair, gold 0.9276 (+0.0022). v2m adds ~nothing to it (cnxl+v2m 0.9147/
  0.9257). K16-inference on K24 ckpts ~free (pair OOF -0.0006, gold flat).
  shared/oof has oof_/gold_cnxl384_v7.csv.
- SUBMITTED v14 18:38 UTC: cnxl 4-fold ALONE + TTA @K16-infer (kernel v20:
  SERIES_PATTERNS=cnxl_f*, INFER_K_CAP=16; weights v11 adds cnxl_f0-3,
  best_f*/mil_f* untouched so kernel v16 = v10 stays reproducible for finals).
  Stub 45s/3 studies (safe). Tests: does a stronger single family move the
  hidden test where ensemble diversity didn't? 1/5 subs left today.
- **v14 RESULT: 0.947** — FIRST LB GAIN since 08-12. cnxl384_v7 4-fold ALONE
  (TTA, K16-infer, 45s stub) beats the 8-model v10 pair. COMPONENT CAPACITY
  moves the hidden test; ensemble tweaks don't. Best config = kernel v20 +
  weights v11 (finals pointer updated from v16/v10).
- cnxxl384_v7 (xlarge) DONE: fold weak-vals 0.9600-0.9636 = TIED with cnxl —
  capacity curve FLAT at large->xlarge, and xlarge is ~1.7x inference cost.
  OOF export pending eval; stack asset only. Next capacity step needs >384px
  (lab server: preprocess npz448?) or a different scaling axis.
- NEXT PROBES (single-variable on v14): cnxl + mil384e10 @1/5 (45+11=56s stub,
  proven-safe runtime) if offline gold >= cnxl alone; cnxl+xlarge stack
  offline-check when eval lands. New UTC day: 5 subs available.
- v15 RESULT: **0.947** = v14 (MIL add LB-neutral again — 3rd confirmation
  that ensemble adds are invisible; only components move LB). Best = v14
  (kernel v20, cnxl alone). cnxle10 (10-ep cnxl) FLAT vs 6-ep (0.9597-0.9636)
  — series converge at 6 ep, confirmed at large too.
- NEW LANE cnxl448_v7: convnext_large trained on 384npz UPSCALED to 448
  (dataset.py/train_series.py grew an --img-size arg; ckpt args carry it).
  At inference the kernel can decode DICOM at TRUE 448 (preprocess is
  size-parametric) -> genuine test-res gain (FixRes). Kernel v22 will need
  size detection via args.img_size. Kernel budget: cnxl@448 K16+TTA ~61s
  (borderline) / K12 ~46s (safe; K12 penalty to be measured offline).
  4 folds on H100s, ~4.5h; allocation ends 08-26 14:28 UTC-5.

## Session end-state 2026-08-26 (node expires 14:28 UTC-5) — READ BEFORE RESUMING
- **BEST CONFIG: v14 = LB 0.947** (kernel VERSION 20 + weights ds v11+;
  = cnxl_f0-3 @384, TTA, K16-infer, mean+max agg, no MIL). kernel_submit_meta
  file is reverted to exactly this config — a plain push_kernel.py + submit
  reproduces it. v10 (0.946, pair) remains reproducible via kernel VERSION 16.
- Session LB probes: v12 0.945 (noTTA+MIL), v13 0.945 (noTTA — TTA is +0.001,
  keep always), v14 0.947 (cnxl component upgrade — CAPACITY is the lever),
  v15 0.947 (MIL add neutral 3rd time). 4/5 subs left today (UTC 08-26).
- cnxl448 (384-upscaled-448 training): offline NEGATIVE (gold 0.9244-0.9255 vs
  cnxl@384 0.9276 at every K) — true-448-decode probe SKIPPED. Real >384
  needs lab-server DICOM re-preprocess (npz448_study) — measurable offline,
  still the cleanest capacity-adjacent lever. Weights ds v12 carries
  cnxl448_f* anyway (unused unless a kernel points at them).
- Trained this session (all on /scratch runs/, OOF in shared/oof):
  mil384_v7 + mil384e10 (10ep >> 6ep for MIL), milb384e10 (redundant),
  sw3d224e10 (Kinetics, too weak), cnxl384_v7 (BEST family: OOF 0.9141 =
  old pair, gold 0.9276), cnxxl (ties cnxl, 1.7x cost), cnxle10 (10ep flat
  for series), cnxl448 (negative). Capacity curve: small<base<large=xlarge.
- NEXT (new allocation / lab server / LS6): (1) devansh MRNet-pretrained MIL
  + r5@384 diversity families — warm-start could lift solo quality to where
  diversity finally pays; (2) lab server npz448 preprocess -> cnxl@true448;
  (3) his weights-r2 share for in-kernel D-fams; (4) keep judging on LB only.
- eval_ensemble.py + dataset.py/train_series.py now support --img-size /
  args.img_size (upscale training + size-aware eval); kernel v22 lineage has
  img_size-aware size detection (kernel code keeps it — harmless for 384 ckpts).
- Offline note for when weights-r2 access lands: cnxl + D3(288-MIL fams) =
  gold 0.9292 (+0.0016) / OOF 0.9125 (-0.0016), corr 0.876 — most diverse
  pairing available but same mixed-yardstick ~+0.002 class that was LB-neutral
  3x. LOW-priority probe; prefer spending subs on genuinely new components
  (MRNet-warm-started families, true-448).

## Session 2026-08-29 (allocation ends 08-30 ~23:20 UTC-5) — TRUE-448 LANE
- LB check 08-29: we are ~#7 at 0.947 (Brandon Low 0.952, Matteo Vitali 0.950,
  Tiago Mazzutti 0.950, MKhlystun 0.948, Scott Willis 0.948, then a 0.947 pack
  incl. us). Best config unchanged: v14 = kernel VERSION 20 + weights v11+
  (cnxl_f0-3 @384, TTA, K16-infer, mean+max agg, no MIL).
- **PREPROCESSING IS NO LONGER LAB-SERVER-ONLY.** Downloaded the full 265 GB
  competition zip to Stampede3 (`kaggle competitions download`, 25 min @190MB/s,
  data/raw_zip/). Installed the decode deps into the igc env: pydicom pebble
  opencv-python-headless pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
  python-gdcm (the JPEG plugins are REQUIRED — pydicom 3 alone cannot decode
  this competition's compressed pixel data). Decoding all 24,371 series at any
  --size takes **~6 minutes** on the 96-core H100 node with --workers 90.
  The ">384px needs a lab-server re-preprocess" blocker in every prior handoff
  was wrong by three orders of magnitude — any resolution is now cheap to test.
- npz448 BUILT (data/processed/npz448, per-series layout, 87 GB): 24,369 ok,
  the SAME 2 corrupt series that the 384 set skipped. Verified single-variable
  vs npz384_study: identical series set, and laterality/mirrored/phys_crop/
  have_geom/n_raw agree 1.000000 on all 24,369; a fresh 448 volume downscaled
  to 384 matches the old 384 volume at corr 0.9994-0.9999. Only pixels differ.
  No repack needed — dataset.py/eval_ensemble.py both accept npz_root/<study>/
  <series>.npz directly and read series_meta.parquet from the npz root.
- IN FLIGHT: cnxl448t_v7 = convnext_large.fb_in22k_ft_in1k on TRUE 448, exact
  cnxl384_v7 recipe with ONLY resolution changed (v7_pl targets, 6 ep, bs 4,
  lr 9e-5, K=24, EMA 0.999, bf16, pinned folds), 4 folds on 4xH100, ~0.52 s/it,
  78.6/95 GB per GPU, ETA ~6h from 13:00 UTC-5 08-29. runs/cnxl448t_v7_f{0..3},
  logs/cnxl448t_f*.log. Launched WITH `--img-size 448` — see the blocker below.
- **NEW KERNEL BLOCKER (found by audit before it could burn a sub).**
  rsna-submit-v1.py:160-161 resolves decode size as
    `int(a.get("img_size") or 0) or (384 if "384" in str(a.get("npz_root","")) else DEFAULT_SIZE)`
  with DEFAULT_SIZE=320 (:86). A 448 checkpoint whose npz_root is
  ".../npz448" and whose img_size is 0/absent contains no "384" substring, so
  the kernel SILENTLY decodes the hidden test at 320px — no error, and a
  deceptively FAST stub (~30s) that reads as "safe". Training with an explicit
  `--img-size 448` stamps args.img_size=448 and defuses it (verified no-op on
  native-448 data: dataset.py:98 guards on x.shape[-1] != img_size).
  NOTE the v14 family (cnxl384_v7) has NO img_size key at all — its correct 384
  decode depends ENTIRELY on the "384" substring in npz_root. Do not "clean up"
  that expression without re-checking v14.
- Kernel budget for 4-fold cnxl@448 + mirror TTA (cost model calibrated to the
  four measured stub anchors v10 56s / v12 42s / v14 45s / v15 56s; 448/384
  forward ratio MEASURED at 1.367 on GPU, matching FLOPs 1.361):
    K16 -> 58.7s stub -> 8.1-8.4h hidden = TOO TIGHT (the kernel's printed
      window starts AFTER the ~2.5 GB cu118 wheel install, so budget ~8.6h not 9h)
    **K14 -> 52.3s stub -> 7.4h, 13-14% margin = RECOMMENDED**
    K12 -> 45.9s stub -> 6.6h, 23% margin = fallback
  Only 0.5% of series have <=14 usable slices (vs 3.0% at <=16), so K14 is still
  genuinely sampling. channels_last measured at exactly 1.000x — that lever is
  empty. Never buy runtime by dropping mirror TTA (LB-measured +0.001).
- DEPLOY CHECKLIST for this lane (do NOT skip): set SERIES_PATTERNS to the new
  filename (use cnxl448t_f*.pt — do NOT reuse cnxl448_f*, weights ds v12 already
  ships the offline-NEGATIVE upscale-448 ckpts under that name), INFER_K_CAP=14,
  **W_MIL=0** (any mil_f* forces 384 into the size set = a second full resize
  pass per series), and KEEP a file literally named best_f0.pt in the weights
  dataset (kernel:27 `WEIGHTS = find_input("best_f0.pt").parent` raises
  FileNotFoundError at import otherwise). Stub gate: PASS <=53s; if >56s drop to
  K12; **if it prints 30-35s ABORT — that is the 320px fallthrough signature.**
- DECISION RULE for spending a submission on this family (offline yardsticks are
  4-for-4 useless at this margin — a paired bootstrap over the five known LB
  outcomes shows OOF read significant-POSITIVE for v10 which was LB-flat, and
  NEGATIVE for v14, the only probe that ever gained): judge on **per-fold best
  weak-val vs cnxl384_v7's 0.9631/0.9638/0.9595/0.9634, require >=3/4 at or
  above**. That is the only offline metric that has ever agreed with LB, and it
  is what correctly flagged the upscale-448 lane as not-better (1/4). Treat
  gold-58/OOF as no-catastrophe gates only. The real justification is that
  true-448 is a COMPONENT CAPACITY change — the one class that has moved this LB.
- **TRUE-448 VERDICT: NEGATIVE (gate not met) — NO SUBMISSION SPENT.**
  cnxl448t_v7 4 folds complete (runs/cnxl448t_v7_f*, ~5h on 4xH100). Best
  weak-val per fold vs the cnxl384_v7 baseline:
      f0 0.9625 vs 0.9631 (-0.0006)   f1 0.9624 vs 0.9638 (-0.0014)
      f2 0.9595 vs 0.9595 (+0.0000)   f3 0.9637 vs 0.9634 (+0.0003)
  2/4 at-or-above, mean -0.00042 — the pre-registered gate (>=3/4) is NOT met.
  Per-fold gold-58 at ep5: 0.9226/0.9247/0.9228/0.9251 (single-model, healthy —
  the family is not broken, it is simply at parity).
  **RESOLUTION IS A DEAD AXIS, now measured twice**: 384-upscaled-to-448 was
  negative, and a genuine 448 DICOM decode is flat. Combined with large=xlarge
  at 384, BOTH obvious capacity axes (parameters, pixels) are saturated for this
  140mm-crop task. Do not spend more GPU-hours on >384 resolution.
- NEW LANE (in flight): PRETRAINING-QUALITY screen at the proven 384 operating
  point — the capacity axis never tested. Note the current LB-0.947 backbone
  `convnext_large.fb_in22k_ft_in1k` is the **224-trained** checkpoint fed 384px
  images; timm also ships a 384-native finetune of the identical architecture.
  Screening 4 backbones on FOLD 0 ONLY (gate: fold-0 weak-val vs 0.9631), all
  with the exact cnxl384_v7 recipe, runs/screen_<name>_f0:
      cnxl384n  convnext_large.fb_in22k_ft_in1k_384      (bs4)   ~3h
      cnxldino  convnext_large.dinov3_lvd1689m           (bs4)   ~3h
      cnxv2l    convnextv2_large.fcmae_ft_in22k_in1k_384 (bs2 a2) ~7.5h
      mvitl     maxvit_large_tf_384.in21k_ft_in1k        (bs2 a2) ~8.5h
  The first three are convnext_large (197.7M) = IDENTICAL kernel inference cost
  to the current model, so a winner ships with zero runtime re-budgeting
  (K16@384, 45s stub, proven). maxvit is 211.8M and a different arch — if it
  wins, re-measure the stub before submitting.
  Plan: whichever of the two fast candidates clears fold-0 baseline, train its
  remaining 3 folds on the freed GPUs, then apply the same >=3/4 gate.
  CAVEAT: maxvit uses BatchNorm, so its bs2+accum2 run is confounded vs the bs4
  baseline; convnextv2 is LayerNorm/GRN only, so bs2+accum2 there is equivalent.
- CODE CHANGE: `train_series.py` gained `--accum` (default 1, no behavior change
  at the default; OneCycle total_steps is now epochs*(len(dl_tr)//accum) and the
  step/EMA/clip happen on accumulation boundaries). Needed because convnextv2
  and maxvit OOM at bs4/K24/384 on a 95GB H100; accumulation keeps the effective
  batch at 4 so the screen stays single-variable.
- **BACKBONE SCREEN RESULTS (fold-0 gate vs cnxl384_v7 f0 = 0.9631):**
    cnxldino  convnext_large.dinov3_lvd1689m  -> **COLLAPSED**: ep0 weak-val
      0.5085 (chance); train loss flatlined at 0.571 (base rate) while a healthy
      run fell 0.622->0.549. DINOv3 self-supervised features do not adapt under
      our recipe (lr 9e-5, OneCycle, 6ep). Killed at ep0; not worth an LR hunt.
      runs/screen_cnxldino_f0_COLLAPSED.
    mvitl     maxvit_large_tf_384.in21k_ft_in1k -> **FAR BEHIND**: ep0 0.8918
      (-0.0348), ep1 0.9390 (-0.0157). Killed at ep1 (also the slowest lane at
      ~8.5h and BatchNorm-confounded by its bs2 run). runs/screen_mvitl_f0_ABANDONED_ep1.
    cnxl384n  convnext_large.fb_in22k_ft_in1k_384 -> **FAILS, 0/2 folds**:
      f0 0.9625 vs 0.9631 (-0.0006), f1 0.9629 vs 0.9638 (-0.0009).
      NOTABLE NEGATIVE: our LB-0.947 backbone is the *224-trained* checkpoint fed
      384px images; swapping to the 384-native finetune of the IDENTICAL
      architecture at IDENTICAL inference cost (textbook FixRes) is flat-to-worse.
      6 epochs of full-backbone finetuning apparently washes out the pretrained
      resolution. (f0 landed on 0.9625 — the exact value cnxl448t f0 gave.)
      runs/screen_cnxl384n_f0, runs/cnxl384n_v7_f1.
    cnxv2l    convnextv2_large.fcmae_ft_in22k_in1k_384 -> ONLY SURVIVOR, all 4
      folds now running (runs/screen_cnxv2l_f0 = fold 0, runs/cnxv2l_v7_f{1,2,3}).
      ep0 +0.0031, ep1 +0.0005 vs baseline — positive but NARROWING. bs2+accum2
      (effective batch 4); convnextv2 is LayerNorm/GRN only so this is NOT
      batch-confounded. Same 197.8M params => kernel runtime unchanged
      (K16@384, 45s stub) if it wins. Gate: >=3/4 folds at-or-above
      0.9631/0.9638/0.9595/0.9634. ETA f0 ~02:00, f1-f3 ~05:30 UTC-5 08-30.
- RUNNING TALLY OF FLAT AXES (all within +/-0.002 of baseline): parameters
  (large=xlarge), pixels (448-upscale negative, true-448 flat), pretraining
  resolution (384-native flat), 3 of 4 pretrainings. If cnxv2l also lands flat,
  the honest read is that the COMPONENT is not the binding constraint and the
  remaining levers are labels/targets or the external-data (MRNet) lane, not
  more backbone screening.
- OPERATIONAL GOTCHAS burned time this session: (1) `pkill -f` / `pgrep -f`
  MATCH THE INVOKING SHELL'S OWN COMMAND LINE — a pkill killed my own shell
  mid-command, and a GPU queue-watcher deadlocked because its `until ! pgrep -f
  "<pattern>"` never emptied (the shell that heredoc'd the script still carried
  the pattern in its cmdline). Use PIDs from /proc, or poll nvidia-smi memory.
  (2) `ps` truncates to ~80 cols with no TTY, so long training command lines are
  NOT greppable via ps — read /proc/<pid>/cmdline instead.

## 2026-08-30 — **MEASURED THE NOISE FLOOR. THE OFFLINE GATE WAS MEASURING NOISE.**
**READ THIS BEFORE TRUSTING ANY OFFLINE NUMBER IN THIS FILE, INCLUDING MINE.**

`train_series.py` sets NO global seed — there is no `torch.manual_seed` anywhere
(the only `seed=42` is in `make_folds`, which pinned folds bypass). So head init,
batch order and augmentation draws differ every run. Nobody had ever measured what
that is worth. I retrained the EXACT cnxl384_v7 fold-0 config 5 more times,
changing nothing at all:

    0.9631  <- the ORIGINAL run, i.e. the fold-0 number of the LB-0.947 model
    0.9626  0.9629  0.9632  0.9625  0.9627   <- 5 replicates, identical config

    mean 0.96283   sd 0.00028   range 0.9625-0.9632 (spread 0.0007)

**The 0.9631 we have been gating against is a +0.95 sd draw, not the true centre.**
95% of reruns of the very same recipe land within +/-0.0005 of 0.9628.

Re-judged against the replicate mean instead of that one lucky run:

    candidate              fold0     vs 0.9631      vs mean    sd units  verdict
    true 448px             0.9625     -0.0006      -0.00033     -1.19     TIE
    384-native weights     0.9625     -0.0006      -0.00033     -1.19     TIE
    convnextv2 FCMAE       0.9623     -0.0008      -0.00053     -1.90     TIE

Every delta I gated on yesterday was 0.0001-0.0008 — ENTIRELY INSIDE the noise
band. convnextv2's fold 1 "failed" by 0.0001. That number decides nothing.

**CORRECTION to my own 08-29 entry.** I wrote that the >=3/4-fold weak-val gate
"is the only offline metric that has ever agreed with LB". That claim does not
survive this measurement and should not be relied on. It was inferred from a
handful of unreplicated single runs; at this effect size the gate resolves below
its own measurement error and can fail on noise (convnextv2: 2/4, mean +0.00008 —
dead even) or pass on noise. Ignore that sentence.

FULL 4-FOLD VERDICTS, correctly stated as ties:
    cnxl448t_v7  (true 448)   2/4, mean delta -0.00042  = -2.12 sd -> marginal,
                              at worst a hair WORSE. Resolution still not a lever.
    cnxv2l_v7    (convnextv2) 2/4, mean delta +0.00008  = +0.40 sd -> clean TIE.
    cnxl384n     (384-native) 0/2 folds, both ~-0.0007  -> TIE.
(The sd of a 4-fold mean difference is ~0.00020, since each fold's baseline is
itself a single unreplicated run.)

### HOW TO RUN AN OFFLINE COMPARISON FROM NOW ON
- Weak-val sd of this recipe is **0.00028** per fold. Required seeds PER ARM
  (80% power): a true +0.0005 needs ~6 seeds; +0.0010 needs ~2; +0.0020 needs 1.
  A single run per arm can resolve NOTHING below ~0.0008.
- Report mean +/- sd over seeds, never a bare single number.
- **gold-58 is far worse and should be retired for model selection.** The trainer
  itself prints "sigma~0.0125" next to it — that is 45x the weak-val sd. Every
  gold-58 delta ever used in this file to justify a decision (they are all in the
  +/-0.003 range) is deep inside noise. This fully explains the long-standing
  "offline yardsticks decoupled from the LB" mystery in the 08-15/08-25 entries:
  they were not decoupled, they were never resolving anything.

### A CLAIM THAT NOW NEEDS RE-EXAMINING (flagged, not resolved)
The strategy in this file rests on "COMPONENT CAPACITY moves the hidden test;
ensemble tweaks don't", concluded from **v14 0.947 vs v10 0.946 — a single
0.001 LB step**. The hidden test is fixed, so re-submitting the SAME weights is
deterministic; but two DIFFERENTLY-SEEDED trainings of the same config are two
different models, and their LB scores will differ by something in the same family
as the 0.00028 weak-val sd (smaller after 4-fold ensembling, but not zero).
A 0.001 step on a 3-decimal board is plausibly ~1 sd. I could NOT test this
without burning submissions and did not. Treat "capacity is the lever" as an
UNVALIDATED hypothesis resting on one 1-sigma-ish observation, not as settled.
Cheapest honest test: submit two independently-seeded builds of the SAME config
and see how far apart they land. That measures the LB's own model-seed noise and
would calibrate every past probe. It costs 2 submissions and is worth more than
another architecture probe.

### WHERE THIS LEAVES THE PROJECT
Parameters (large=xlarge), pixels (448 upscale AND true decode), pretraining
resolution, and 3 of 4 pretraining objectives are now ALL ties within noise.
The component is not the binding constraint. Do not spend more GPU-hours
screening backbones — that lane is measured out. Remaining untested levers are
the targets/labels and the external-data (MRNet) lane.
NO SUBMISSION WAS SPENT on any of this. LB config unchanged: v14 = kernel
VERSION 20 + weights v11+ = 0.947.

Artifacts on /scratch: runs/baseline_replicate{,2,3,4,5}_f0 (the noise floor),
runs/cnxl448t_v7_f{0-3}, runs/screen_cnxv2l_f0 + runs/cnxv2l_v7_f{1,2,3},
runs/screen_cnxl384n_f0 + runs/cnxl384n_v7_f1, data/processed/npz448.

## 2026-08-30 evening — SEED-CALIBRATION SUBMITTED + v9 TARGET LANE (fresh 48h alloc, node c562-010, ends ~09-01 20:35)
Standing at session start: **CloseAI 0.947, rank 9** (Brandon Low 0.952; Vitali
/Mazzutti 0.950; MKhlystun/Willis/Molvaer 0.948). We have not submitted since 08-26.

### seedB is COMPLETE — the LB seed-noise experiment is now runnable
The previous session finished a config-identical, differently-seeded 4-fold rebuild
of the LB-0.947 cnxl384_v7 build. Verified by diffing checkpoint args: the ONLY
differences are `accum` None->1 and `img_size` None->0, both new-arg defaults that
are exact no-ops (`img_size` only resizes when the npz res differs; npz384 is 384).

    fold        v14 (seed A, on LB @0.947)   seedB (seed B)
    0              0.96310                     0.96262   <- runs/baseline_replicate_f0
    1              0.96376                     0.96204
    2              0.95952                     0.95940
    3              0.96335                     0.96236
    mean           0.96243                     0.96161   (-0.00082, 4/4 folds lower)

NOTE seedB is uniformly a hair below seed A offline. Fold 0 is fine (its 0.96262 is
the first of the 5 replicates, whose mean is 0.96277); the -0.00082 mean is carried
mostly by f1 (-0.00172). Read the LB result with that in mind: seed A looks like a
lucky draw across folds, so a seedB LB score BELOW 0.947 is expected and the
question is only HOW FAR below.

### The 3-point design now in flight (weights dataset v13, kernel version 22)
All 8 checkpoints live in `liaoyixu/rsna-knee-weights` as `cnxl_f*.pt` (seed A) and
`cnxlB_f*.pt` (seed B), so BOTH probes come off ONE dataset version — no version
race. Model selection is by the kernel's `SERIES_PATTERNS` glob, so the arms are
switched in the kernel script, not by re-uploading weights. Both arms run W_MIL=0.0
so they sit directly against v14 (cnxl alone) = 0.947, undiluted by the MIL head.

    v14  SERIES_PATTERNS=["cnxl_f*.pt"]                 = 0.947   (already on board)
    v16  SERIES_PATTERNS=["cnxlB_f*.pt"]                = ?  SUBMITTED 22:28, kernel v22
    v17  SERIES_PATTERNS=["cnxl_f*.pt","cnxlB_f*.pt"]   = ?  SUBMITTED 22:35, kernel v23

BOTH STUBS VERIFIED BEFORE SUBMITTING (this is the step never to skip):
    v16: 4 cnxlB models, K=16 size=384, 0 MIL, 3 rows, 0 series failures, 43.3s/3 studies
    v17: 8 models (both seeds), K=16 size=384, 0 MIL, 3 rows, 0 failures, 72.9s/3 studies
The size=384 confirms the npz_root-substring decode path fired correctly (these
checkpoints have img_size=0, so they ride entirely on that substring — the 320px
silent-fallthrough trap did NOT bite).

RUNTIME NOTE for v17: naive 3-study extrapolation says ~8.8h vs the 9h cap, which
looks alarming, but a 3-study stub badly over-reads per-study cost (CUDA warm-up is
a large fraction of it) — the same arithmetic predicts 12.3h for v10, which actually
COMPLETED. The real anchor is that v10 was 8x384 @ K24 + TTA and finished; v17 is
the same 8 models at K16, ~33% less inference work, so it is strictly inside a
proven-fitting config. If v17 nonetheless times out, that retires the "8x384 max"
rule and the fix is INFER_K_CAP=12 or dropping to 6 models.

v16 answers the question flagged as unresolved in the 08-30 noise-floor entry: how
big is the LB's OWN model-seed noise? Every past 0.001-0.002 LB step — including
the "capacity is the lever" conclusion built on v14 0.947 vs v10 0.946 — is
uninterpretable until this number exists. v17 tests whether seed-ensembling the
best component buys anything; 8x384 @ K16 + TTA is comfortably inside the 9h
ceiling (v14 was 4 models @ ~45s stub; 12 models @ K24 was the timeout).

### GPU lane: cnxl384_v9 — the targets axis, single-variable
Backbones are measured out (08-30 entry). Targets/labels is one of the two levers
left. IMPORTANT — do not just reuse `labels_train_v8_pl.parquet`: v8 broke the
chain that actually worked, TWICE. It rebased onto T_dev (the devansh extractor,
a MEASURED-NEGATIVE source, see 08-12 Task-1) and it silently DROPPED the 12 `_w`
per-label loss-weight columns (v8 parquet has 13 cols; v6/v7 have 25). Any v8-vs-v7
comparison therefore moves two variables at once.

Recovered the real v7 recipe numerically: v7 ~= 0.4*v5_aligned + 0.6*(v6-gen pair
OOF), uncovered rows keep v5 exactly (mean recon err 0.005 at w=0.40; 0.0096 at
0.35, 0.0151 at 0.50). So v9 is that same blend with ONLY the OOF generation
swapped to our best available OOF:

    python src/train/make_labels_v6.py \
      --v5 data/processed/labels_train_30b_v5_aligned.parquet \
      --oof preds/cnxl384_v7/oof_study_scores.csv \
      --w-extractor 0.4 --out data/processed/labels_train_v9_pl.parquet

(cnxl384_v7 OOF = pooled 0.9141 / gold 0.9276, vs the weaker v6-generation OOF that
built v7.) v9 keeps 25 cols incl. `_w`, row order matches v5-aligned -> pinned folds
intact. Coverage 4349/4407 (58 gold held out, correct).

Launched via `run_v9_lane.sh` (in kaggle/): 4 folds, one per H100, exact cnxl384_v7
recipe with ONLY --labels changed. ~0.39 s/it -> ~3h wall for all 4.
Verified pinned folds loaded (f1 = 18366 train / 5667 val series, identical to
seedB_f1). Deliberately did NOT pass --img-size, to keep it a true single-variable
change vs v7/seedB (kernel's "384"-in-npz_root substring gives the right decode).

### Gotcha added this session
`kaggle datasets version` REPLACES the version with the dir contents, so the upload
dir must carry every file you want to keep. weights_upload_v13 was built by
hardlinking weights_upload_v12 (`cp -l`, no extra space on /scratch) and adding the
4 cnxlB files.

### cnxl384_v9 COMPLETE (4 folds, ~3h on 4xH100)
    fold   best weak-val(v9 targets)   gold-58 final
    0          0.96209                    0.9242
    1          0.96143                    0.9240
    2          0.95791                    0.9264
    3          0.96240                    0.9268
    gold mean  0.9254   vs v7 0.9250   vs seedB 0.9238

DO NOT compare that weak-val column to v7's 0.9624/0.9638/0.9595/0.9634. Weak-val is
scored against each run's OWN training targets, and v9's targets differ from v7's —
this is the "never compare weak-val across label versions" gotcha. It shows up
immediately: v9's ep0 weak-val (0.9113-0.9164) reads ~0.008 BELOW v7's (0.9189-0.9229)
purely as a label-version artifact, while its gold-58 over the same epochs sat inside
the v7/seedB band the whole way. gold-58 is the only cross-version-comparable metric
here, and at sigma~0.0125 it cannot select — v9 0.9254 / v7 0.9250 / seedB 0.9238 is a
three-way tie.

The valid comparison is OOF re-scored against the FIXED yardstick
(labels_train_30b_v5_aligned.parquet). Running BOTH families through it now
(`run_v9_eval.sh`, or the parallel 2-GPU form) rather than reusing the stored
preds/cnxl384_v7/eval.log, because that log does not record which --labels produced
its 0.9141 — assuming would risk exactly the cross-version error above.
Outputs: preds/cnxl384_v9_yard/, preds/cnxl384_v7_yard/, logs/eval_*_yard.log.

HOW TO READ THE RESULT: per the 08-30 noise floor, a single-seed 4-fold offline delta
below ~0.0008 resolves NOTHING. Treat this eval as a regression sanity check, not a
gate. The v6->v7 precedent was flat offline and +0.002 on LB, so unless v9 shows a
real drop the informative next step is an LB probe, not an offline verdict.
