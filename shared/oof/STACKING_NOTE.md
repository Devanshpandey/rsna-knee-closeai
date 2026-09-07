# Cross-team stacking — first measurement (2026-08-14, devansh)

Families: L = liaoyixu tdev pair (cnxt384_tdev, v2m384_tdev), D = devansh MIL pair
(mil_effnetv2s_288, mil_seresnext50_288). All folds_v4 (hash 17161916260226758772).

## Gold-58 (57 common studies) — uniform prob averaging
| mix | gold AUC |
|---|---|
| L pair | 0.9143 |
| D pair | 0.9003 |
| all 4 uniform | 0.9153 |
| L:D = 2:1 | 0.9161 |
| L:D = 3:1 | **0.9168** |

Monotone gain from adding down-weighted D — but n=57, treat as directional only.

## OOF prediction correlation (n=4,349; mean per-label Pearson)
- within L: 0.942 · within D: 0.918 · **cross-team: 0.837–0.855**

The cross-team correlation is far below within-team — real architectural diversity,
exactly what the saturated uniform ensembles (~0.896 OOF ceiling) lacked.

## Asks for the stacking session
1. Re-run the mix table with the **v6 rank-1 pair** OOF (not exported yet) in place of tdev.
2. Weight search on OOF ranks (not gold): expected sweet spot L:D between 2:1 and 4:1.
3. If LGBM/logistic study-aggregator is trained, D-family per-series stats are NOT available
   (MIL predicts study-level only) — include D as study-level prob features.
4. Candidate LB probe when ready: v6 pair + 0.25-0.33x weighted D pair.

## Addendum (r5 families added)
D3 = mil_cnxtsmall_288 (gold 0.894 family avg; fold0 0.9098 = best single D fold),
D4 = mil_maxvit_224. Six-family gold-57 mixes: all L-heavy variants cluster at
0.9163–0.9168 (vs L pair 0.9143) — D contribution robust (~+0.002) but composition
within D doesn't matter at n=57; pick by OOF weight search. Cross-corr: D3/D4 vs L
= 0.87 (D1/D2 remain most diverse at 0.84–0.855). All-D-uniform overweighting hurts
(0.9129) — keep D at 1/4 to 1/3 pool weight. devansh weights being packaged to
Kaggle dataset `devansh26/rsna-knee-weights-r2` (new version) for kernel integration.

## Addendum 2 (2026-08-15, devansh) — vs the actual v10/LB-0.946 pair
| mix | gold-57 |
|---|---|
| v10 pair (cnxb384_v7 + v2m384_v6) | 0.9254 |
| + D 3-fam (effv2s, srx50, cnxts) @ 1/4 pool | 0.9269 |
| + Lc7 (cnxt384_v7) + D 3-fam | **0.9274** |

Key structural fact: v10 pair internal OOF corr = **0.978** (near-redundant);
D-vs-L corr = 0.85. The pair is diversity-starved — that's where the stack gain lives.

**Probe recommendation:** v10 pair + cnxt384_v7 + D families, D at ~1/4 pool weight.
Runtime guard for the 9h kernel: our MIL ckpts cost ~12-15 min each on T4 (288px,
K=20, no TTA needed — TTA is flat for MIL); if budget is tight, drop to D1+D3
(8 ckpts, gold 0.9263, still +0.001). Our weights: Kaggle dataset
devansh26/rsna-knee-weights-r2 (latest version: r4_v2/r4_v3/r5_cnxts × 4 folds
+ src/predict.py + src/model.py — study-level probs, mean over ckpts; needs
DICOM->320px/140mm decode path identical to predict.py's load_series_volume).
Since v11 showed OOF->LB mapping saturating: judge this probe on LB directly;
gold + diversity are the offline evidence.

## Addendum 3 (2026-08-31) — upgraded MIL@384 pair on v7 targets
mil_effs384_v7 gold 0.9066 (best single family, devansh side), mil_srx384_v7 0.9024;
pair 0.9085. Stack vs v10-era L pair: +0.0022 at 1/2 MIL weight (0.9276).
Correlation vs L rose to 0.92 (shared v7 targets) — stronger but less orthogonal;
net stack gain unchanged. Recommendation: fold into the gold-conservative FINAL pick
and the learned aggregator; not proposed as a solo LB probe given 3x MIL-neutral reads.
fastMRI noisy-student lane in progress (10k external knees, pseudo-label pretrain) —
that's the lane aimed at breaking the 0.947 plateau, ETA first results ~2 days.

## Addendum 4 (2026-09-01) — NOISY-STUDENT BREAKTHROUGH (fastMRI external data)
Recipe: pseudo-label 10,010 fastMRI knee studies (47,552 series, preprocessed with the
standard 140mm/384px path) using the 8-ckpt MIL@384 ensemble -> pretrain trunk 4 epochs
on pseudo-labels -> fine-tune on v7 targets (lr 1e-4, 8 ep).

| family | v7-only | +fastMRI pretrain |
|---|---|---|
| MIL effnetv2s@384 | 0.9017 | **0.9198** |
| MIL seresnext@384 | 0.8970 | **0.9233** |

**NS pair (8 ckpts) gold 0.9265 > v10 pair 0.9254. NS + v10 pair 1:1 = 0.9284 (team best).**
Largest single-intervention gain measured by either side (+0.018-0.026/family).

@liaoyixu: this should transfer to your lane directly — pseudo-labels in
shared/labels/fastmri_pseudo_v1.csv (join on StudyInstanceUID; fastMRI knee DICOM
batches 1+2, your npz pipeline). cnxl/cnxv2l x v9 targets + fastMRI pretrain is the
obvious 0.95+ candidate. OOF files: oof_mil_ns_{effs,srx}384_nsv1.csv.
LB probe of the NS pair going out from devansh26 kernel today; seed replicas queued.

## Addendum 5 (2026-09-03) — NS pair LB CONFIRMED: 0.947 public
NS-PROBE submission (devansh26, sub 55975512, 2026-09-03): ns_effs384 + ns_srx384,
8 ckpts, no TTA, gold57 0.9265 -> **LB 0.947**. Two families trained only on our side
land within 0.003 of the team best (0.950, v21 cnxv2l x v9), at cross-team corr ~0.85.
Gold->LB mapping for the NS lane: 0.9265 -> 0.947 (consistent with the pair-family map).

Implications:
1. fastMRI noisy-student pretrain is LB-real, not a gold-57 artifact.
2. @liaoyixu — highest-priority action on your side: fastMRI pretrain under
   cnxv2l x v9/v21 (recipe Addendum 4; pseudo-labels shared/labels/fastmri_pseudo_v1.csv).
   Expected from the per-family lift (+0.018-0.026 gold): the single most likely
   0.952+ mover either side has. Everything you need is already in shared/.
3. Team stack (NS + your v10/v21 pair) is the gold-conservative final pick shape:
   offline 0.9284 team-best; to probe it on LB one kernel needs both weight sets —
   share your weights dataset with devansh26 (or vice versa; ours is
   devansh26/rsna-knee-weights-r2) and either side can push the stacked kernel.
4. devansh side in flight: convnext_small NS family (3rd NS family, jobs 3412190-94,
   ETA ~2 days incl. queue) for a wider NS ensemble; seeds measured +-0 by L so
   SUs go to families, not replicas. MLL SUs expire Sep 30 — anything you want
   trained on LS6, ask before ~Sep 25.

## Addendum 6 (2026-09-07) — NS ROUND 2: iterated noisy-student pays again
Recipe: pseudo-label fastMRI a second time with the LB-confirmed NS pair (0.9265) as
teacher -> re-pretrain trunks on fastmri_folds_v2 -> fine-tune on v7. Same budget as r1.

| family | v7-only | NS r1 | NS r2 |
|---|---|---|---|
| MIL effnetv2s@384 | 0.9017 | 0.9217 | 0.9247 |
| MIL seresnext@384 | 0.8970 | 0.9254 | **0.9300** |
| MIL convnext_s@384 | — | 0.9258 | (training) |

**ns_srx384_r2 gold 0.9300 = best single family either side** (cnxv2l ~0.9283).
NS2 pair 0.9277; NS2 pair + ns_cnxts384 = **0.9291** (12 ckpts, Kaggle-feasible).
3-family r1 average tied the pair (0.9264 vs 0.9265) — flat offline, LB will decide.

LB probe of the 0.9291 combo IN FLIGHT (kernel v9, 12 ckpts). cnxts_r2 chain training.
New OOF/gold exports: {gold,oof}_ns_{cnxts384,effs384_r2,srx384_r2}.csv (no TTA, v7 folds).

@liaoyixu: iteration compounds — +0.005/+0.046 over r1/base for srx. If you run fastMRI
pretrain on cnxv2l, plan for TWO rounds from the start; round-2 teacher can be our
NS2 ensemble (happy to re-pseudo-label with your family included once weights land in
shared/ or the Kaggle dataset).
