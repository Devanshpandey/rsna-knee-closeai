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

## Addendum 7 (2026-09-07) — NS2 12-ckpt = LB 0.950 (ties team best)
NS2-PROBE (sub 56068446): ns_effs384_r2 + ns_srx384_r2 + ns_cnxts384, 12 ckpts,
no TTA -> **public LB 0.950**, equal to team best (v21 cnxv2l x v9) from the devansh
lane alone. Gold->LB mapping holding linear: 0.9265->0.947, 0.9291->0.950.
Leader 0.952 (Brandon Low).

Read-through:
1. Iterated NS confirmed on LB twice. The lane is not saturated: each pseudo-label
   upgrade (0.909 -> 0.9265 teacher) returned ~+0.003 LB.
2. @liaoyixu: cnxv2l x fastMRI-NS is now conservatively projected >= 0.951-0.952 solo
   (your family gold 0.9283 + our measured NS lift). Two rounds planned from the start.
3. Team stack is the 0.952+ shot: our 0.950 lane x your 0.950 lane at cross-corr
   ~0.85. One kernel, both weight sets — share your weights dataset with devansh26
   (or drop ckpts in shared/) and we push it same-day.
4. In flight our side: cnxts_r2 (completes 3-family round-2 set), then optional NS
   round 3 with the 0.9291 ensemble as teacher.

## Addendum 8 (2026-09-10) — NS round 3: solo plateau, ensemble still climbing
r3 families (teacher = LB-0.950 12-ckpt ensemble): effs 0.9263, srx 0.9275, cnxts 0.9248.
srx REGRESSED vs r2 (0.9300->0.9275): solo iteration converged after 2 rounds.
@liaoyixu: plan exactly TWO pseudo-label rounds on cnxv2l — third adds nothing solo.

BUT cross-round mixing still pays (diversity, not strength):
best 12-ckpt combo = ns_cnxts384(r1) + ns_srx384_r2 + ns_srx384_r3 gold **0.9314**
(129-combo search; several independents cluster 0.9303-0.9314 > 0.9291, so real).
LB probe of 0.9314 combo IN FLIGHT (kernel v10). Projected 0.951+-0.001.

Team stack refresh (measured on gold, 1:1): our 0.9291 x your OLD cnxl384_v7 (0.9276)
= **0.9312**; x cnxxl 0.9317. Your v21-era families are stronger than v7-era, so the
true team stack is >0.932 gold ~ LB 0.952+. Weight-sharing remains the single
highest-value pending action team-wide. All 9 of our NS family exports now in shared/oof.

## Addendum 9 (2026-09-11) — NS-mix probe 0.949: gold combo-search OVERFITS
NS-MIX (sub 56140669, cnxts_r1+srx_r2+srx_r3, gold 0.9314) -> **LB 0.949**, i.e.
BELOW the NS2 combo (gold 0.9291 -> LB 0.950). The linear gold->LB map broke exactly
where selection entered: max-of-129-combos on 57 studies is overfit by ~0.002-0.003.
Rule going forward (both sides): gold-57 may VALIDATE a recipe change (pretrain,
targets) but must never SELECT among near-tied ensembles. LB-validated combos only.

Standing state: devansh lane solo ceiling = **0.950** (NS2 12-ckpt, kernel v9 combo:
ns_effs384_r2 + ns_srx384_r2 + ns_cnxts384). No further solo probes planned.
Remaining moves ranked: (1) your cnxv2l x fastMRI-NS (2 rounds) — everything staged
in shared/; (2) team-stack kernel (weights sharing); (3) our SUs are yours until
Sep 30 — name any run and we execute on LS6.

## Addendum 10 (2026-09-20) — cnxv2l x NS = LB 0.952 (new team best); board escalated
CNXV2L-PROBE v2 (sub 56403254): ns_cnxv2l_r2 + ns_srx384_r2 + ns_cnxts384, 12 ckpts
-> **LB 0.952**. Gold->LB map held again (0.9319 -> 0.952). Third LB confirmation of
the fastMRI-NS lane; convnextv2_large trained on LS6 A100-40GB via --grad-ckpt
(train.py) + 2-round pseudo-labels.

Infra that made it fit Kaggle: predict.py now shards checkpoints across ALL visible
GPUs (thread per device) — 2xT4 halves ensemble time; first attempt (single-GPU)
timed out at 9h. Both in lonestar/ (commits 52c2022, 2ca93ad). If your kernels are
single-GPU, adopt the same pattern for any heavy final ensemble.

BOARD MOVED: top now 0.957 (Scott Willis), 0.956, pack at 0.955. Our 0.952 = team
best but 0.955 needs more than either lane solo.
Ranked path to 0.955+:
1. TEAM STACK on LB — your 0.950 lane x our 0.952 lane, corr ~0.85. Weights sharing
   is the only blocker. This is worth ~+0.002-0.004 and is pure logistics.
2. Your cnxv2l(H100, 384 native) x our NS recipe, 2 rounds — should beat our
   A100-constrained 0.952 (we trained at 320 crop, ep1-trunk only).
3. Our side (in flight when SSH returns): cnxv2l x round-3 labels.
MLL SUs die Sep 30 — 10 days. Name any run you want on LS6.

## Addendum 11 (2026-09-24) — R3 swap ties (0.952); efficiency probe 0.945; single-family map penalty
R3-SWAP (sub 56509171, cnxv2l_r3 combo, gold 0.9321) -> LB 0.952 = tie. Label-iteration
axis on cnxv2l closed; devansh solo ceiling = 0.952 (confirmed twice).
EFFICIENCY probe (sub 56525046, srx_r2 ONLY 4ckpt, gold 0.9300) -> LB **0.945**:
single-family maps ~-0.005 below the combo line — diversity carries LB weight that
gold-57 does not price. Efficiency track candidate v2 = light PAIR (srx_r2+cnxts_r2,
gold 0.9301, ~3h) expected ~0.948-0.950; probing next.
Final-pick shape (current, absent team stack): MAIN = cnxv2l combo 0.952 (kernel v13/v14);
SECOND = efficiency light pair or conservative stack. Team stack still worth +0.002-0.004
and still blocked on weight sharing. SUs die Sep 30.

## Addendum 12 (2026-09-24) — probe program complete; final-pick matrix
EFF-v2 (sub 56529590, srx_r2+cnxts 8ckpt light pair) -> **LB 0.949 @ ~3h** — the
efficiency-track candidate (vs 0.952 @ ~5.5h for the full combo).

Measured LB ladder (all devansh26 kernels, reproducible):
  0.945  srx_r2 only, 4ckpt (~2h)        — single-family penalty measured
  0.949  srx_r2+cnxts, 8ckpt (~3h)      — EFFICIENCY pick candidate
  0.950  effs_r2+srx_r2+cnxts, 12ckpt   — kernel v9
  0.952  cnxv2l_{r2,r3}+srx_r2+cnxts    — kernel v13/v14, MAIN pick candidate
Current picks absent team input: MAIN = v13 or v14 (tied), SECOND = eff-v2 or a
conservative variant. @liaoyixu: the team stack remains unclaimed (+0.002-0.004,
one weight-share away) and SUs on LS6 die Sep 30.

## Addendum 13 (2026-10-12) — BLEND-NS 0.957: public stack + our NS leg
Board moved to 0.966 via public notebooks (heliosli two-way blend 0.956 = Raptor
CoAtNet + ConvNeXt-S6, percentile-rank blend). Our counter: 3-way equal rank blend
adding our fastMRI-NS light pair as a decorrelated leg -> **LB 0.957** (sub 57117428,
kernel devansh26/rsna-knee-blend-ns v3). +0.005 over our solo lane, +0.001 over the
public two-way. Infra notes: API-pushed kernels cannot pin dataset versions (submission
rejected "non-versioned datasets") — fixed via a byte-identical single-version mirror,
integrity-checked by the notebook's own bundle-id assert; S6 deadline cut 8.3->5.6h,
NS leg adaptive (pair/srx-only/skip).
Next probe: reweight 0.25/0.25/0.5 (public legs are inter-correlated; the independent
leg deserves more mass). CAVEAT shared with the whole LB top: Raptor weights are
OAI-pretrained and OAI eligibility is an open organizer question.
