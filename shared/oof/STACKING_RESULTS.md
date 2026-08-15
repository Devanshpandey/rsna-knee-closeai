# Knee-MRI Ensemble Study — Decision Document

Baseline = current LB 0.946 config: uniform prob-avg {cnxt384_v7, v2m384_v7} = **0.91203 pooled OOF / 0.92439 gold** (pipeline validation hit the required 0.9120/0.9244 exactly, all 4 strategies).
Metric: y binarized at 0.5, per-label AUC over all 4268 pooled OOF studies, unweighted mean of 12 labels. Gold n=57 (sigma ~0.0125) — direction check only, never fitted on. "Verified" = independent recomputation by the audit agent (CONFIRMED, no leakage found).

**Bottom line: ship uniform prob-avg {cnxb384_v7, v2m384_v6}. Same kernel cost as today (8 L models @384), +0.0020 OOF, fold-consistent 4/4, gold agrees, zero fitted parameters, verified.**

## 1. Deployable configs (kernel-feasible, <=8 L models @384), ranked by pooled OOF

| # | Config (uniform weights unless noted) | Method | OOF | Gold | dOOF | dGold | L/D models | Verified | Folds beating baseline |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cnxb384_v7 + v2m384_v6 | rank-avg | 0.91413 | 0.9253 | +0.0021 | +0.0009 | 8/0 | YES (0.914132/0.925269) | 4/4 |
| 2 | cnxb384_v7 + v2m384_v6 | prob-avg | 0.91406 | 0.9254 | +0.0020 | +0.0010 | 8/0 | YES (0.914056/0.925387) | 4/4 |
| 3 | 0.975·(row 2) + 0.025·(tta_r4_v2+tta_r4_v3)/2 | prob-avg | 0.91408 | 0.9260 | +0.0021 | +0.0016 | 8/8 | no | 4/4 vs baseline; only 2/4 vs row 2 (dOOF +0.00002) |
| 4 | cnxb384_v7 + v2m384_v7 | rank-avg | 0.91320 | 0.9242 | +0.0012 | -0.0002 | 8/0 | no | 4/4 |
| 5 | cnxt384_v7 + v2m384_v6 (one-member swap of current) | rank-avg | 0.91296 | 0.9254 | +0.0009 | +0.0010 | 8/0 | no | 3/4 |
| 6 | v2m384_v6 + v2m384_v7 | rank-avg | 0.91263 | 0.9229 | +0.0006 | -0.0015 | 8/0 | no | 4/4 (single-backbone; weak gold) |
| 7 | cnxt384_v6 + v2m384_v6 (v6-pair) | prob-avg | 0.91264 | 0.9228 | +0.0006 | -0.0016 | 8/0 | no | 2/4 |
| — | BASELINE cnxt384_v7 + v2m384_v7 | prob-avg | 0.91203 | 0.9244 | 0 | 0 | 8/0 | YES | — |
| — | cnxt384_v8 + v2m384_v8 (v8-pair) — AVOID | prob-avg | 0.90658 | 0.9270 | -0.0055 | +0.0026 | 8/0 | no | 0/4; high gold is n=57 noise |

Over-ceiling references (NOT deployable): best trio cnxb384_v7+v2m384_v6+v2m384_v7 rank-avg 0.91378/0.9241 (12 models); modern6 uniform 0.91311/0.92559 (24); all-8-modern 0.91139 (32); uniform all-18 0.90719 (72). Bigger is strictly worse than the best pair — v5/v8/tdev members dilute.
Prob-avg vs rank-avg ties everywhere (|d| <= 0.0001). Prob-avg is simpler in-kernel and needs no test-population ranking. cnxb384_v7 appears in every top config.

## 2. Cross-team L:D mix (answer for devansh)

Grid: w·L-anchor + (1-w)·D-mix, D-mix = uniform(tta_r4_v2, tta_r4_v3) (TTA variants preferred; plain+TTA never mixed). Best cell per anchor:

| L anchor | Best w | Implied L:D | dOOF vs pure-L | Fold-consistent? | Note |
|---|---|---|---|---|---|
| cnxb384_v7+v2m384_v6 (best) | 0.975 | 39:1 | +0.00002 | 2/4 | statistically identical to pure L |
| cnxt384_v7+v2m384_v7 | 0.975 | 39:1 | +0.00006 | 3/4 | at w=0.8 (4:1) ALL 4 folds negative |
| cnxt384_v6+v2m384_v6 | 0.975 | 39:1 | +0.00003 | 3/4 | deltas <= 0.0001 |
| cnxb384_v7+cnxt384_v7 | 0.975 | 39:1 | +0.00004 | 2/4 | best-gold anchor |
| cnxt384_v8+v2m384_v8 | 0.925 | 12:1 | +0.00018 | 2/4 | weakest anchor overall |
| modern6 (6 L families) | 1.000 | pure L | 0 | — | no D weight helps at all |

**The predicted 2:1–4:1 sweet spot did NOT hold.** At 3:1 (w=0.75) OOF drops in all 4 folds on the best anchor (-0.0014/-0.0012/-0.0026/-0.0022); same all-negative pattern at 4:1 on the v7 pair. Gold consistently prefers heavier D (e.g. +0.0025 gold at 3:1) but gold is the same 57 studies shared across every grid cell — direction hint only, do not ship on it. D-team alone: 0.8718 OOF, far below every L anchor (0.9066–0.9141). Rank-avg mixing was uniformly worse than prob-avg. Recommendation: D weight 0; if D goes in as test-diversity insurance, cap at <=5–10% (w>=0.9) — costs ~0 OOF plus 8 cheap @288 models.

## 3. Learned stackers: NO-GO

| Method (18 families in) | Protocol | OOF | Gold | vs modern6 (0.91311) | vs best pair (0.91413) |
|---|---|---|---|---|---|
| logreg, rank features, nested C | nested inner-3-fold CV | 0.91408 (repro 0.91483; seed ±0.0008) | 0.929 (repro 0.926) | +0.0010, wins 4/4 folds | ~tie |
| logreg, rank features, fixed C=1 | fold-honest | 0.91349 | 0.928 | +0.0004 | -0.0006 |
| logreg, raw probs, fixed C=1 | fold-honest | 0.91262 | 0.926 | -0.0005 | -0.0015 |
| NNLS simplex, raw | fold-honest | 0.91226 | 0.9248 | -0.0009 | -0.0019 |
| LGBM (fixed a-priori hparams) | fold-honest | 0.90682 | 0.9236 | -0.0063 | dead end |

**Verdict: NO-GO for deployment.** The best stacker only TIES the free, unfitted best pair while requiring all 18 families in-kernel (56 L + 16 D models, ~7x over ceiling). It is honest (nested C; fold-honest fits; rank transform is label-free — a strictly train-only ECDF variant scores the same, 0.91470–0.91497, so transductive ranking contributes ~+0.0001). It contains plain+TTA duplicates; the rule-compliant 16-family variant is slightly BETTER (0.91503) but still infeasible. Context features hurt in every matched comparison; trees overfit at n~3200/label. Stacking is a real but tiny effect (+0.001 over the best uniform baseline) that cannot be shipped at current kernel budget.

## 4. Greedy selection optimism: measured at zero

Deploy greedy (selection on pooled OOF, all 18 families, mil/tta exclusion enforced) picks {cnxb384_v7, v2m384_v6} and stops at 2 — the best 3rd addition is negative. Nested fold-honest greedy selects the IDENTICAL 2-family set in all 4 folds, for prob-avg and rank-avg, restricted (<=3L+<=2D) and unrestricted: **selection optimism = 0.000000**. (The rank-avg nested headline 0.913599 vs deploy 0.914131 is a ranking-population artifact — within-held-out-block vs full-population ranks; re-ranking the same sets over the full population reproduces 0.914131 exactly.) Residual caveat: the pair-scan winner still carries mild winner's-curse as an LB estimate; gold direction (+0.0009/+0.0010) supports the gain.

## 5. Recommended LB probes (mapping: +1.3 LB per 1.0 OOF; current LB 0.946, leader 0.949)

| # | Probe | dOOF vs current | Expected LB | What it decides |
|---|---|---|---|---|
| 1 | {cnxb384_v7, v2m384_v6} uniform prob-avg — drop-in, same 8-model cost | +0.00203 | ~0.9486 | New default. Does the best-pair gain transfer to LB (winner's-curse check)? If >=0.948 we are within one probe of the leader. |
| 2 | Same pair, rank-avg over the test batch | +0.00210 | ~0.9487 | prob vs rank on real test — free, same 8 models. Expected tie; if tie, keep prob-avg permanently (simpler). |
| 3 | Same pair at w=0.975 + 0.025 D-mix (tta_r4_v2 + tta_r4_v3) | +0.00205 | ~0.9486 (gold hints +0.0006) | Devansh's question on real test data: does a small D touch add anything? If not, D is out for good. |

Probes 2–3 are only worth spending if submissions are not scarce; both are within LB noise of probe 1 by design — they resolve method questions, not score.

## 6. Caveats — what was NOT tested, where optimism can hide

- **Winner's curse:** the best pair was chosen by scanning pairs/greedy on the same pooled OOF. Nested greedy bounds selection optimism at 0.000000 and gold agrees, but the LB estimate can still be ~0.001 optimistic.
- **Gold overuse:** gold (n=57) was consulted across hundreds of comparisons as a direction check; never fitted on, but repeated looks create soft selection pressure. The v8-pair (gold 0.927, OOF 0.9066) shows how noisy it is.
- **NOT tested:** non-uniform L-L pair weights (only 50/50; the D-grid suggests little headroom near uniform); per-label family selection or per-label prob/rank mixing; calibration/temperature scaling; TTA for L families; trios under a raised kernel budget (best trio 0.91378 needs 12 models — revisit only if budget grows); small feasible stackers (logreg on just the 2–3 shipped families) — untested middle ground.
- **Stacker stability:** logreg nested result is seed-sensitive (±0.0008); its gold 0.929 reproduced at 0.926.
- **LB mapping:** +1.3 LB/OOF is fitted from few team submissions; treat expected-LB values as ±0.001.
- **Rank-avg in-kernel** requires ranking jointly over the whole test population; if the kernel ever scores in chunks, use prob-avg.
- **Fold heterogeneity:** fold 3 is systematically easy (~0.928 for all configs), fold 2 hard (~0.909). Config RANKING was stable across folds, but absolute deltas on a fold-2-like private set would shrink.
