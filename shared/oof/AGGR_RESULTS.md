# AGGR_RESULTS — Study Aggregation for the Deployed Pair {cnxb384_v7, v2m384_v6}

Date: 2026-08-14. Data: 50/50 ensemble of the two OOF series-pred parquets, inner join
(0 drops; metadata and fold columns identical), 4268 studies, pooled OOF macro-AUC over 12 labels.
Pipeline validation: baseline 0.5*(mean+max) = **0.913762** — inside the required [0.9135, 0.9150].
All deltas below are vs this baseline.

## 1. Ranked results

| # | Method | Pooled OOF | Delta | Folds > base | Verified? |
|---|--------|-----------|-------|--------------|-----------|
| 1 | **top-4 mean** (mean of 4 largest series probs) | 0.914677 | +0.000915 | **4/4** | **YES — CONFIRMED, grid reproduced to 6 dp, no leakage** |
| 2 | top-3 mean | 0.914538 | +0.000777 | **4/4** | yes (same verified grid) |
| 3 | softmax-weighted mean, T=0.5 | 0.914119 | +0.000358 | 3/4 | no |
| 4 | 0.8*mean + 0.2*max | 0.914055 | +0.000294 | 3/4 | no |
| 5 | 0.8*rank(base) + 0.2*rank(rank-LR) | 0.914033 | +0.000271 | 4/4 | no (needs deployed LR) |
| 6 | power mean p=2 | 0.913989 | +0.000227 | 3/4 | no |
| 7 | fluid-sensitive-only 0.5*(mean+max) | 0.913989 | +0.000228 | 2/4 | no |
| — | **BASELINE 0.5*(mean+max)** | 0.913762 | — | — | yes (reproduced) |
| 8 | rank blend base × LGBM, a=0.25 | 0.913654 | −0.000108 | 4/4 (pooled loss) | no |
| 9 | per-plane 0.5*(mean+max), max over planes | 0.912587 | −0.001175 | 0/4 | no |
| 10 | best fitted model (per-label LR on rank feats) | 0.910701 | −0.003061 | 2/4 | no |
| 11 | all other fitted (5× LR / LGBM, 14–28 feats) | 0.9049–0.9101 | −0.0037 to −0.0088 | ≤3/4 | no |

## 2. Unfitted-grid landscape

0.5*(mean+max) is already very close to optimal: every unfitted family (w-blend, power mean,
softmax mean, anatomical priors) moves the pooled macro-AUC by at most ±0.0004, with smooth
unimodal curves whose optima sit near the baseline (w-blend peaks at w=0.8, +0.0003). The one
direction with real structure is top-k mean — a k=3..4 plateau (+0.0008 to +0.0009, 4/4 folds
for both k) that dominates every other variant — while all fitted stackers (LR/LGBM, 14–28
features, fold-honest or nested) lose 0.003–0.009 outright.

## 3. Decision: NO-GO (at the stated bar) — near-miss flagged

Bar: delta >= +0.001, fold-consistent, verified, kernel-implementable without labels at test time.
Best candidate, top-4 mean, scored against it:
- delta >= +0.001: **FAIL** — +0.000915, short by 0.000085 (and carries mild grid-selection
  optimism; the corroborating k=3 cell is +0.000777).
- Fold-consistent: PASS (4/4 folds, the only variant to do so among pure aggregators).
- Verified: PASS (independent reproduction to 6 decimals; no leakage; join clean).
- Kernel-implementable, label-free, zero runtime: PASS.

So the strict answer is **keep 0.5*(mean+max)**. However, this is a near-miss, not a null result:
top-4 mean passes every criterion except magnitude, has no demonstrated downside in any fold, and
costs nothing. If the owner accepts ~+0.0009 as material at zero cost, the swap below is safe.

## 4. Formula (only if the bar is relaxed to accept +0.0009)

Per label, per study: `score = mean(sorted(series_probs, descending)[:min(4, n_series)])`
(99.9% of studies have >=4 series; median 5, so this typically drops the 1–2 lowest series.)

## 5. Caveats

- The +0.000915 is the best cell of a 5-point unfitted grid selected on pooled OOF; full grid was
  reported per protocol and the k=3–4 plateau limits, but does not eliminate, selection optimism.
- OOF is for this exact 50/50 pair; if the deployed ensemble weights or model families change,
  re-check k (though the k=3–4 plateau suggests robustness).
- Fold-level variance is large (fold macro-AUCs 0.910–0.930); a +0.0009 pooled gain is well within
  what public/private LB noise can mask on a hidden test set.
- Fitted aggregators are conclusively ruled out at this data size (~3200 train studies/fold):
  every one of 7 fitted variants lost pooled, including nested-CV-tuned LR.
