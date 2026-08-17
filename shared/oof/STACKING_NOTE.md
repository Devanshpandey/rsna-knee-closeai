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
