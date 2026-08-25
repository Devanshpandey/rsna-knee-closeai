# devansh MIL-family OOF (folds_v4, hash 17161916260226758772)
- oof_mil_effnetv2s_288.csv / oof_mil_seresnext50_288.csv — strictly out-of-fold on folds_v4; NO TTA
- gold_*.csv — mean of 4 fold models on gold-58 (models never saw gold); NO TTA
- arch: study-level MIL (backbone->slice-attn->series-meta->series-attn->12 logits)
- data: 320px/140mm-crop volumes, crop 288, k=20, mirror-aug+label-swap, EMA 0.999, 8 ep, lr 1.5e-4
- targets: per-label blend of devansh extractor + 5-generation model OOF (train_folds_v4_r2.csv)
- trained 2026-08-13 LS6

# stampede study-MIL@384 (folds_v4, hash 17161916260226758772)
- oof_mil384_v7.csv / gold_mil384_v7.csv — devansh MIL arch (study_model.py) x stampede recipe:
  npz384_study (384px/140mm), K=16, max-series 6, v7_pl targets, plane-aware mirror+label-swap,
  EMA 0.999, bf16, 6 ep OneCycle lr 6e-5, bs2 accum2; best.pt on weak-val; NO TTA.
- scores: OOF (v5-aligned yard) 0.8926, gold-57 0.9118 — weak as a solo family, but
  OOF corr vs v10 pair 0.937 (pair internal 0.978) and pair+MIL gold-57 0.9254->0.9272
  (+0.0018 at 1/5-1/3 weight, flat across weights). OOF-yard reads negative (-0.001) —
  known decoupled; gold+diversity is the evidence.
- per-fold best weak-val (v7 targets, inflated): 0.9458/0.9453/0.9347/0.9406, all at ep5
  (still rising) -> 10-ep retrain launched 2026-08-25 on Stampede3.
- trained 2026-08-24 Stampede3 4xH100, ~1.5h/fold; ckpts /scratch/09059/xliaoyi/kaggle/runs/mil384_v7_f*/
