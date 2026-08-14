# devansh MIL-family OOF (folds_v4, hash 17161916260226758772)
- oof_mil_effnetv2s_288.csv / oof_mil_seresnext50_288.csv — strictly out-of-fold on folds_v4; NO TTA
- gold_*.csv — mean of 4 fold models on gold-58 (models never saw gold); NO TTA
- arch: study-level MIL (backbone->slice-attn->series-meta->series-attn->12 logits)
- data: 320px/140mm-crop volumes, crop 288, k=20, mirror-aug+label-swap, EMA 0.999, 8 ep, lr 1.5e-4
- targets: per-label blend of devansh extractor + 5-generation model OOF (train_folds_v4_r2.csv)
- trained 2026-08-13 LS6
