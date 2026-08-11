# CloseAI Unified Pipeline Plan — RSNA Knee Abnormality Detection

**Audience:** liaoyixu's Claude agents on TACC Stampede3 (4× H100).
**From:** devansh (minato-side pipeline, TACC Lonestar6).
**Date:** 2026-08-11 · deadline Oct 22 · submissions 5/day shared, 2 finals.
**Goal:** merge the two team pipelines' complementary strengths and take top-1 (0.943+).

---

## 0 · Why merge the pipelines (30-second summary)

Two independently-built pipelines exist on this team. Head-to-head:

| Component | liaoyixu (Stampede3) | devansh (LS6) | Winner |
|---|---|---|---|
| Report extractor (gold-58 macro AUC) | severity-graded, ≈ **0.84** | rubric-calibrated Qwen 14B+7B soft ensemble, **0.876** | **devansh** |
| Training recipe | 320–384 px, physical 140 mm crop, mirror-canonical + label-swap aug, EMA, K=24 + Fourier pos, 6 ep | 224 px, no mirror, 10 ep (now upgraded to match yours) | **liaoyixu** |
| Fold hygiene | scanner ∪ byte-identical-report GroupKFold | was leaky (171 studies across 49 dup-report clusters — since fixed) | **liaoyixu** |
| Architecture | per-series 2.5D → 0.5·(mean+max) study aggregation | study-level MIL: slice-attn → series-attn → 12 logits | **complementary** |
| Public LB | **0.938** (#5) | 0.918 | — |

liaoyixu's report lists "Extractor v6+ — better report labels" as *the deepest lever, since label
noise now binds everything*. That lever is ready: devansh's extractor is +3.5 points on gold.
The single highest-expected-value action in this document is **Task 1**.

---

## 1 · Assets devansh provides (schemas + where they live)

All in the `RSNA_knee` git repo (devansh will share the repo / files directly):

### 1.1 Extractor soft labels (the main asset)
- `results/labels_llm_14b_v2.csv` and `results/labels_llm_7b_v2.csv`
- Schema: `StudyInstanceUID, parse_ok, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`
- Values are soft confidences in **[0,1]** (0–100 rescaled). 14B has 4407/4407 parsed; 7B has
  4324/4407 (83 NaN rows — fall back to 14B where 7B is NaN).
- Recommended teacher = `mean(14B, 7B)` with NaN-safe averaging.
- **Join key is StudyInstanceUID — never row order.** (Your fold-hash/row-order discipline noted;
  re-align to your labels-parquet v4 order after joining and verify fold-hash.)

### 1.2 The rubric that made the extractor good (for building extractor v7)
Model: Qwen2.5-14B-Instruct + 7B via vLLM, temperature 0, strict JSON, integer 0–100 per label.
Full prompt in `scripts/labeling/extract_labels_llm.py`. Calibration rules discovered by reading
every gold-58 disagreement (these ARE the difference between 0.81 and 0.876):
- **ACL:** only fiber-discontinuity tears / rupture / failed graft = present. Grade 1–2 signal,
  "interstitial tear" with intact fibers, mucoid/cystic degeneration, bony avulsion without
  ligament tear = absent.
- **MCL:** grade ≥2 sprain or tear = present. Grade 1 sprain, periligamentous edema/fluid around
  intact ligament, chronic thickening, retinaculum injury = absent.
- **Menisci:** tear must reach an articular surface (or be root/radial/complex/bucket-handle/
  displaced/macerated). Intrasubstance/mucoid degeneration, extrusion alone = absent.
- **Effusion:** moderate/large/unqualified = present. Trace/small/minimal/mild/"some"/physiologic
  = absent. Never inferred from injury severity.
- **Contusion vs Fracture:** marrow edema surrounding a fracture line → Fracture, not Contusion.
  Subchondral edema under a chondral lesion → neither (degenerative).
- **PF OA:** includes chondromalacia / patellar or trochlear chondropathy.
- **Fracture:** any (subchondral, occult, avulsion incl. Segond, impression, insufficiency);
  watch non-English terms (kırık, Fraktur, фрактура, κάταγμα, fisura).
- ~1–2 cases/label in gold-58 contradict their own report (image-derived labels) → extractor
  ceiling ≈ 0.95; don't over-tune to gold (n=58, σ≈0.0125 — matches your estimate).

### 1.3 Per-label trust map (from devansh's round-2 blend, tuned on gold-58)
When blending extractor vs image-model-OOF per label, the optimum was NOT uniform 0.5:

| Trust the images (w_model ≥ 0.75) | Trust the reports (w_model ≤ 0.25) |
|---|---|
| Synovitis (1.0), Effusion, Baker's, Contusion, Fracture | ACL (0.0), Lateral OA (0.0), Medial/Lateral Meniscus, PF OA |
| | MCL, Medial OA at 0.5 |

Interpretation: image-conspicuous findings (fluid, cysts, marrow edema) — the model outgrows the
reports; interpretation-heavy findings (ligament grading, compartment attribution) — reports win.
Your planned v6 = uniform 0.5/0.5 leaves value on the table; per-label w is worth one experiment
(**Task 2**). Note this is target-blending (generalized in our runs: +0.02–0.03 gold across all
folds) — distinct from per-label *ensemble* weights, which both teams independently found do NOT
generalize.

### 1.4 Model zoo for the team ensemble (architecturally different from yours)
Study-level MIL: shared timm backbone over K=16 slice-triplets/series → attention pool over
slices → +plane/fluid metadata embedding → attention pool over series → 12 logits.
Trained checkpoint families (4 folds each, on round-2 blended targets):
- `r2_v2` tf_efficientnetv2_s @224 — gold 0.897 avg (best single family)
- `r2_v3` seresnext50_32x4d @224 — gold 0.894 avg
- being retrained now at your recipe (320 px/140 mm, mirror-swap aug, EMA) once LS6 maintenance ends
- OOF + gold predictions per fold: `results/<family>/fold*/{val_preds,gold_preds}.csv`
  (schema: 12 label columns + StudyInstanceUID; probabilities in [0,1])

### 1.5 Negative results — do not re-spend GPU on these
Replicated or established on LS6 (several also match your findings):
1. Capacity: tf_efficientnetv2_m (54M) ≤ efficientnetv2_s (21M) on identical targets (0.885 vs 0.897).
2. Resolution without physical crop: native 256 full-FOV < 224 random-crop (0.880 vs 0.897).
   (Your 320 px works because of the 140 mm crop — pixels must go to anatomy.)
3. Uniform mega-ensembles: 12–16 checkpoints ≤ best 4 (0.902 vs 0.907) — same saturation you saw
   at OOF ≈ 0.896.
4. Statistical Synovitis relabeling from correlated outputs: LOO 0.70 < model-alone 0.76 (n=58
   can't support even 2 fitted params).
5. K=24 slices at inference on K=16-trained models: ≈ +0.002 (you saw the same).
6. Per-label ensemble blend weights: doesn't generalize (both teams, independently).

---

## 2 · Task list for Stampede3 agents (priority order)

### Task 1 — Retrain best families on devansh's extractor labels 【highest EV】
- Build targets: `T_dev = nanmean(labels_llm_14b_v2, labels_llm_7b_v2)` joined on
  StudyInstanceUID, re-ordered to labels-parquet v4 row order, fold-hash verified.
- Retrain your two best 384 px families (cnxt-s@384, effnetv2_m@384) on `T_dev`, everything else
  identical (EMA, mirror-swap, 6 ep).
- **Acceptance:** pooled OOF and gold-58 vs the same families on v5 labels. Expect +0.5–1.5 OOF
  points if the extractor gap transfers (0.84→0.876 teacher = large under your +1.3 LB/OOF-point
  mapping). If gold improves but OOF drops, remember OOF is now measured against different
  (better) targets — compare on gold + one LB probe, not OOF alone.
- Cheap variant worth one run: `T_blend = mean(T_dev, v5_severity)` — two independent extractors
  ensemble like two models.

### Task 2 — Extractor v7 targets: per-label blend instead of uniform 0.5/0.5
- Your v6 = 0.5·extractor + 0.5·ensemble-OOF. Replace with per-label
  `w_l·OOF + (1−w_l)·T_dev`, where `w_l` from the §1.3 map (or refit on gold-58 with w capped at
  0.75 unless extractor gold-AUC < 0.75; grid {0,.25,.5,.75,1}).
- Devansh's implementation to port: `scripts/training/make_round2_targets.py`.
- **Acceptance:** blended-target quality vs gold (devansh's run: mean 0.93 vs 0.876 teacher);
  then one 384 px family retrained on v7 vs v6, judged on gold + LB probe.

### Task 3 — Unify folds and exchange OOF (enables the learned team ensemble)
- Canonical folds = yours (scanner ∪ report-cluster union-find). Publish
  `folds_v4.csv` (`StudyInstanceUID, fold`) + fold-hash to the shared repo; devansh regenerates
  LS6-side OOF on those folds after retraining.
- Exchange format per model family: `oof_<family>.csv` and `gold_<family>.csv`
  (StudyInstanceUID + 12 prob columns), plus a manifest line
  (family, arch, px, targets-version, folds-hash, train date).
- **Acceptance:** a stacking notebook that fits the learned study-aggregation / LGBM layer on the
  union of both teams' OOF under honest per-fold evaluation. Two architecture lineages
  (per-series mean+max vs study-MIL attention) is exactly the diversity uniform averaging
  saturated without.

### Task 4 — Keep your active work unchanged
Learned study aggregation (LGBM over per-series stats), effnetv2_m@384, greedy family-subset
selection on OOF — all unchanged; they compose with Tasks 1–3.

### Task 5 — Synovitis / PF OA label rescue (background, human-gated)
- Both extractors and both model zoos are stuck (Synovitis ≈ 0.73–0.78 on gold; self-training
  made it worse — recycled beliefs, no new information).
- devansh has a case-review notebook (borderline-case queue, ~90 s/case) if any musculoskeletal
  radiologist contact can donate an hour. 30–50 calls → per-label recalibration set. Until then,
  accept the ceiling; do not spend H100 time here.

---

## 3 · Validation & submission discipline (team-wide)

1. **Gold-58 is sacred:** never in training, never in target-fitting beyond the capped per-label
   w grid; σ ≈ 0.0125 ⇒ deltas < ~0.01 on gold are noise. Decide on pooled OOF + gold direction
   + LB probes.
2. **OOF→LB mapping** (+1.3 LB per OOF point, stable so far) is the planning tool; recompute
   after the target change in Task 1 (mapping shifts when the OOF yardstick changes).
3. **Submission budget:** 5/day shared. Reserve probes for: T_dev retrain (Task 1), v7 vs v6
   (Task 2), stacked team ensemble (Task 3). Avoid LB-probing anything decidable offline.
4. **Final 2 picks (Oct):** one CV/gold-conservative, one LB-best. Public↔private prevalence
   shift is explicit in the rules — teams above us overfitting the public board is expected.
5. **Winners' obligations** (if it lands): open-source code+weights, video, forum post. Both
   repos are git-clean; keep it that way.

## 4 · Infrastructure crib sheet (merged hard-won constraints)

- Kernel push: kagglesdk 0.1.28, `machine_shape="NvidiaTeslaT4"` (P100 pool lacks sm_60 torch
  support; CLI default lands there). devansh needs a copy of `push_kernel.py`.
- Dataset-version race: `status` says ready while processing — verify with
  `kaggle datasets files` before kernel push, else the kernel silently attaches stale files.
- cu118 torch+torchvision wheel dataset (`rsna-torch-cu118`): replace both together or
  `import timm` breaks; Kaggle strips `+` from wheel names — restore before pip install.
- Autocast: bf16 on ≥sm_80, fp16 on T4 (sm_75) — gate on
  `torch.cuda.get_device_capability()`.
- `/kaggle/input` layout is now `competitions/<name>` and `datasets/<user>/<name>` — search both.
- Defensive kernel: write a prior-rate fallback submission first; skip undecodable series;
  studies with no usable series get extractor prior rates.
- LS6 side quirks (for shared scripts): login nodes have 8 GB vmem (stream big downloads with
  curl on the signed URL; HF downloads with `max_workers=1`, `HF_HUB_DISABLE_XET=1`); never
  pip-install into an env while jobs are running from it.

## 5 · Current joint scoreboard & the path to 0.943+

- Team best 0.938 (#5); top-1 0.943. Gap ≈ 0.005 = ~0.4 OOF points under the mapping.
- Expected contributions: Task 1 (extractor upgrade, biggest), Task 2 (per-label targets),
  Task 3 (cross-architecture stack). Any one at its midpoint estimate closes the gap;
  together they overshoot it.
- devansh's LS6 track (after maintenance): retrain study-MIL families at the unified recipe on
  unified folds → deliver OOF/gold/test predictions per §3's exchange format.

*Questions → devansh. Everything referenced is committed in the RSNA_knee repo.*
