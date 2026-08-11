# RSNA Knee Abnormality Detection — Team CloseAI

Private team monorepo. Two parallel pipelines, one exchange area.

```
lonestar/    devansh's pipeline (study-level MIL, LLM extractor, LS6 3xA100)
stampede/    liaoyixu's pipeline (per-series 2.5D, severity extractor, Stampede3 4xH100)
shared/      team contracts — the ONLY cross-pipeline interface
  labels/    extractor outputs (soft labels, join on StudyInstanceUID)
  folds/     canonical fold assignment (folds_v4.csv + fold-hash) — liaoyixu's union-find folds
  oof/       out-of-fold + gold predictions per model family (see docs/team_merge_plan.md 3)
docs/        merge plan, method reports, solution write-ups
```

Rules of the road:
- Push directly to main; paths are disjoint so conflicts can't happen. Rebase, don't merge-commit.
- Nothing from `data/raw` (DICOMs, report text) is ever committed.
- `shared/` files are append/version, never edit in place: bump a version suffix.
- Repo stays PRIVATE until competition end (public sharing during the comp = sharing with all competitors); it becomes the open-source vehicle for winners' obligations after.

Start here: `docs/team_merge_plan.md`.
