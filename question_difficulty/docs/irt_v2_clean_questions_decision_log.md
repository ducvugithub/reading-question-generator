# IRT v2: Clean-Question Rebuild — Decision Log

## What we learned (2026-11-13/14)

FiLM difficulty-conditioning (v2-v8, see `question_generation/docs/difficulty_steering_mechanisms.md`)
plateaued on quality metrics that looked good by exact-string comparison but
collapsed under scrutiny:

- **Exact-string vs. semantic collapse rate (v8)**: full collapse 9% (exact
  string) vs. **19.3%** (embedding cosine sim > 0.85). Fully-distinct 52%
  (exact) vs. **29.3%** (semantic). The naive metric overstated real
  differentiation by ~2x.
- **Template consistency**: clustering all v8 generations by embedding, 0/20
  clusters (size>=5) reach even 60% purity for one difficulty class — no
  question template (including "best title", "which of the following is
  TRUE") has a stable, learnable link to difficulty. Best cluster (inference
  phrasing -> HARD) only hit 59%.

This pointed at the **label**, not the model. Direct check on the IRT
difficulty scores themselves (`irt_rasch_mcfiltered_difficulty.json`, 9,236
items, independent of any QG training):

- **Near-duplicate questions across different passages** (cosine sim > 0.85,
  68,115 pairs) score barely better than random pairs on `|score diff|`:
  1.53 vs 1.75 raw, 1.22 vs 1.38 after passage-centering.
  correlation(similarity, |score diff|) ~= 0 in both cases.
- **Bucket agreement**: near-duplicate questions land in the same
  EASY/MEDIUM/HARD bucket only **35.2%** of the time vs **33.1%** for random
  pairs (chance = 33.3%).
- Concrete example: the literal question "What does the passage mainly tell
  us?" appears as EASY, MEDIUM, and HARD across different passages with no
  visible pattern. Even within the SAME passage, near-identical question
  templates ("We can infer that _" vs "It can be learned that _") get
  meaningfully different scores (0.66 vs 1.32).

**Conclusion**: the cascade-based Rasch IRT difficulty is too noisy at the
item level to be a learnable difficulty label — not a normalization/centering
bug (centering removes passage-mean offset correctly, but the noise is in the
per-item residual, not the passage mean).

## Suspected contributor: cloze-style questions

Looked at whether question *format* explains some of the noise. Of 76,804
MC-dependent-filtered pool items, **44,725 (58%) are cloze/fill-in-the-blank**
("We can infer from the passage that __.") vs **32,079 "clean"** (real
question sentences). Hypothesis: fill-in-the-blank items are more ambiguous
for the answerer cascade to judge consistently (blind cascade matches the
blank completion against 4 options without the semantic anchor a real
question gives), adding noise on top of the already-weak signal.

## Decision: rebuild from scratch, clean questions only

1. **Filter**: clean (non-cloze) questions only, passages with >=3 clean
   questions (not >=2 — more robust per-passage centering/ranking, less
   sensitive to one noisy item swinging the passage mean).
   Result: **13,449 questions across 3,964 passages** (avg 3.39/passage).
   Of these, only 1,549 already have cascade results — **~11,900 need a
   fresh cascade run**.
2. **Re-run the answerer cascade** on the ~11,900 new items.
3. **Re-map/merge** cascade results -> scored pool. Skipping the LLM
   (Haiku) tiebreaker this round (cost/time); revisit if the resulting IRT
   fit is still too noisy to tell whether tiebreaking would have helped.
4. **Re-fit Rasch IRT** on the new clean-only pool.
5. **Rebuild QG data + retrain FiLM from scratch** on the new labels.

Not yet done: no evidence yet that removing cloze questions actually reduces
the noise (it's a hypothesis, not a confirmed cause) — the near-dup validity
check should be re-run on the new IRT fit before committing further training
budget.
