# Experiment Plan: Difficulty- and Focus-Span-Controlled Question Generation

Project-wide roadmap. Module-specific detail lives in each module's own
`docs/` — this file stays a high-level index + current status, not a
duplicate of the detail.

## Research Questions

1. Can a learned QDE reliably classify per-*question* difficulty (not just
   passage-subset identity)? See `question_difficulty/docs/cognitive_difficulty_estimation.md`.
2. Does difficulty-token conditioning (RACE++) produce measurably
   difficulty-differentiated questions? **Current answer: no, not with the
   current data — see status below.**
3. Does conditioning on a focus span (HotpotQA comparison-type evidence
   sentences) shift questions toward inference rather than span-locating?
4. Does combining both controls produce the best difficulty alignment and
   inference quality? (blocked on Q2)

---

## Current status (updated from this session's findings)

| Model type | Status |
|---|---|
| `baseline-race` | Trained (t5-base, flan-t5-base) |
| `diff-control-race` | Trained (t5-base, flan-t5-base). **Token conditioning does not produce measurable difficulty steering** — forced-token generation test showed no more variation than `baseline-race`'s pure sampling noise. Root cause: every RACE passage belongs to exactly one difficulty subset, so the token is collinear with passage style during training. Full writeup: `question_generation/docs/training_details.md`. |
| `baseline-hotpot` / `focus-control-hotpot` | Trained (t5-base). Not yet evaluated for focus-span relevance. |
| flan-t5-large (both RACE model types) | Retraining after fixing an incorrect learning rate (5e-4 too high for ~770M params, caused early-stop within ~300 steps at a worse loss than flan-t5-base) |

**Active investigation:** whether a genuine per-*question* difficulty signal
can be extracted (attention dispersion, QA-model pass-rate, answer
extractiveness, question-answer similarity — all systematic, no LLM judgment,
no synthetic generation) to replace the passage-inherited label and unblock
Q2/Q4. See `question_difficulty/docs/cognitive_difficulty_estimation.md`'s
"Method 4" for the full plan, and `question_generation/docs/difficulty_steering_mechanisms.md`
for the planned continuous/FiLM-conditioned adapter that would consume it.

Dropped from the pipeline entirely: `MultiRC` (missing evidence annotations
in the available dataset variant), IRT-based QDE (too few QA "student"
models for a statistically reliable fit), synthetic LLM-generated training
questions (quality concerns), LLM-as-annotator (wanted something more
systematic).

---

## Model types

Two fair-comparison pairs (see `question_generation/docs/training_details.md`
for exact dataset sizes and split details):

| Model type | Input | Training data |
|---|---|---|
| `baseline-race` | `passage` | RACE++ (RACE-middle/high/C) |
| `diff-control-race` | `<EASY\|MEDIUM\|HARD> passage` | RACE++, same split as `baseline-race` |
| `baseline-hotpot` | `passage` | HotpotQA (comparison-type only) |
| `focus-control-hotpot` | `passage_with_focus_spans` | HotpotQA, same split as `baseline-hotpot` |

Each pair shares identical train/val/test passages and questions, differing
only in input conditioning — isolates the effect of the conditioning
mechanism rather than dataset distribution shift.

---

## QDE (Question Difficulty Estimator)

See `question_difficulty/docs/cognitive_difficulty_estimation.md` for full
detail: 3 methods trained against the RACE-subset-inherited label (feature-
based, encoder fine-tune, contrastive), all sharing the same passage-confound
this experiment plan's Q2 is blocked on, plus the new "Method 4" per-question
signal extraction effort aimed at resolving it.

---

## Model Configuration

Current training config (see `question_generation/docs/training_details.md`
for the authoritative, up-to-date version — this table is a summary):

| Parameter | Value |
|---|---|
| Backbones | `google-t5/t5-base`, `google/flan-t5-base`, `google/flan-t5-large` |
| Batch size | 16 (t5-base/flan-t5-base), 8 (flan-t5-large) |
| Learning rate | 5e-4 (base models), 1e-4 (large models — 5e-4 was too high, caused early divergence) |
| Max input | 512 tokens (HotpotQA), 1024 tokens (RACE — adaptive per dataset, see training_details.md) |
| Max output | 64 tokens |
| Epochs | 5 (with early stopping, patience 3) |
| Precision | bf16 (not fp16 — fp16 caused NaN/zero-loss collapse on flan-t5 models) |
| Hardware | 1x GH200 (CSC Roihu `gpumedium`) |

---

## Prior Work to Beat

| Paper | Method | Metric |
|---|---|---|
| Du et al. 2017 — Learning to Ask | LSTM on SQuAD | BLEU-4 ~13 |
| Gao et al. 2019 — Difficulty-Controlled QG | Heuristic signals, no standard framework | Difficulty alignment ~60% |
| Pan et al. 2020 — Semantic Graphs for Deep Questions | AMR graph + GNN | BERTScore F1 ~0.62 |

`diff-control-race` needs to beat Gao et al. 2019 on difficulty alignment —
currently blocked on the per-question signal work above before this
comparison is meaningful.

---

## Docs index

- `question_generation/docs/training_details.md` — training config, dataset
  sizes/splits, the token-conditioning confound writeup
- `question_generation/docs/evaluation_plan.md` — evaluation pipeline stages,
  scripts, metrics
- `question_generation/docs/difficulty_steering_mechanisms.md` — the 5
  candidate conditioning mechanisms, current status of each
- `question_generation/docs/related_work_qg.md` — literature review
- `question_difficulty/docs/cognitive_difficulty_estimation.md` — QDE methods,
  the passage-confound limitation, the per-question signal extraction plan
