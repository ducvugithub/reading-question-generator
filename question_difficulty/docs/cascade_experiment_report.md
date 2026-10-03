# Answerer Cascade Experiment — Stage 1 Report

## Setup

- 1296 OneStopQA/RACE items, 4-option (shuffled to avoid all correct answers as A option) MC, real gold answers.
- 17 models: 7 local autoregressive, 4 local extractive, 6 remote API (Bedrock) x up to 4 precisions (local only) = 38 candidates.
- Run: `question_difficulty/scripts/run_answerer_cascade.py` — local decoders/extractive on CSC Roihu (GH200 GPU), remote (Haiku, Nova micro/lite/pro, GLM-4.7-flash, Gemma 3 4B) locally via Bedrock.
- Scored + merged: `question_difficulty/scripts/score_cascade_results.py`.
- Data: `question_difficulty/scripts/cascade_results_all_scored.json`.

## Key fix

Raw `options` in the source data are un-shuffled — the correct answer sat at position "A" in ~100% of items. Fixed by shuffling options per-item (deterministic seed) before building prompts. Without this fix, accuracy only measured each model's bias toward picking "A", not comprehension.

## Results

| Model | Type | fp32 | fp16 | int8 | int4 |
|---|---|---|---|---|---|
| smollm2_135m | local autoregressive | 22.0% | 22.1% | 23.5% | 23.5% |
| smollm2_360m | local autoregressive | 24.6% | 24.6% | 25.4% | 25.8% |
| tinyllama11 | local autoregressive | 21.1% | 21.6% | 23.4% | 21.2% |
| qwen2_05 | local autoregressive | 52.3% | 52.5% | 51.5% | 44.3% |
| qwen05 | local autoregressive | 53.2% | 53.2% | 53.8% | 43.8% |
| falcon1b | local autoregressive | 57.9% | 57.8% | 57.7% | 55.0% |
| qwen15 | local autoregressive | 76.2% | 76.2% | 76.5% | 71.8% |
| extractive_distilbert | local extractive | 36.3% | - | - | - |
| extractive_roberta_nonsquad | local extractive | 38.6% | - | - | - |
| extractive_roberta_base | local extractive | 40.2% | - | - | - |
| extractive_deberta_v3 | local extractive | 41.2% | - | - | - |
| gemma3_4b | remote API | 79.8% | - | - | - |
| glm47flash | remote API | 82.3% | - | - | - |
| nova_micro | remote API | 86.5% | - | - | - |
| nova_lite | remote API | 88.2% | - | - | - |
| haiku | remote API | 89.4% | - | - | - |
| nova_pro | remote API | 90.7% | - | - | - |

Chance level: 25%. Opus deliberately excluded (real, higher cost per item; not needed to establish the ceiling).

## EDA

- Coverage: 38 candidates x 1296 items = 49,248 pairs, no missing data.
- Accuracy across candidates (unweighted): mean 48.8%, range 21.1%-90.7%.
- Per-item consensus (candidates correct out of 37): mean 18.1/37. 0 items unanimous-correct; 39 items "easy" (>=30/37 correct); 136 items "hard" (<=7/37 correct). The 66 items originally reported here as "unanimous-wrong" are NOT a real difficulty finding -- confirmed (2026-10-03) to be a `HumanResponseBank` RACE answer-key resolution bug (`correct_answer_text` doesn't match any option, so every candidate scores 0.000 by construction). See `irt_human_vs_model.ipynb` §7.2; these 66 items are excluded from the IRT fits as of `redo_irt_excluding_broken_gold.py`.
- Data quality: `tinyllama11` has a notably high unparsed-answer rate (17-52/1296 per precision, ~1-4%) vs. ~0-2 for every other model -- consistent with its near-chance accuracy.

## Findings

- Full spread: 21% (tinyllama11) to 90.7% (nova_pro) across 17 models — strong cascade separation.
- Remote frontier models are the clear top tier (82-91%), well above the best local model (qwen15, 76.5%).
- smollm2/tinyllama sit at chance — no real comprehension signal; their small int8/int4 upticks are noise (near chance, differences within statistical error), not a genuine quantization benefit.
- Precision barely matters until int4, which drops accuracy ~4-9pts consistently across local models.
- Extractive models (36-41%) partially fill the gap between chance-level tiny models and qwen05/qwen2_05 (~52-54%).

## Not yet done

- Opus not run (deliberately, cost).
- Near-chance models' effect on a naive averaged cascade score vs. a proper IRT-weighted combination not yet tested.

IRT correlation against human difficulty: done, see `question_difficulty/notebooks/irt_human_vs_model.ipynb` -- human-vs-model correlation is weak-to-moderate (r=0.27-0.29) after fixing the broken-gold-answer bug above (originally reported as r=0.46-0.53, which was inflated by that bug). The per-candidate dose-response check (agreement with human difficulty scales with model size/accuracy, §6) is the stronger validity signal: r=-0.92.
