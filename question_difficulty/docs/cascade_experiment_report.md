# Answerer Cascade Experiment — Stage 1 Report

## Setup

- 1296 OneStopQA/RACE items, 4-option MC, real gold answers.
- 16 models: 7 local autoregressive, 4 local extractive, 5 remote API (Bedrock) x up to 4 precisions (local only) = 37 candidates.
- Run: `question_difficulty/scripts/run_answerer_cascade.py` — local decoders/extractive on CSC Roihu (GH200 GPU), remote (Haiku, Nova micro/lite/pro, GLM-4.7-flash) locally via Bedrock.
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
| glm47flash | remote API | 82.3% | - | - | - |
| nova_micro | remote API | 86.5% | - | - | - |
| nova_lite | remote API | 88.2% | - | - | - |
| haiku | remote API | 89.4% | - | - | - |
| nova_pro | remote API | 90.7% | - | - | - |

Chance level: 25%. Opus deliberately excluded (real, higher cost per item; not needed to establish the ceiling).

## Findings

- Full spread: 21% (tinyllama11) to 90.7% (nova_pro) across 16 models — strong cascade separation.
- Remote frontier models are the clear top tier (82-91%), well above the best local model (qwen15, 76.5%).
- smollm2/tinyllama sit at chance — no real comprehension signal; their small int8/int4 upticks are noise (near chance, differences within statistical error), not a genuine quantization benefit.
- Precision barely matters until int4, which drops accuracy ~4-9pts consistently across local models.
- Extractive models (36-41%) partially fill the gap between chance-level tiny models and qwen05/qwen2_05 (~52-54%).

## Not yet done

- Opus not run (deliberately, cost).
- No correlation against IRT difficulty yet (does accuracy track question difficulty).
- Near-chance models' effect on a naive averaged cascade score vs. a proper IRT-weighted combination not yet tested.
