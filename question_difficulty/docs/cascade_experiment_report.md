# Answerer Cascade Experiment — Stage 1 Report

## Setup

- 1296 OneStopQA/RACE items, 4-option MC, real gold answers.
- 11 models (7 autoregressive, 4 extractive) x up to 4 precisions (fp32/fp16/int8/int4) = 32 candidates.
- Run: `question_difficulty/scripts/run_answerer_cascade.py` on CSC Roihu (GH200 GPU).
- Scored: `question_difficulty/scripts/score_cascade_results.py`.
- Data: `question_difficulty/scripts/cascade_results_roihu_full_scored.json`.

## Key fix

Raw `options` in the source data are un-shuffled — the correct answer sat at position "A" in ~100% of items. Fixed by shuffling options per-item (deterministic seed) before building prompts. Without this fix, accuracy only measured each model's bias toward picking "A", not comprehension.

## Results

| Model | Type | fp32 | fp16 | int8 | int4 |
|---|---|---|---|---|---|
| smollm2_135m | autoregressive | 22.0% | 22.1% | 23.5% | 23.5% |
| smollm2_360m | autoregressive | 24.6% | 24.6% | 25.4% | 25.8% |
| tinyllama11 | autoregressive | 21.1% | 21.6% | 23.4% | 21.2% |
| qwen2_05 | autoregressive | 52.3% | 52.5% | 51.5% | 44.3% |
| qwen05 | autoregressive | 53.2% | 53.2% | 53.8% | 43.8% |
| falcon1b | autoregressive | 57.9% | 57.8% | 57.7% | 55.0% |
| qwen15 | autoregressive | 76.2% | 76.2% | 76.5% | 71.8% |
| extractive_distilbert | extractive | 36.3% | - | - | - |
| extractive_roberta_nonsquad | extractive | 38.6% | - | - | - |
| extractive_roberta_base | extractive | 40.2% | - | - | - |
| extractive_deberta_v3 | extractive | 41.2% | - | - | - |

Chance level: 25%.

## Findings

- Clear spread: 21% (tinyllama11) to 76.5% (qwen15) — good cascade separation.
- smollm2/tinyllama sit at chance — no real comprehension signal; their small int8/int4 upticks are noise (near chance, differences within statistical error), not a genuine quantization benefit.
- Precision barely matters until int4, which drops accuracy ~4-9pts consistently across models.
- Extractive models (36-41%) partially fill the gap between chance-level tiny models and qwen05/qwen2_05 (~52-54%).
- qwen15 is the clear ceiling, stable across precision.

## Not yet done

- Haiku (and Opus) not run yet — Bedrock skipped on Roihu (no compute-node internet). Next: run Haiku locally, merge into this result set.
- No correlation against IRT difficulty yet (does accuracy track question difficulty).
- Near-chance models' effect on a naive averaged cascade score vs. a proper IRT-weighted combination not yet tested.
