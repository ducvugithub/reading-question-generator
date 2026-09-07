# Threshold Sensitivity: Model Comparison Summary

Combined gate at f1_thresh=0.5, ro_thresh=0.8, length-guard=30% -- same 250 samples scored by every model.

## Overall yield

| model | n_captured | n_total | pct |
|---|---|---|---|
| `deepset/roberta-base-squad2` | 67 | 250 | 26.8% |
| `deepset/deberta-v3-base-squad2` | 75 | 250 | 30.0% |
| `consciousAI/question-answering-roberta-base-s-v2` | 79 | 250 | 31.6% |

## Yield by source

| source | `deepset/roberta-base-squad2` | `deepset/deberta-v3-base-squad2` | `consciousAI/question-answering-roberta-base-s-v2` |
|---|---|---|---|
| OneStopQA | 18.0% | 18.0% | 18.0% |
| RACE-C | 0.0% | 0.0% | 2.0% |
| RACE-high | 6.0% | 10.0% | 8.0% |
| RACE-middle | 20.0% | 24.0% | 30.0% |
| SQuAD | 90.0% | 98.0% | 100.0% |

## Yield by RACE difficulty level (EASY/MEDIUM/HARD)

Same EASY=RACE-middle, MEDIUM=RACE-high, HARD=RACE-C mapping `validate_difficulty_signals.py` uses. Cells show pct (n_captured/n_total). OneStopQA has its own separate elementary/intermediate/advanced scale, not tracked per-level in this script's sampling -- would need `sample_onestopqa` to sample+tag each level separately to break down the same way. SQuAD has no difficulty tiers.

| level | `deepset/roberta-base-squad2` | `deepset/deberta-v3-base-squad2` | `consciousAI/question-answering-roberta-base-s-v2` |
|---|---|---|---|
| EASY | 20.0% (10/50) | 24.0% (12/50) | 30.0% (15/50) |
| MEDIUM | 6.0% (3/50) | 10.0% (5/50) | 8.0% (4/50) |
| HARD | 0.0% (0/50) | 0.0% (0/50) | 2.0% (1/50) |
