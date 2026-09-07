# Threshold Sensitivity: `consciousAI/question-answering-roberta-base-s-v2`

n=250 samples (OneStopQA=50, RACE-C=50, RACE-high=50, RACE-middle=50, SQuAD=50)

Length guard on `recall_overlap`: prediction must be <= 30% of the passage's word count.

## Yield by `token_f1` threshold alone

| f1_thresh | n_captured | pct |
|---|---|---|
| 0.30 | 93 | 37.2% |
| 0.40 | 80 | 32.0% |
| 0.50 | 77 | 30.8% |
| 0.60 | 69 | 27.6% |
| 0.70 | 62 | 24.8% |

## Yield by `recall_overlap` threshold, with vs. without length guard

| ro_thresh | no_guard_pct | with_guard_pct | n_blocked_by_guard |
|---|---|---|---|
| 0.50 | 36.4% | 35.2% | 3 |
| 0.70 | 28.8% | 27.6% | 3 |
| 0.80 | 26.0% | 25.2% | 2 |
| 1.00 | 25.6% | 25.2% | 1 |

## Combined gate: `captured_correct` = (f1 >= f1_thresh) OR (recall_overlap >= ro_thresh AND length-guarded)

| f1_thresh | ro>=0.5 | ro>=0.7 | ro>=0.8 | ro>=1.0 |
|---|---|---|---|---|
| 0.30 | 38.4% | 37.6% | 37.6% | 37.6% |
| 0.40 | 35.6% | 33.2% | 32.8% | 32.8% |
| 0.50 | 35.2% | 32.4% | 31.6% | 31.6% |
| 0.60 | 35.2% | 29.6% | 28.4% | 28.4% |
| 0.70 | 35.2% | 28.4% | 26.8% | 26.8% |

## Yield by source (f1_thresh=0.5, ro_thresh=0.8)

| source | n_captured | n_total | pct |
|---|---|---|---|
| OneStopQA | 9 | 50 | 18.0% |
| RACE-C | 1 | 50 | 2.0% |
| RACE-high | 4 | 50 | 8.0% |
| RACE-middle | 15 | 50 | 30.0% |
| SQuAD | 50 | 50 | 100.0% |
