# Threshold Sensitivity: `deepset/deberta-v3-base-squad2`

n=250 samples (OneStopQA=50, RACE-C=50, RACE-high=50, RACE-middle=50, SQuAD=50)

Length guard on `recall_overlap`: prediction must be <= 30% of the passage's word count.

## Yield by `token_f1` threshold alone

| f1_thresh | n_captured | pct |
|---|---|---|
| 0.30 | 92 | 36.8% |
| 0.40 | 79 | 31.6% |
| 0.50 | 72 | 28.8% |
| 0.60 | 67 | 26.8% |
| 0.70 | 60 | 24.0% |

## Yield by `recall_overlap` threshold, with vs. without length guard

| ro_thresh | no_guard_pct | with_guard_pct | n_blocked_by_guard |
|---|---|---|---|
| 0.50 | 35.2% | 32.8% | 6 |
| 0.70 | 27.2% | 26.0% | 3 |
| 0.80 | 25.2% | 24.4% | 2 |
| 1.00 | 24.8% | 24.0% | 2 |

## Combined gate: `captured_correct` = (f1 >= f1_thresh) OR (recall_overlap >= ro_thresh AND length-guarded)

| f1_thresh | ro>=0.5 | ro>=0.7 | ro>=0.8 | ro>=1.0 |
|---|---|---|---|---|
| 0.30 | 38.0% | 37.6% | 37.6% | 37.6% |
| 0.40 | 34.8% | 32.4% | 32.4% | 32.4% |
| 0.50 | 33.6% | 30.4% | 30.0% | 29.6% |
| 0.60 | 32.8% | 28.4% | 28.0% | 27.6% |
| 0.70 | 32.8% | 27.2% | 26.4% | 26.0% |

## Yield by source (f1_thresh=0.5, ro_thresh=0.8)

| source | n_captured | n_total | pct |
|---|---|---|---|
| OneStopQA | 9 | 50 | 18.0% |
| RACE-C | 0 | 50 | 0.0% |
| RACE-high | 5 | 50 | 10.0% |
| RACE-middle | 12 | 50 | 24.0% |
| SQuAD | 49 | 50 | 98.0% |
