# Threshold Sensitivity: `deepset/roberta-base-squad2`

n=250 samples (OneStopQA=50, RACE-C=50, RACE-high=50, RACE-middle=50, SQuAD=50)

Length guard on `recall_overlap`: prediction must be <= 30% of the passage's word count.

## Yield by `token_f1` threshold alone

| f1_thresh | n_captured | pct |
|---|---|---|
| 0.30 | 84 | 33.6% |
| 0.40 | 69 | 27.6% |
| 0.50 | 63 | 25.2% |
| 0.60 | 59 | 23.6% |
| 0.70 | 49 | 19.6% |

## Yield by `recall_overlap` threshold, with vs. without length guard

| ro_thresh | no_guard_pct | with_guard_pct | n_blocked_by_guard |
|---|---|---|---|
| 0.50 | 33.2% | 30.0% | 8 |
| 0.70 | 24.0% | 22.4% | 4 |
| 0.80 | 22.4% | 20.8% | 4 |
| 1.00 | 22.0% | 20.8% | 3 |

## Combined gate: `captured_correct` = (f1 >= f1_thresh) OR (recall_overlap >= ro_thresh AND length-guarded)

| f1_thresh | ro>=0.5 | ro>=0.7 | ro>=0.8 | ro>=1.0 |
|---|---|---|---|---|
| 0.30 | 35.6% | 34.8% | 34.8% | 34.8% |
| 0.40 | 31.6% | 28.8% | 28.8% | 28.8% |
| 0.50 | 30.4% | 27.2% | 26.8% | 26.8% |
| 0.60 | 30.4% | 26.0% | 25.6% | 25.6% |
| 0.70 | 30.0% | 23.6% | 22.8% | 22.8% |

## Yield by source (f1_thresh=0.5, ro_thresh=0.8)

| source | n_captured | n_total | pct |
|---|---|---|---|
| OneStopQA | 9 | 50 | 18.0% |
| RACE-C | 0 | 50 | 0.0% |
| RACE-high | 3 | 50 | 6.0% |
| RACE-middle | 10 | 50 | 20.0% |
| SQuAD | 45 | 50 | 90.0% |
