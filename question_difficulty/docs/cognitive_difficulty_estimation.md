# Question Difficulty Estimator (QDE)

Classifies reading comprehension questions into `EASY / MEDIUM / HARD` using curriculum-grounded labels derived from dataset benchmark performance (SQuAD → EASY, RACE-middle → MEDIUM, RACE-high → HARD).

The QDE is used to:
1. Evaluate QG models: does the generated question match the requested difficulty?
2. Enrich HotpotQA/MultiRC with difficulty labels for M6 (Step 4) training data.

---

## Training Data

All three classes come from the RACE family — same distribution, only exam level differs. This avoids cross-dataset alignment problems and forces the model to learn genuine difficulty signals.

| Label | Source | HF path | Train size |
|---|---|---|---|
| EASY | RACE-middle (Chinese middle-school English exams) | `ehovy/race` middle | ~25K |
| MEDIUM | RACE-high (Chinese high-school English exams) | `ehovy/race` high | ~62K |
| HARD | RACE-C (Chinese college entrance / Gaokao) | `tasksource/race-c` | ~12.7K |

Script: `question_difficulty/scripts/prepare_qde_data.py`

Use `--balanced` to cap at ~12.7K per class (RACE-C is the bottleneck).

---

## Methods

Three methods trained in parallel on the same data. Best one (by macro F1 on val) is used in the enrichment pipeline.

### Method 1: Feature-Based (GradientBoostingClassifier)

14 linguistic features extracted from (passage, question, answer):

| Feature | Description |
|---|---|
| `q_wh_type` | Question word encoded as ordinal (when < where < who < what < which < how < why) |
| `q_n_tokens` | Question length |
| `q_avg_wlen` | Average word length in question |
| `q_avg_zipf` | Average Zipf frequency (wordfreq) of question words |
| `q_frac_rare` | Fraction of question words below Zipf 3.0 |
| `p_n_tokens` | Passage length |
| `p_n_sents` | Number of sentences in passage |
| `p_avg_slen` | Average sentence length |
| `p_ttr` | Type-token ratio of passage |
| `p_avg_zipf` | Average Zipf frequency of passage words |
| `a_n_tokens` | Answer length |
| `a_avg_zipf` | Average Zipf frequency of answer |
| `q_p_overlap` | Fraction of question tokens found in passage |
| `a_in_passage` | 1 if answer text is a substring of the passage, else 0 |

Since all classes now come from RACE, `a_in_passage` is always 0 — the model cannot exploit dataset identity. Top features should shift to genuine difficulty signals: question word type, passage complexity, answer word frequency. Macro F1 is the primary metric.

Script: `question_difficulty/scripts/train_feature_based.py`
Output: `question_difficulty/models/feature_based/gbt_model.pkl`

---

### Method 2: Encoder-Based (RoBERTa / DeBERTa-v3)

Fine-tune a pre-trained encoder; [CLS] representation → linear classifier (3 classes).

**Input format:**
```
[CLS] question [answer: A] [SEP] passage [SEP]
```

**Backbone options:**
- `roberta-base` (default, 125M params, strong SQuAD baseline)
- `microsoft/deberta-v3-base` (stronger, better calibration, slower)

**Training:** Cross-entropy loss, best checkpoint by val accuracy, 5 epochs.

Script: `question_difficulty/scripts/train_encoder.py`
Slurm: `question_difficulty/slurms/train_qde_encoder.job` (`--array=0-1` for both backbones)

---

### Method 3: Contrastive (Triplet Loss + Logistic Regression Probe)

Learn a difficulty-ordered embedding space; difficulty prediction via a linear probe.

**Architecture:**
```
encoder (frozen backbone) → dropout → 2-layer projection head → L2-normalize → embedding
```

**Training phase 1:** Triplet margin loss with online mining.
- Distance: `1 − cosine_similarity`
- Margin: 0.5
- Triplet ordering: EASY as anchor, MEDIUM as positive (closer), HARD as negative (farther), or vice versa

**Training phase 2:** Freeze encoder, train logistic regression probe on embeddings.

**Why contrastive?** Learns a metric space where difficulty is ordered — useful for applications like "find questions similar in difficulty" or ranking, beyond 3-class classification.

Script: `question_difficulty/scripts/train_contrastive.py`
Slurm: `question_difficulty/slurms/train_qde_contrastive.job`

---

## Evaluation

- **Primary metric:** Macro F1 on balanced test set (equal weight to all three classes)
- Accuracy is misleading due to class imbalance (EASY >> HARD >> MEDIUM in unbalanced data)
- Report confusion matrix to detect systematic misclassification (e.g., MEDIUM confused with EASY)

---

## Legacy: Rule-Based Estimator

The old rule-based estimator (`question_generation/difficulty/cognitive.py`, `GraphCognitiveDifficultyEstimator`) used four heuristic signals:

```
score = 0.45 × s_qtype + 0.30 × s_coref + 0.15 × s_coverage + 0.10 × s_density
```

This is KG-dependent and not used in the new roadmap. It remains in the codebase as a reference and for legacy pipeline compatibility (M2/M3/M4 with haiku_cog labels).

The new QDE (`question_difficulty/`) is dataset-trained and does not require KG extraction.
