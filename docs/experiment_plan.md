# Experiment Plan: Difficulty- and Focus-Span-Controlled Question Generation

## Research Questions

1. Can a learned QDE reliably classify question difficulty across EASY / MEDIUM / HARD?
2. Does curriculum-label conditioning (RACE++) produce questions that are measurably harder than SQuAD-style baseline?
3. Does conditioning on a focus span (HotpotQA/MultiRC evidence sentences) shift questions toward inference rather than span-locating?
4. Does combining both controls (M6) produce the best difficulty alignment and inference quality?

---

## Roadmap

| Step | Model | Input | Training data | Status |
|---|---|---|---|---|
| 0 | Baseline QG | `passage` | RACE++ + HotpotQA + MultiRC (no conditioning) | pending |
| 1 | QDE | `passage + question + answer` | RACE-middle→EASY, RACE-high→MEDIUM, RACE-college→HARD | code done |
| 2 | Difficulty QG | `passage + <difficulty>` | RACE++ (curriculum labels) | data ready |
| 3 | Focus-span QG | `passage + focus_span` | HotpotQA (comparison/yes-no) + MultiRC | pending |
| 4 | M6 (full) | `passage + focus_span + <difficulty>` | Steps 2+3 enriched via QDE | depends on 1+3 |

**Note on answer:** Answer span is NOT an input to the generator (unlike standard SQuAD QG). RACE questions have multiple-choice answers; RACE++ training keeps the answer text available as context but the generator produces the question, not the answer.

---

## Step 0 — Baseline QG (M1)

Fine-tune T5-base on RACE++ + HotpotQA + MultiRC: `passage → question` (no difficulty token, no focus span).

Serves as the uncontrolled ablation baseline. Using the same source datasets as Steps 2–4 means comparisons isolate the effect of difficulty/focus-span conditioning rather than dataset distribution shift (which would happen if baseline were SQuAD-trained).

---

## Step 1 — Question Difficulty Estimator (QDE)

Three methods trained in parallel. Best one used to enrich data for Step 4.

### Training data

All three classes come from the RACE family — same distribution, only exam level differs. Avoids cross-dataset alignment problems; all answers are multiple-choice (non-span), so no `a_in_passage` artifact.

| Label | Source | HF path | Train size |
|---|---|---|---|
| EASY | RACE-middle (Chinese middle-school English exams) | `ehovy/race` middle | ~25K |
| MEDIUM | RACE-high (Chinese high-school English exams) | `ehovy/race` high | ~62K |
| HARD | RACE-C (Chinese college entrance / Gaokao) | `tasksource/race-c` | ~12.7K |

Use `--balanced` flag to cap at ~12.7K per class (RACE-C is the bottleneck).

### Methods

| Method | Architecture | Key detail |
|---|---|---|
| Feature-based | GradientBoostingClassifier | 14 linguistic features (q_wh_type, q_avg_zipf, a_in_passage, p_n_sents, …) |
| Encoder | RoBERTa / DeBERTa-v3 fine-tune | Input: `[CLS] question [answer: A] [SEP] passage [SEP]`, [CLS] → linear classifier |
| Contrastive | Triplet loss + projection head | Online triplet mining, L2-normalized embeddings, LR probe phase 2 |

Feature-based model: `a_in_passage` is now less dominant since all three classes come from RACE (all non-span answers). The model must learn genuine difficulty signals. Macro F1 is the primary metric.

Scripts: `question_difficulty/scripts/train_{feature_based,encoder,contrastive}.py`
Slurm jobs: `question_difficulty/slurms/train_qde_{feature_based,encoder,contrastive}.job`

### Evaluation

- Macro F1 on balanced test set (primary)
- Confusion matrix (detect systematic misclassification)
- Feature importance (feature-based only)

---

## Step 2 — Difficulty-Controlled QG (M5)

Fine-tune T5-base on RACE++ with curriculum difficulty token prepended.

```
difficulty: hard context: Young people nowadays...
→ "What is the author's attitude toward social media?"
```

Difficulty labels: `middle → MEDIUM`, `high → HARD`, `college (Gaokao) → HARD`.
SQuAD records (from M1 data) are added with forced `EASY` label.

Input format is identical to M2 (haiku\_cog-based), so M2 and M5 are comparable ablations.

### Evaluation

- QA-eval: run RoBERTa-SQuAD2 on generated questions, measure EM + F1
- **Difficulty alignment**: run QDE (Step 1) on generated questions, compute `alignment@level = % requests honored`
- BLEU-4, BERTScore-F1 vs reference questions

---

## Step 3 — Focus-Span QG

Fine-tune T5-base on evidence-annotated datasets.

```
focus: [Scott Derrickson is an American director] [Ed Wood was an American filmmaker]
context: [full 10-passage context]
→ "Were Scott Derrickson and Ed Wood of the same nationality?"
```

### Training data

| Dataset | Evidence granularity | Q types | Size |
|---|---|---|---|
| HotpotQA (`type == "comparison"`) | Sentence-level supporting\_facts | Yes/No, multi-hop inference | ~18K (comparison subset) |
| MultiRC | Sentence-level evidence | Binary T/F, inference | ~9.7K |

Filter HotpotQA to `type == "comparison"` to exclude bridge questions where the answer IS a span.

### Evaluation

- QA-eval (answerability)
- Human eval: does the question require reasoning from the focus span, not just locating an answer?
- Proxy: what fraction of generated questions are yes/no or require multi-hop (vs. span-locating wh-)?

---

## Step 4 — M6: Focus Span + Difficulty

Combine Steps 2 and 3 via QDE-based data enrichment.

### Bootstrapping pipeline

```
HotpotQA / MultiRC
  → run QDE (Step 1) on each (passage, question, answer)
  → assign EASY / MEDIUM / HARD label
  → now have (passage, focus_span, difficulty, question) tuples

Combined training data:
  RACE++ records (difficulty from curriculum, no focus span → span = "")
  + enriched HotpotQA/MultiRC (focus span + QDE difficulty)

Input: passage + focus_span + <difficulty> → question
```

Caveat: QDE is trained on SQuAD/RACE distribution; applying it to HotpotQA/MultiRC is extrapolation — difficulty signal will be noisier. Report this in the paper.

### Evaluation

- Difficulty alignment (QDE on outputs)
- Focus span relevance (human eval or proxy)
- QA-eval (answerability)

---

## Model Configuration

| Parameter | Value |
|---|---|
| Backbone | `google-t5/t5-base` (220M params) |
| Batch size | 16 (gradient accumulation ×4 = effective 64) |
| Learning rate | 5e-4, linear warmup |
| Max input | 512 tokens |
| Max output | 64 tokens |
| Epochs | 5 |
| Hardware | 1× A100 (Mahti `gpusmall`) |

---

## Prior Work to Beat

| Paper | Method | Metric |
|---|---|---|
| Du et al. 2017 — Learning to Ask | LSTM on SQuAD | BLEU-4 ~13 |
| Gao et al. 2019 — Difficulty-Controlled QG | Heuristic signals, no standard framework | Difficulty alignment ~60% |
| Pan et al. 2020 — Semantic Graphs for Deep Questions | AMR graph + GNN | BERTScore F1 ~0.62 |

Step 2 (M5) should beat Gao et al. 2019 on difficulty alignment. M6 is novel — no prior work combines learned difficulty + focus-span control without an answer span.
