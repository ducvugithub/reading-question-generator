# Evaluation Plan — Difficulty- and Focus-Span-Controlled QG

## Metrics by Step

### Step 1: QDE Evaluation

| Metric | Details |
|---|---|
| Macro F1 | Equal weight to EASY / MEDIUM / HARD (primary) |
| Confusion matrix | Detect systematic bias (e.g., MEDIUM always predicted as EASY) |
| Feature importance | Feature-based model only — sanity check that `a_in_passage` isn't the only signal |

---

### Steps 0, 2, 3, 4: QG Evaluation

All QG models share the same three metric categories.

#### 1. QA-eval — Answerability

Generate a question → run a QA model → compare predicted answer to original.

| Model | Role |
|---|---|
| `deepset/roberta-base-squad2` | Standard |
| `deepset/deberta-v3-base-squad2` | Stronger |

Scores: **Exact Match (EM)** and **F1** (token overlap).

A question is good if a QA model can answer it correctly from the passage — i.e., it is answerable and grounded.

---

#### 2. Difficulty Alignment — Main Contribution Metric (Steps 2, 4)

Run the best QDE (Step 1) on generated questions. Measure how often the predicted difficulty matches the requested difficulty token.

```
alignment@level = % of generated questions where QDE(q) == requested_level
```

Expected: Step 0 (no control) → ~33% (random). Steps 2, 4 → significantly higher.

Also report via student simulator as a model-free check:

| Simulator | Model | Represents |
|---|---|---|
| Weak | `distilbert-base-cased-distilled-squad` | Struggling reader |
| Medium | `deepset/roberta-base-squad2` | Average reader |
| Strong | `deepset/deberta-v3-base-squad2` | Advanced reader |

Difficulty score = proportion of simulators that answer correctly (3/3 → EASY, 2/3 → MEDIUM, ≤1/3 → HARD).

---

#### 3. Focus Span Relevance (Step 3, 4)

No automated metric fully captures whether a question requires reasoning from a focus span vs. locating an answer.

| Proxy | Method |
|---|---|
| Question type distribution | Fraction of yes/no and multi-hop wh- vs. span-locating questions |
| QA-eval delta | Does removing the focus span from context reduce QA-eval F1? |
| Human eval | Annotators judge: "Can this question be answered by locating a span, or does it require inference?" |

Human eval is the gold standard for Step 3/4.

---

#### 4. Standard NLG Metrics (for prior work comparability)

| Metric | Tool | Purpose |
|---|---|---|
| BERTScore F1 | `bert-score` | Semantic similarity to reference question |
| BLEU-4 | `sacrebleu` | N-gram overlap; reported for comparison with prior work |

Note: BERTScore and BLEU measure similarity to reference questions, not question quality directly. A harder question that is phrased differently from the SQuAD reference may score low even if it is better. Use as secondary metrics only.

---

## Evaluation Procedure

```
For each model (Step 0, 2, 3, 4):
  1. Generate questions on test set
     - Step 0: one pass (no conditioning)
     - Steps 2, 4: three passes — EASY, MEDIUM, HARD
     - Steps 3, 4: one pass per focus span

  2. QA-eval
     - roberta-base-squad2 + deberta-v3-base-squad2
     - Compute EM and F1

  3. Difficulty alignment (Steps 2, 4)
     - Run QDE on generated questions
     - Compute alignment@easy, alignment@medium, alignment@hard
     - Run student simulator; correlate with QDE (Spearman ρ)

  4. Focus span relevance (Steps 3, 4)
     - Count question type distribution (yes/no, wh-, etc.)
     - Optional: human eval on 100-question sample

  5. BERTScore + BLEU-4
     - Compare against reference questions in test split
```

---

## Key Comparisons

| Question | Comparison |
|---|---|
| Does difficulty conditioning work? | Step 0 vs Step 2: alignment@level |
| Does focus span shift question type? | Step 0 vs Step 3: question type distribution, human eval |
| Does combining both help? | Steps 2 + 3 individually vs Step 4 (M6) |
| Does QDE-enrichment hurt data quality? | Step 4 difficulty alignment vs Step 2 (clean labels vs noisy QDE labels) |
