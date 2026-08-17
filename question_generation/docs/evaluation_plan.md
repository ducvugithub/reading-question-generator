# Evaluation Plan — Difficulty-Controlled Question Generation

## Goal

Validate that conditioning a QG model on difficulty signals (LLM-judged cognitive difficulty + KG structure) produces questions that:
1. Are answerable and relevant to the passage
2. Actually match the requested difficulty level

---

## Dataset Split

- **Source**: EN eval set, ~20k annotated QA pairs (passage-level split)
- **Train**: 80% (~16k QA pairs)
- **Test**: 20% (~4k QA pairs)
- Difficulty labels derived from `question_cognitive_diff`: easy <0.33, medium 0.33–0.67, hard >0.67

---

## Models

| ID | Input | Difficulty | KG |
|----|-------|:----------:|:--:|
| M1 | passage + answer | — | — |
| M2 | passage + answer | ✓ | — |
| M3 | passage + answer | — | ✓ |
| M4 | passage + answer | ✓ | ✓ |

All models start from raw `t5-base` fine-tuned on the same training split.

---

## Evaluation Metrics

### 1. QA-eval — Primary Quality Metric

Generate a question → run a QA model to answer it from the passage → compare predicted answer to original answer.

**Models:**
- `deepset/roberta-base-squad2` (standard)
- `deepset/deberta-v3-base-squad2` (stronger)

**Scores:**
- **Exact Match (EM)**: predicted span == original answer
- **F1**: token overlap between predicted and original answer

A question is good if it is answerable and elicits the correct answer span.

---

### 2. Difficulty Alignment — Main Contribution Metric

For M2 and M4 (difficulty-conditioned models), generate questions at each requested level (EASY, MEDIUM, HARD). Run the LLM judge (Haiku) on generated questions and measure how often the actual difficulty matches the request.

**Metric:**
```
alignment@level = % of generated questions whose scored level matches requested level
```

**Expected result:** M1 (no control) shows random alignment ~33%. M2/M4 show significantly higher alignment.

---

### 3. Student Simulator — Model-Based Difficulty Validation

Use a cascade of QA models of increasing capability to simulate students of different proficiency levels:

| Simulator | Model | Represents |
|-----------|-------|------------|
| Weak | `distilbert-base-cased-distilled-squad` | Struggling student |
| Medium | `deepset/roberta-base-squad2` | Average student |
| Strong | `deepset/deberta-v3-base-squad2` | Advanced student |

**Difficulty score** = proportion of simulators that answer correctly:
- 3/3 correct → easy
- 2/3 correct → medium  
- 0–1/3 correct → hard

**Uses:**
1. **Validate LLM judge**: does Haiku's `question_cognitive_diff` correlate with simulator-based difficulty?
2. **Evaluate generated questions**: do questions requested at HARD actually fail weak simulators?

---

### 4. Secondary Metrics

| Metric | Tool | Purpose |
|--------|------|---------|
| BERTScore (F1) | `bert-score` | Semantic similarity to reference, handles paraphrases |
| BLEU-4 | `sacrebleu` | N-gram overlap, reported for prior work comparability |

---

## Evaluation Procedure

```
For each model M1–M4:
  1. Generate questions on test set
     - M1/M3: generate once (no difficulty conditioning)
     - M2/M4: generate 3× — once per difficulty level (EASY, MEDIUM, HARD)
  
  2. QA-eval
     - Run roberta-base-squad2 + deberta-v3-base-squad2 on generated questions
     - Compute EM and F1 per model
  
  3. Difficulty alignment (M2, M4 only)
     - Run Haiku annotator on generated questions
     - Compute alignment@easy, alignment@medium, alignment@hard
  
  4. Student simulator
     - Run 3 QA simulators on generated questions
     - Compute simulator-based difficulty score
     - Correlate with Haiku scores (Spearman ρ)
  
  5. BERTScore + BLEU-4
     - Compare against reference questions in test set
```

---

## Key Research Questions

| Question | Metric |
|----------|--------|
| Does difficulty conditioning improve alignment? | alignment@level: M1 vs M2/M4 |
| Does KG conditioning improve question quality? | QA-eval F1: M1/M2 vs M3/M4 |
| Do LLM difficulty scores reflect actual answerability difficulty? | Spearman ρ between Haiku scores and simulator scores |
| Does combining both signals (M4) outperform each alone? | All metrics: M2 vs M3 vs M4 |
