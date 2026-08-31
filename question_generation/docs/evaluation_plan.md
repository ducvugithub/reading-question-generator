# Evaluation Plan — Difficulty- and Focus-Span-Controlled QG

Merged from the former top-level `QG_EVALUATION_FLOW.md` (pipeline/scripts)
and this file's original metrics/procedure content. Model names updated to
the current pipeline (`baseline-race`, `diff-control-race`, `baseline-hotpot`,
`focus-control-hotpot` — see `question_generation/docs/training_details.md`);
the old "Step 0-4"/"M6" numbering and `MultiRC` references are dropped since
that dataset/step structure is no longer part of the pipeline.

**Status note:** `diff-control-race`'s token conditioning has been tested via
the forced-token generation check below (Stage 1 + manual inspection) and
found not to produce measurable steering — see
`question_generation/docs/difficulty_steering_mechanisms.md`'s update section
and `training_details.md`'s "Known limitation". The QDE-based "Difficulty
Alignment" metric below is still the right metric once a working steering
mechanism exists; it just doesn't have a positive result to report yet for
token-conditioning.

---

## Pipeline

```
Stage 1: Generate Questions
    ↓
Stage 2a: QA Model Validation (SQuAD-based)
    ↓
Stage 2b: LLM Quality Assessment (open-source 7B models)
    ↓
Stage 3: Evaluate & Combine Signals (TBD)
```

### Stage 1: Generate Questions

**Script:** `question_generation/scripts/generate_qg_questions.py`

**Input:** Test set from `data/qg/baseline-race/test.jsonl`

**Output:** `question_generation/results/qg_generated.jsonl`

**What it does:**
- Loads trained QG models (`baseline-race`, `diff-control-race`)
- For each test passage, generates questions at all difficulty levels (EASY, MEDIUM, HARD) —
  forcing all 3 tokens on every passage regardless of its real label, so `baseline-race`
  serves as a noise-floor control (it never sees the token, so any variation across its
  3 "difficulty" outputs is pure sampling noise)
- Configurable: `--num-per-difficulty N` (default 1 per level), `--do-sample`/`--num-samples`/
  `--temperature`/`--top-p` for sampling-based generation instead of greedy beam search
  (see `difficulty_steering_mechanisms.md` — beam search can mask distributional shifts
  that sampling reveals)
- `--base-model` selects which trained checkpoint family to load (`t5-base`, `flan-t5-base`,
  `flan-t5-large`)

**Format:**
```json
{
  "passage": "...",
  "original_question": "...",
  "true_difficulty": "EASY",
  "model": "baseline-race" | "diff-control-race",
  "target_difficulty": "EASY" | "MEDIUM" | "HARD",
  "generated_question": "...",
  "beam_rank": 0
}
```

**Local usage (quick test):**
```bash
python question_generation/scripts/generate_qg_questions.py \
  --models baseline-race diff-control-race \
  --base-model flan-t5-base \
  --num-per-difficulty 1 \
  --limit 100
```

**On Roihu (full test set):**
```bash
sbatch question_generation/slurms/generate_qg_questions.job
```

---

### Stage 2a: Validate Answerability (QA Models)

**Script:** `question_answering/scripts/run_qa_models.py`

**Input:** `question_generation/results/qg_generated.jsonl` (from Stage 1)

**Output:** `question_answering/results/qa_scores.jsonl`

**What it does:**
- Loads the 4-model QA battery — see `question_answering/docs/qa_model_battery.md`
  for which models and the known `deberta-v3-base` (not actually SQuAD-finetuned)
  issue
- For each generated question, runs inference to extract answer spans + confidence
- Records all model outputs: scores, extracted answers, agreement
- Leaves metric decisions to Stage 3 (evaluation can use score-based or agreement-based validation)

**Format:**
```json
{
  "passage": "...",
  "generated_question": "...",
  "target_difficulty": "EASY",

  "qa_model_results": {
    "deepset/roberta-base-squad2": {
      "score": 0.95,
      "answer": "Paris",
      "start": 10,
      "end": 15
    }
  },

  "qa_scores": {"model1": 0.95, "model2": 0.92},
  "qa_answer_spans": {"model1": "Paris", "model2": "Paris"},
  "qa_agreement": true,
  "qa_consensus_answer": "Paris",
  "qa_avg_score": 0.93,
  "qa_num_models": 4
}
```

**Local usage:**
```bash
python question_answering/scripts/run_qa_models.py
```

**On Roihu (full):**
```bash
sbatch question_answering/slurms/run_qa_models.job
```

---

### Stage 2b: LLM Quality Assessment

**Script:** `question_answering/scripts/assess_llm_quality.py`

**Models:** Qwen 2-7B, Llama 2-7B-chat, Mistral-7B

**Input:** `question_generation/results/qg_generated.jsonl` (from Stage 1)

**Output:** `question_answering/results/llm_quality_assessments.jsonl`

**What it does:**
- Loads 3 open-source 7B LLMs (fit on single V100 GPU)
- Assesses each question on: Grammaticality (1-5), Answerability (1-5),
  Clarity (1-5), Relevance (1-5)
- Computes overall quality and consensus across models

**Format:**
```json
{
  "passage": "...",
  "generated_question": "...",
  "target_difficulty": "EASY",

  "llm_assessments": {
    "Qwen/Qwen2-7B-Instruct": {
      "grammaticality": 4,
      "answerability": 5,
      "clarity": 4,
      "relevance": 4,
      "overall_quality": 4.25,
      "reasoning": "..."
    }
  },

  "llm_consensus": {
    "avg_overall_quality": 4.15,
    "quality_std": 0.08,
    "num_models_succeeded": 3
  }
}
```

**Local usage:**
```bash
python question_answering/scripts/assess_llm_quality.py --limit 10
```

**On Roihu (full):**
```bash
sbatch question_answering/slurms/assess_llm_quality.job
```

---

### Stage 3: Evaluate & Combine Signals (TBD — not yet written)

**Script:** `question_generation/scripts/evaluate_qg.py` (to be written)

**Input:**
- `question_answering/results/qa_scores.jsonl` (Stage 2a)
- `question_answering/results/llm_quality_assessments.jsonl` (Stage 2b)
- QDE predictions on generated questions (once a working QDE + working
  steering mechanism both exist — see `cognitive_difficulty_estimation.md`)

**Output:** Evaluation report combining the metrics below.

---

## Metrics

### QDE Evaluation (prerequisite for Difficulty Alignment below)

| Metric | Details |
|---|---|
| Macro F1 | Equal weight to EASY / MEDIUM / HARD (primary) |
| Confusion matrix | Detect systematic bias (e.g., MEDIUM always predicted as EASY) |
| Feature importance | Feature-based model only — sanity check that `a_in_passage` isn't the only signal |

### 1. QA-eval — Answerability

Generate a question → run a QA model → compare predicted answer to original.

| Model | Role |
|---|---|
| `deepset/roberta-base-squad2` | Standard |
| `deepset/deberta-v3-base-squad2` | Stronger (note: different checkpoint than the non-finetuned one used in Stage 2a — needs reconciling) |

Scores: **Exact Match (EM)** and **F1** (token overlap).

A question is good if a QA model can answer it correctly from the passage — i.e., it is answerable and grounded.

### 2. Difficulty Alignment — Main Contribution Metric

Run the best QDE on generated questions. Measure how often the predicted difficulty matches the requested difficulty token.

```
alignment@level = % of generated questions where QDE(q) == requested_level
```

Expected: uncontrolled baseline (`baseline-race`) → ~33% (random). A working
steering mechanism should show significantly higher. **Current status:
`diff-control-race`'s token conditioning does not clear this bar — see status
note at top of this doc.**

Also report via student simulator as a model-free check:

| Simulator | Model | Represents |
|---|---|---|
| Weak | `distilbert-base-cased-distilled-squad` | Struggling reader |
| Medium | `deepset/roberta-base-squad2` | Average reader |
| Strong | `deepset/deberta-v3-base-squad2` | Advanced reader |

Difficulty score = proportion of simulators that answer correctly (3/3 → EASY, 2/3 → MEDIUM, ≤1/3 → HARD).

### 3. Focus Span Relevance (`baseline-hotpot` / `focus-control-hotpot`)

No automated metric fully captures whether a question requires reasoning from a focus span vs. locating an answer.

| Proxy | Method |
|---|---|
| Question type distribution | Fraction of yes/no and multi-hop wh- vs. span-locating questions |
| QA-eval delta | Does removing the focus span from context reduce QA-eval F1? |
| Human eval | Annotators judge: "Can this question be answered by locating a span, or does it require inference?" |

Human eval is the gold standard here.

### 4. Standard NLG Metrics (for prior work comparability)

| Metric | Tool | Purpose |
|---|---|---|
| BERTScore F1 | `bert-score` | Semantic similarity to reference question |
| BLEU-4 | `sacrebleu` | N-gram overlap; reported for comparison with prior work |

Note: BERTScore and BLEU measure similarity to reference questions, not question quality directly. A harder question that is phrased differently from the reference may score low even if it is better. Use as secondary metrics only.

---

## Evaluation Procedure

```
For each model pair (baseline-race/diff-control-race, baseline-hotpot/focus-control-hotpot):
  1. Generate questions on test set
     - baseline-*: one pass (no conditioning)
     - diff-control-race: three passes — EASY, MEDIUM, HARD forced per passage
     - focus-control-hotpot: one pass per focus span

  2. QA-eval
     - roberta-base-squad2 + deberta-v3-base-squad2
     - Compute EM and F1

  3. Difficulty alignment (diff-control-race only)
     - Run QDE on generated questions
     - Compute alignment@easy, alignment@medium, alignment@hard
     - Run student simulator; correlate with QDE (Spearman ρ)

  4. Focus span relevance (focus-control-hotpot only)
     - Count question type distribution (yes/no, wh-, etc.)
     - Optional: human eval on 100-question sample

  5. BERTScore + BLEU-4
     - Compare against reference questions in test split
```

---

## Key Comparisons

| Question | Comparison |
|---|---|
| Does difficulty conditioning work? | `baseline-race` vs `diff-control-race`: alignment@level (currently: no) |
| Does focus span shift question type? | `baseline-hotpot` vs `focus-control-hotpot`: question type distribution, human eval |
| Does a real per-question difficulty signal exist? | See `cognitive_difficulty_estimation.md`'s Method 4 — prerequisite for a working difficulty-alignment result |

---

## Quick Local Test

```bash
python question_generation/scripts/generate_qg_questions.py \
  --models baseline-race \
  --base-model flan-t5-base \
  --num-per-difficulty 1 \
  --limit 10

python question_answering/scripts/run_qa_models.py

head question_answering/results/qa_scores.jsonl | jq '.qa_scores'
```

## Parameters & Customization

**Stage 1 (Generation):**
- `--models baseline-race diff-control-race` — which models to generate with
- `--base-model` — which trained checkpoint family (`t5-base`, `flan-t5-base`, `flan-t5-large`)
- `--num-per-difficulty N` — how many questions per difficulty (default 1)
- `--num-beams` — beam search width (default 4; more = slower but more diverse)
- `--do-sample`/`--num-samples`/`--temperature`/`--top-p` — sampling instead of beam search
- `--limit N` — cap number of test examples (for testing)

**Stage 2 (QA):**
- `--models <list>` — override which QA models to use (default: all 4)
- `--batch-size` — inference batch size (default 8)

**Stage 3 (Evaluate):** TBD once the script is written.
