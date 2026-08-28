# QG Evaluation Flow

Three-stage pipeline for evaluating difficulty-controlled question generation:

```
Stage 1: Generate Questions
    ↓
Stage 2a: QA Model Validation (SQuAD-based)
    ↓
Stage 2b: LLM Quality Assessment (open-source 7B models)
    ↓
Stage 3: Evaluate & Combine Signals (TBD)
```

---

## Stage 1: Generate Questions

**Script:** `question_generation/scripts/generate_qg_questions.py`

**Input:** Test set from `data/qg/{baseline}/test.jsonl`

**Output:** `question_generation/results/qg_generated.jsonl`

**What it does:**
- Loads trained QG models (baseline, diff-control)
- For each test passage, generates questions at all difficulty levels (EASY, MEDIUM, HARD)
- Configurable: `--num-per-difficulty N` (default 1 per level)
- Each model×difficulty combination is independent

**Format:**
```json
{
  "passage": "...",
  "original_question": "...",
  "model": "baseline" | "diff-control",
  "target_difficulty": "EASY" | "MEDIUM" | "HARD",
  "generated_question": "...",
  "beam_rank": 0
}
```

**Local usage (quick test):**
```bash
python question_generation/scripts/generate_qg_questions.py \
  --models baseline diff-control \
  --num-per-difficulty 1 \
  --limit 100
```

**On Roihu (full test set):**
```bash
sbatch question_generation/slurms/generate_qg_questions.job
```

---

## Stage 2: Validate Answerability (QA Models)

**Script:** `question_answering/scripts/run_qa_models.py`

**Input:** `question_generation/results/qg_generated.jsonl` (from Stage 1)

**Output:** `question_answering/results/qa_scores.jsonl`

**What it does:**
- Loads 4 pre-trained QA models (RoBERTa, BERT, DistilRoBERTa, DeBERTa)
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
    },
    ...
  },
  
  "qa_scores": {"model1": 0.95, "model2": 0.92, ...},
  "qa_answer_spans": {"model1": "Paris", "model2": "Paris", ...},
  "qa_agreement": true,
  "qa_consensus_answer": "Paris",
  "qa_avg_score": 0.93,
  "qa_num_models": 4
}
```

**Note:** Stage 2 records everything; Stage 3 decides whether to use:
- Score-based: average confidence across models
- Agreement-based: do all models extract the same answer?
- Custom: any other heuristic

**Local usage:**
```bash
python question_answering/scripts/run_qa_models.py
```

**On Roihu (full):**
```bash
sbatch question_answering/slurms/run_qa_models.job
```

---

## Stage 2b: LLM Quality Assessment

**Script:** `question_answering/scripts/assess_llm_quality.py`

**Models:** Qwen 2-7B, Llama 2-7B-chat, Mistral-7B

**Input:** `question_generation/results/qg_generated.jsonl` (from Stage 1)

**Output:** `question_answering/results/llm_quality_assessments.jsonl`

**What it does:**
- Loads 3 open-source 7B LLMs (fit on single V100 GPU)
- Assesses each question on:
  - Grammaticality (1-5)
  - Answerability (1-5)
  - Clarity (1-5)
  - Relevance (1-5)
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
    },
    "meta-llama/Llama-2-7b-chat-hf": {...},
    "mistralai/Mistral-7B-Instruct-v0.1": {...}
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
python question_answering/scripts/assess_llm_quality.py \
  --limit 10  # test with 10 examples
```

**On Roihu (full):**
```bash
sbatch question_answering/slurms/assess_llm_quality.job
```

---

## Stage 3: Evaluate & Combine Signals (TBD)

**Script:** `question_generation/scripts/evaluate_qg.py` (to be written)

**Input:** 
- `question_answering/results/qa_scores.jsonl` (from Stage 2)
- QDE predictions on generated questions

**Output:** Evaluation report with:
- **Difficulty validation** (QDE predictions match target)
- **Answerability** (QA model scores)
- **Linguistic metrics** (length, vocabulary, complexity by difficulty)

---

## Full Workflow on Roihu

```bash
# 1. Generate questions (2h GPU)
sbatch question_generation/slurms/generate_qg_questions.job
# Wait for job to finish; check logs/qg_generate_*.log

# 2a. Run QA models (SQuAD-based) (4h GPU)
sbatch question_answering/slurms/run_qa_models.job
# Wait for job to finish; check logs/qa_models_*.log

# 2b. Assess with LLMs (quality assessment) (6h GPU)
sbatch question_answering/slurms/assess_llm_quality.job
# Wait for job to finish; check logs/llm_assess_*.log

# 3. Run QDE on generated questions (to be integrated into Stage 3)
# This validates that HARD generations actually score as HARD, etc.

# 4. Evaluate & Combine (Stage 3 script — TBD)
# Inputs: QA scores + LLM quality + QDE difficulty predictions
# Generates report with metrics
```

---

## Quick Local Test

```bash
# Generate 10 examples with baseline model only
python question_generation/scripts/generate_qg_questions.py \
  --models baseline \
  --num-per-difficulty 1 \
  --limit 10

# Run QA on them (will use CPU if no GPU)
python question_answering/scripts/run_qa_models.py

# Inspect output
head question_answering/results/qa_scores.jsonl | jq '.qa_scores'
```

---

## Parameters & Customization

**Stage 1 (Generation):**
- `--models baseline diff-control` — which models to generate with
- `--num-per-difficulty N` — how many questions per difficulty (default 1)
- `--num-beams` — beam search width (default 4; more = slower but more diverse)
- `--limit N` — cap number of test examples (for testing)

**Stage 2 (QA):**
- `--models <list>` — override which QA models to use (default: all 4)
- `--batch-size` — inference batch size (default 8)

**Stage 3 (Evaluate):**
- TBD once we finalize metrics
