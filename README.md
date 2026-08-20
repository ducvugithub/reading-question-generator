# Reading Question Generator

Research project: difficulty-controlled and focus-span-conditioned question generation for English reading comprehension, with an IRT-validated Question Difficulty Estimator (QDE).

## Research Roadmap

| Step | Name | Input | Output | Status |
|---|---|---|---|---|
| **0** | Baseline QG | `passage` | `question` | data ready, training pending |
| **1** | QDE | `passage + question + answer` | `EASY / MEDIUM / HARD` | code done, training pending |
| **2** | Difficulty-controlled QG | `passage + difficulty` | `question` | data ready, training pending |
| **3** | Focus-span QG | `passage + focus_span` | `question` | data TBD |
| **4** | Full (M6) | `passage + focus_span + difficulty` | `question` | depends on 1+3 |

**Step 4** is the main novelty: QDE (Step 1) enriches HotpotQA/MultiRC with difficulty labels → combined training for focus-span + difficulty controlled generation. Answer span is dropped as input.

See `docs/experiment_plan.md` for full details.

---

## Modules

```
question_difficulty/         Question Difficulty Estimator (QDE)
  methods/
    feature_based/           GradientBoosting + 14 linguistic features
    encoder/                 RoBERTa/DeBERTa fine-tune ([CLS] classifier)
    contrastive/             Triplet loss + projection head + LR probe
  scripts/
    prepare_qde_data.py      SQuAD→EASY, RACE-middle→MEDIUM, RACE-high→HARD
    train_feature_based.py
    train_encoder.py
    train_contrastive.py
  slurms/                    Mahti GPU jobs for encoder + contrastive training
  docs/
    cognitive_difficulty_estimation.md   QDE design and method comparison

question_generation/         Question Generation (seq2seq T5)
  scripts/
    build_qg_dataset.py      SQuAD + KG extraction pipeline
    add_difficulty_annotations.py   Haiku-based cognitive difficulty labelling
    prepare_t5_inputs.py     Format JSONL → T5 input strings (M1–M5)
    train_seq2seq.py         T5 fine-tuning
  slurms/
    train_qg_t5base.job      Main Mahti job (array over model types)
  docs/
    experiment_plan.md       Experimental conditions and training plan
    evaluation_plan.md       Metrics and evaluation procedure
    related_work_qg.md       Literature survey

knowledge_graph/             KG extraction from passages (used by M3/M4)
  extractor.py               NER + dependency parsing → Triple objects
  graph.py                   NetworkX MultiDiGraph
  coref.py                   Heuristic coreference resolution

scripts/
  download_resources.py      Download SQuAD, RACE, model weights
```

### Legacy note

`question_generation/difficulty/` contains the old rule-based difficulty estimator (Bloom-level heuristics + CEFR scoring). It is not used in the new roadmap but kept for reference. The new QDE lives in `question_difficulty/`.

---

## Datasets

| Dataset | HF path | Role |
|---|---|---|
| RACE++ | `chujiezheng/RACE++` | QDE labels (middle→EASY, high→MEDIUM, college→HARD) + Steps 0 & 2 QG training |
| HotpotQA | `hotpot_qa` distractor | Steps 0 & 3 (supporting\_facts as focus span; filter `type=="comparison"`) |
| MultiRC | `super_glue` multirc | Steps 0 & 3 (evidence sentences as focus span) |

---

## Setup

```bash
# Download datasets and weights
python scripts/download_resources.py

# Prepare QDE training data
python question_difficulty/scripts/prepare_qde_data.py

# Prepare T5 inputs for QG (e.g., M2 = difficulty-controlled)
python question_generation/scripts/prepare_t5_inputs.py --model-types m2
```

Mahti training: push to git → `git pull` on Mahti → `sbatch <job>`.

---

## Model Naming

| ID | Description | Input |
|---|---|---|
| M1 | Baseline | `passage` |
| M2 | + haiku\_cog difficulty token | `passage + <difficulty>` |
| M3 | + linearized KG | `passage + kg_text + <difficulty>` |
| M4 | + KG + difficulty | `passage + kg_text + <difficulty>` |
| M5 | Curriculum labels (SQuAD+RACE) | `passage + <difficulty>` |
| M6 | Focus span + difficulty (Step 4) | `passage + focus_span + <difficulty>` |
