# Reading Question Generator

Research project: difficulty-controlled and focus-span-conditioned question
generation for English reading comprehension, plus a Question Difficulty
Estimator (QDE).

**Current status:** difficulty-token conditioning (`diff-control-race`) is
trained but does not produce measurable difficulty steering — see
`docs/experiment_plan.md` for the current status table and root cause.

## Modules

```
question_generation/     QG training, generation, evaluation (T5 seq2seq)
  scripts/                prepare_qg_test_sets.py, prepare_qg_data.py,
                          train_seq2seq.py, generate_qg_questions.py,
                          augment_qg_difficulty_llm.py
  slurms/                 Roihu GPU jobs (train_qg_t5base.job, etc.)
  docs/
    training_details.md              training config, dataset sizes/splits
    evaluation_plan.md                evaluation pipeline, scripts, metrics
    difficulty_steering_mechanisms.md 5 candidate conditioning mechanisms
    related_work_qg.md               literature survey

question_difficulty/     Question Difficulty Estimator (QDE)
  methods/
    feature_based/        linguistic features (features.py)
  scripts/
    prepare_qde_data.py, train_feature_based.py, train_encoder.py,
    train_contrastive.py
  slurms/                 Roihu GPU jobs
  docs/
    cognitive_difficulty_estimation.md   QDE methods, known passage-confound
                                          limitation, per-question signal plan

question_answering/      QA-model-based answerability validation
  scripts/
    run_qa_models.py, assess_llm_quality.py, llm_assessor.py
  slurms/
  docs/
    qa_model_battery.md   Which QA models, why, known deberta-v3-base issue

scripts/
  download_resources.py  Download RACE, HotpotQA to HF cache
  setup.sh

legacy/                  Deprecated M1-M6/KG-based pipeline — not part of the
                          current roadmap, kept for reference only
```

## Datasets

| Dataset | HF path | Role |
|---|---|---|
| RACE-middle/high | `ehovy/race` | `baseline-race`/`diff-control-race` training, EASY/MEDIUM labels |
| RACE-C | `tasksource/race-c` | `baseline-race`/`diff-control-race` training, HARD label |
| HotpotQA (comparison-type) | `hotpotqa/hotpot_qa` distractor | `baseline-hotpot`/`focus-control-hotpot` training |

`MultiRC` was dropped from the pipeline (missing evidence annotations in the
available dataset variant).

## Setup

```bash
bash scripts/setup.sh                          # Git hooks, Claude memory
python scripts/download_resources.py           # Download datasets to HF cache
bash scripts/prepare_all_data.sh --qg-only      # Split + format QG training data
```

Roihu training: push to git → `git pull` on Roihu → `sbatch <job>`
(`.job` files are gitignored — sync via `scp` instead of git).

## Docs index

- `docs/experiment_plan.md` — project-wide roadmap, current status, research questions
- `question_generation/docs/training_details.md` — training config, dataset sizes/splits
- `question_generation/docs/evaluation_plan.md` — evaluation pipeline, scripts, metrics
- `question_generation/docs/difficulty_steering_mechanisms.md` — conditioning mechanisms compared
- `question_generation/docs/related_work_qg.md` — literature review
- `question_difficulty/docs/cognitive_difficulty_estimation.md` — QDE methods, known limitations
- `question_answering/docs/qa_model_battery.md` — QA model list, why these, known `deberta-v3-base` issue
