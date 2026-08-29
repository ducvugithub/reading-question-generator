# QG Training Details

## ⚠️ Known limitation: difficulty token is confounded with passage style

Every RACE passage belongs to exactly one difficulty subset (RACE-middle→EASY,
RACE-high→MEDIUM, RACE-C→HARD are disjoint sources), so **no passage is ever
seen with more than one difficulty token during training** — despite each
passage having ~3.65 questions on average (see "Passage-level split" below),
all of them share the same single difficulty label.

This means low training loss on `diff-control-race` does **not** by itself
prove the `<EASY|MEDIUM|HARD>` token is steering generation. It's equally
consistent with the model picking up on passage-level style/vocabulary
differences between the three RACE subsets and ignoring the token entirely —
the token could be fully redundant with what the passage itself already
implies. This is a structural property of RACE's design, not something fixable
in our data prep.

**Mitigation — test this explicitly, don't assume it from training metrics:**
force all three difficulty tokens (`<EASY>`, `<MEDIUM>`, `<HARD>`) onto the
*same* test passage and check whether the generated question actually changes.
If it doesn't change regardless of the forced token, the model learned to
ignore the token. `question_generation/scripts/generate_qg_questions.py` does
exactly this (loops all 3 difficulties per passage, plus `baseline-race` as a
control group that should show ~no variation since it never sees the token at
all). Cross-check with QDE difficulty scoring on the generated questions once
that's run.

## Base model

T5 family, run across 3 base models for the 2 comparison pairs (4 model types,
12 runs total). `baseline-all` was dropped — it's not part of either fair
comparison pair and isn't needed to validate difficulty/focus-span control.

| Base model | Params | Batch size |
|---|---|---|
| `google-t5/t5-base` | ~220M | 16 |
| `google/flan-t5-base` | ~220M | 16 |
| `google/flan-t5-large` | ~770M | 8 (auto-shrunk, see job script) |

`google/flan-t5-large` was chosen over plain `t5-large` because flan-T5's
instruction tuning generally gives a better fine-tuning starting point; running
`t5-base` (non-flan) alongside `flan-t5-base` gives a clean flan-vs-non-flan
comparison at the base size, so we don't need a non-flan large run too.

**Future work (not yet implemented):** GPT-style decoder-only models (e.g.
`gpt2`) fine-tuned with a concatenated passage+prompt+question causal-LM format.
Different architecture family from T5's encoder-decoder, so it needs its own
data-prep and training script rather than a `--base-model` swap.

Other alternatives considered but not pursued for this experiment:

| Model | Params | Notes |
|---|---|---|
| `t5-small` | ~60M | Faster, weaker generation quality — useful for quick iteration only |
| `t5-large` | ~770M | Skipped in favor of flan-t5-large (see above) |
| BERT-style (RoBERTa, DeBERTa, etc.) | — | Encoder-only, can't generate text without bolting on a randomly-initialized decoder; already used elsewhere in this project as QDE difficulty classifiers, not for QG |
| `facebook/bart-base` | ~140M | Different architecture, common in QG literature, needs different input formatting |

### Output path

Models are saved under `question_generation/models/qg/<base_model_slug>/<model_type>/final/`,
e.g. `question_generation/models/qg/t5-base/baseline-race/final/` or
`question_generation/models/qg/flan-t5-large/diff-control-race/final/` —
`base_model_slug` is the part after the last `/` in `--base-model`.

## Data

- Source: `data/qg/<model_type>/{train,val,test}.jsonl`, produced by
  `question_generation/scripts/prepare_qg_test_sets.py` (raw split) +
  `prepare_qg_data.py` (per-model-type formatting).
- Max target length: 64 tokens (p99 is 29-32 across all model types; outliers
  above 64 are <0.05% of records, negligible to truncate)
- Max input length — **adaptive per dataset**, auto-selected in
  `train_qg_t5base.job` based on whether `race` appears in the model type:
  - `baseline-hotpot` / `focus-control-hotpot`: 512 (HotpotQA gold passages
    are short — only 0.1% exceed 512)
  - `baseline-race` / `diff-control-race`: 1024 (RACE passages, especially
    RACE-C/college-level, run much longer — 512 truncated 11-13% of inputs;
    p99 is ~950-955, so 1024 covers nearly all of them)

Token length distribution measured on roihu-gpu (`google-t5/t5-base` tokenizer,
train split, before `baseline-all` was dropped):

```
baseline-race:
  input  — mean=386 p95=638 p99=949 max=1632 >512: 12.6%
  target — mean=14 p95=23 p99=29 max=90 >64: 0.0%
diff-control-race:
  input  — mean=392 p95=644 p99=955 max=1637 >512: 13.3%
  target — mean=14 p95=23 p99=29 max=90 >64: 0.0%
baseline-hotpot:
  input  — mean=189 p95=334 p99=407 max=579 >512: 0.1%
  target — mean=20 p95=31 p99=41 max=103 >64: 0.0%
focus-control-hotpot:
  input  — mean=193 p95=334 p99=409 max=579 >512: 0.1%
  target — mean=20 p95=31 p99=41 max=103 >64: 0.0%
```

### Dataset sizes (train / val / test)

| model_type | train | val | test |
|---|---|---|---|
| baseline-race | 86,898 | 10,079 | 3,591 |
| diff-control-race | 86,898 | 10,079 | 3,591 |
| baseline-hotpot | 14,016 | 1,684 | 1,756 |
| focus-control-hotpot | 14,016 | 1,684 | 1,756 |

`baseline-race`/`diff-control-race` and `baseline-hotpot`/`focus-control-hotpot`
are fair-comparison pairs — each pair shares identical train/val/test passages
and questions, differing only in input conditioning.

### Passage-level split (no train/test leakage)

The train/val/test split is keyed on the **passage** (MD5 hash), not the
question — `_split_bucket(passage)` in `prepare_qg_test_sets.py`. Since RACE
has multiple questions per passage (avg 3.64, mode 3-4), splitting by row
instead of by passage would leak the same passage across splits; hashing by
passage guarantees every question for a given passage lands in the same split.
Verified: zero passage overlap between train/val/test for both datasets.

| Dataset | Split | Questions | Unique passages |
|---|---|---|---|
| RACE | train | 86,898 | 23,894 |
| RACE | val | 10,079 | 2,773 |
| RACE | test | 3,591 | 900 |
| RACE | **total** | **100,568** | **27,567** |
| HotpotQA (comparison) | train | 14,016 | 13,952 |
| HotpotQA (comparison) | val | 1,684 | 1,682 |
| HotpotQA (comparison) | test | 1,756 | 1,746 |
| HotpotQA (comparison) | **total** | **17,456** | **17,380** |

RACE averages ~3.65 questions/passage (23,894 unique passages → 86,898 train
questions); HotpotQA is close to 1:1 (13,952 passages → 14,016 train questions)
since each HotpotQA gold passage is a specific 2-document pairing built for one
question.

**Known confound:** see "⚠️ Known limitation" at the top of this doc — each
RACE passage belongs to exactly one difficulty subset, so no passage is ever
seen with more than one difficulty token during training.

## Optimization

- Epochs: 5
- Batch size: 16 (per device, train & eval)
- Learning rate: 5e-4
- Warmup steps: 200
- Weight decay: 0.01
- Mixed precision: **bf16** enabled (not fp16 — see note below)

**fp16 vs bf16:** the flan-t5-base/flan-t5-large runs were initially launched with
`--fp16` and silently produced garbage (`loss: 0` from gradient-scaler overflow,
`eval_loss: nan`) — a known incompatibility, since FLAN-T5 was pretrained in
bfloat16 and has a narrower safe range under float16's limited dynamic range.
Switched to `--bf16` (GH200 supports it natively) for all base models going
forward, including plain `t5-base`.

## Evaluation / checkpointing

- Eval strategy: every 500 steps
- Save strategy: every 500 steps, `save_total_limit=2` (keeps only 2 checkpoints on disk)
- Metric for best model: `eval_loss` (lower is better)
- `load_best_model_at_end=True` — final saved model is the best-eval checkpoint,
  not necessarily the last
- Early stopping: patience = 3 evals with no improvement (stops if `eval_loss`
  doesn't improve for 3×500 = 1500 steps)
- Generation during eval: `predict_with_generate=True`, `generation_max_length=64`
- Eval metric logged: ROUGE (via `evaluate` library), if `evaluate`/`rouge_score`
  are installed (done at job start via `pip install --user evaluate rouge_score sacrebleu`)

## Compute (SLURM — `question_generation/slurms/train_qg_t5base.job`)

- Partition: `gpumedium`
- 1x GH200 GPU per array task
- 16 CPUs, time limit 8h
- Requested 96G mem (SLURM auto-bumps to ~212G based on node/GPU pairing)

## Launch

```bash
cd /scratch/project_2006601/ducvu/reading-question-generator

# t5-base (default)
sbatch --array=0-3 question_generation/slurms/train_qg_t5base.job

# flan-t5-base
sbatch --array=0-3 --job-name=qg_flan_base \
    --export=BASE_MODEL=google/flan-t5-base \
    question_generation/slurms/train_qg_t5base.job

# flan-t5-large (batch size auto-shrinks to 8 inside the job script)
sbatch --array=0-3 --job-name=qg_flan_large \
    --export=BASE_MODEL=google/flan-t5-large \
    question_generation/slurms/train_qg_t5base.job
```

Array index → model_type mapping (`STEPS` in the job file), same for every base model:

| array index | model_type | max-input |
|---|---|---|
| 0 | baseline-race | 1024 |
| 1 | diff-control-race | 1024 |
| 2 | baseline-hotpot | 512 |
| 3 | focus-control-hotpot | 512 |

Output: each saves to `question_generation/models/qg/<base_model_slug>/<model_type>/final/`.
