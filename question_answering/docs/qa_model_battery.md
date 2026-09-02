# QA Model Battery

Single source of truth for which QA models this project uses and why. This
battery is dual-purpose — consumed by two different pipelines, which each
document *how* they use it, not *which models* or *why*:

- Answerability validation of generated QG questions — see
  `question_generation/docs/evaluation_plan.md` (Stage 2a)
- Per-question difficulty signal extraction (attention dispersion, pass-rate)
  — see `question_difficulty/docs/cognitive_difficulty_estimation.md`'s
  "Method 4"

Defined in `question_answering/scripts/run_qa_models.py`'s `_QA_MODELS`.

## The models

| Model | SQuAD-finetuned? | Role |
|---|---|---|
| `deepset/roberta-base-squad2` | ✅ | Strongest of the four; primary source for attention-dispersion extraction |
| `google-bert/bert-base-uncased-finetuned-squad` | ✅ | Mid-capability |
| `mrm8488/distilroberta-base-finetuned-squad` | ⚠️ **Broken (2026-09-02)** | 404s on HF Hub as of this date — worked earlier in this project (used successfully in the n=102 signal validation run), likely renamed/removed since. Needs a replacement, not yet picked. |
| `microsoft/deberta-v3-base` | ⚠️ **No** | This is the base pretrained checkpoint, never fine-tuned on SQuAD. It can still run and produce *some* output, but its attention/predictions aren't grounded in the QA task the way the other three are. |

## Known issue: `deberta-v3-base` isn't actually a QA model

It's listed alongside three genuinely SQuAD-finetuned models but was never
trained to do extractive QA. Two consequences:

- **For answerability validation** (`evaluation_plan.md`): its pass/fail
  result shouldn't be trusted the same way as the other three — it hasn't
  learned to locate answer spans.
- **For the pass-rate difficulty signal** (`cognitive_difficulty_estimation.md`):
  it's excluded from the battery entirely for this purpose — an untrained
  model failing every item would just look like "everything is HARD," not a
  real difficulty signal.

**Fix — found and verified (2026-09-02):** `deepset/deberta-v3-base-squad2` is
a real SQuAD2-finetuned DeBERTa checkpoint (confirmed loadable, not guessed).
Not yet swapped into `run_qa_models.py`'s `_QA_MODELS` — still needs that
update, plus a replacement for the now-broken `mrm8488` entry above.
