# Question Difficulty Estimator (QDE)

Classifies reading comprehension questions into `EASY / MEDIUM / HARD` using curriculum-grounded labels derived from which RACE exam subset a passage came from (RACE-middle → EASY, RACE-high → MEDIUM, RACE-C → HARD).

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

## ⚠️ Known limitation: all three methods share the same passage-confound

Methods 1-3 all train against the same RACE-subset-inherited label. Since a
passage belongs to exactly one subset, every one of that passage's ~3.65 real
questions shares the same label — the classifier has no forcing pressure to
distinguish genuine per-question difficulty from passage-level style/vocabulary
differences between RACE-middle/high/C. A model that just learns "this reads
like a RACE-C passage" can hit good macro F1 without ever looking at what
makes an individual *question* harder. This is the same confound documented
in `question_generation/docs/training_details.md`'s "Known limitation"
section, discovered while investigating why `diff-control-race`'s difficulty
token wasn't producing measurable steering — see Method 4 below for the fix
under investigation.

---

## Method 4: Per-Question Signal Extraction (addresses the passage-confound)

**Goal:** unlike Methods 1-3, extract a difficulty signal from properties of
the *individual question* (and its relationship to the passage/answer) that
does not just re-derive which RACE subset a passage came from. If this works,
a single passage's several real questions could get *different* difficulty
scores — real, non-synthetic, same-passage contrastive signal, useful both as
a better QDE and as new training labels for `diff-control-race`.

**Ruled out approaches:**
- Synthetic LLM-generated questions — quality concerns (a Haiku-generated
  MEDIUM question read harder than its paired HARD question; Sonnet was
  better but still needs prompt iteration and is an ongoing quality/trust
  risk for training data).
- LLM-as-annotator (rating existing real questions) — wanted something more
  systematic than an LLM's subjective judgment call.
- Full IRT via simulated QA-model "students" — 2PL IRT needs to jointly
  estimate item difficulty and respondent ability, which requires dozens-to-
  hundreds of respondents to be statistically reliable. With only ~3-4 QA
  models as "students," the fit would be underdetermined noise.
- Distractor plausibility (similarity between correct answer and wrong
  options) — not worth pursuing per project call.

**Signal sources (all systematic, no LLM judgment, no generation):**

1. **Attention dispersion** — feed `(question, passage)` into a SQuAD-finetuned
   QA model (`deepset/roberta-base-squad2`), extract the question→passage
   attention sub-block, summarize how concentrated (one sentence, simple) vs.
   spread out (many sentences, harder) it is.
2. **QA-model pass-rate** — run the 3 properly SQuAD-finetuned models from
   the QA battery (see `question_answering/docs/qa_model_battery.md` — excludes
   `microsoft/deberta-v3-base`, which isn't actually SQuAD-finetuned) against
   the gold answer. Fraction of models correct = rough difficulty indicator.
3. **Answer extractiveness** — plain text overlap between the gold answer and
   the passage (no model). Verbatim/near-verbatim match → easy (direct
   lookup); not stated directly anywhere → harder (requires synthesis).
4. **Question-answer similarity** — plain text/lexical similarity between the
   question and the gold answer (no model). Direct restatement → easy;
   indirect relation → harder.

Requires recovering `options`/`answer` from the raw HF RACE dataset into
`prepare_qg_test_sets.py`'s output — currently dropped since QG training
doesn't need them. This is a separate side-channel for difficulty signal
extraction, not a change to QG training data itself.

**Interface design** — one independently pluggable, testable class per
signal, combined by a simple aggregator:

```python
from abc import ABC, abstractmethod


class DifficultySignal(ABC):
    """One systematic, model-behavior-or-text-based signal contributing to
    per-question difficulty. Implementations should not depend on any other
    signal, and should not use the RACE subset label at all."""

    name: str

    @abstractmethod
    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        """Return one or more named scalar features for this triple."""
        ...


class AttentionDispersionSignal(DifficultySignal):
    name = "attention_dispersion"

    def __init__(self, qa_model_name: str = "deepset/roberta-base-squad2"):
        ...

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        ...


class QAPassRateSignal(DifficultySignal):
    name = "qa_pass_rate"

    def __init__(self, qa_model_names: list[str] | None = None):
        ...

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        ...


class AnswerExtractivenessSignal(DifficultySignal):
    name = "answer_extractiveness"

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        ...


class QuestionAnswerSimilaritySignal(DifficultySignal):
    name = "question_answer_similarity"

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        ...


class DifficultySignalExtractor:
    """Runs all registered signals over one (passage, question, answer) triple
    and returns their combined feature dict."""

    def __init__(self, signals: list[DifficultySignal]):
        self.signals = signals

    def extract(self, passage: str, question: str, answer: str) -> dict[str, float]:
        features: dict[str, float] = {}
        for signal in self.signals:
            features.update(signal.compute(passage, question, answer))
        return features
```

Implementation would extend `question_difficulty/methods/feature_based/features.py`
(new signal functions alongside the existing question-only + interaction
features), with a script under `question_difficulty/scripts/` to run them at
scale over RACE. It imports the QA model list from
`question_answering/scripts/run_qa_models.py` rather than duplicating it, so
there's one source of truth for which QA models are "official" in this
project.

**Next step — validate before building anything further.** Compute all 4
signals on a sample of real RACE questions and check:

1. Does each signal actually vary meaningfully across questions (not flat/noise)?
2. Does it correlate at all with the existing RACE subset label (sanity check)?
3. **Critical test**: does it vary *within* a single passage's multiple real
   questions? If yes — real same-passage contrastive signal exists. If no —
   this doesn't solve the confound either, and the plan needs to change.

Only after step 3 passes does it make sense to turn any of this into a
trained embedding/classifier (feature-based classifier on top of these
signals, or a neural embedding) or feed it into an adapter's FiLM-style
conditioning for QG training (see
`question_generation/docs/difficulty_steering_mechanisms.md`).

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
