# QA Model Architecture: Extractive vs. Generative

`question_answering/qa_model.py` defines `QAModel` (abstract base), `ExtractiveQAModel`,
and `GenerativeQAModel` — both implemented. This exists because "just swap the model name"
doesn't work across QA paradigms: extractive and generative models produce answers by
fundamentally different mechanisms, so attention extraction and answer decoding both need
paradigm-specific code, not just a different `model_name` string.

## Extractive (`ExtractiveQAModel`) — BERT / RoBERTa / DeBERTa + `AutoModelForQuestionAnswering`

```
INPUT:  [CLS] question tokens [SEP] passage tokens [SEP]
                    │
                    ▼
        ┌───────────────────────┐
        │   ENCODER              │   <- ONE forward pass, that's it
        │   (self-attention,     │
        │    per layer)          │
        └───────────┬────────────┘
                    │
        seq_len × seq_len attention matrix
                    │
        slice: rows = question tokens
               cols = passage tokens
                    │
                    ▼
        question→passage attention block
                    │
          mean over question-rows
                    ▼
        per-passage-token vector  ──────►  ENTROPY
                    │
        separately: start_logits / end_logits
                    ▼
        answer = passage[start_idx : end_idx]   <- a SPAN, pointed to, not written
```

One forward pass gives you everything: the self-attention matrix (sliced to
question-rows × passage-cols) *is* the "where did the model look" signal, and start/end logits
point at a contiguous span in the passage. `ExtractiveQAModel.get_attention_distribution()` and
`predict_answer()` both just run this one forward pass.

## Generative (`GenerativeQAModel`) — T5 / BART + `AutoModelForSeq2SeqLM`

The entropy signal uses ONLY the encoder, never the decoder -- deliberately. Decoder
cross-attention is computed fresh at each generation step (one matrix per generated token), so
using it for entropy would require picking how to collapse N per-step matrices into one, and
that choice is entangled with generated-answer length (a confound: answer length correlates with
question type, the exact thing this signal is trying to measure). "Question attending to
passage" only exists in the ENCODER's self-attention -- computed in one forward pass, before any
generation happens, completely independent of how long the eventual generated answer turns out
to be:

```
INPUT:  "question: ... context: ..."
                    │
                    ▼
        ┌───────────────────────┐
        │   ENCODER               │   <- ONE forward pass, via model.get_encoder() --
        │   (self-attention)      │      the decoder is never touched for this signal
        └───────────┬────────────┘
                    │
        question→passage attention block   (same slicing idea as extractive,
                    │                        just located via character offsets
                    ▼                        against the prompt template instead
              ENTROPY                        of sequence_ids() -- T5 has no
                                              [SEP]/segment-ID concept)

        Separately, for predict_answer() only (NOT part of the entropy signal):

        ┌─────────────────────────────────────────────┐
        │  DECODER (autoregressive, one step at a time) │
        │  step 1: generate "go"                         │
        │  step 2: generate "to"   (conditioned on "go")  │
        │  step 3: generate "school"                      │
        │      ...                                        │
        └─────────────────────┬─────────────────────────┘
                              │
        answer = "go to school"   <- WRITTEN token-by-token, not a span.
        confidence = mean per-step token probability (from generate(...,
        output_scores=True)) -- there's no start×end softmax analog here,
        so this is the generative equivalent, not a mechanical port.
```

`GenerativeQAModel.get_attention_distribution()` returns the same keys as
`ExtractiveQAModel.get_attention_distribution()` (`sentences`, `distribution`, `entropy`,
`entropy_norm`, `num_tokens`, `tok_entropy`, `tok_entropy_norm`, `layer`), so callers don't need
to branch on which paradigm a `QAModel` instance is.

## Why the split, concretely

- `QAModel` (base): holds entropy math (`entropy`, `normalized_entropy`, `normalize`,
  `sentence_spans`) — this is pure math, identical for both paradigms once you have *a*
  distribution to feed it.
- `ExtractiveQAModel`: owns tokenizer/model loading, `predict_answer()`, and both attention
  extraction methods (`get_attention_distribution` for one layer, `get_all_layers_attention_distribution`
  for every layer in one forward pass).
- `GenerativeQAModel`: same public shape as `ExtractiveQAModel` for `predict_answer()` and
  `get_attention_distribution()`, but no `get_all_layers_attention_distribution()` yet -- not
  fundamentally harder (same `model.get_encoder()(..., output_attentions=True)` call already
  returns every layer's attentions), just not written since nothing's used it yet.
- `question_difficulty/methods/feature_based/difficulty_signals.py`'s `AttentionDispersionSignal`
  (the `DifficultySignal` used for QG training feature extraction) now holds an `ExtractiveQAModel`
  instance (`self.qa_model`) and delegates to it, instead of duplicating tokenizer/model loading
  and attention-slicing itself. `question_difficulty/notebooks/entropy_manual_review.ipynb` does
  the same — `predict_answer = signal.qa_model.predict_answer` instead of a hand-rolled copy of
  the span-decoding logic that used to live directly in the notebook.
