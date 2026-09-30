"""
QA model wrappers, split by paradigm -- extractive (answer = a span pointed
to in the passage, one forward pass) vs. generative (answer = text written
token-by-token, one forward pass per output token). See
question_answering/docs/qa_model_architecture.md for the full flow diagrams;
this docstring is the short version.

Extractive (BERT/RoBERTa/DeBERTa + AutoModelForQuestionAnswering):
  question+passage -> ONE encoder forward pass -> one self-attention matrix
  per layer -> slice question-rows x passage-cols -> that IS the "where did
  the model look" signal. start/end logits point at a passage span.

Generative (T5/BART + AutoModelForSeq2SeqLM):
  encoder self-attention exists too, but doesn't reflect "while answering".
  The model writes the answer one token at a time; each step gets its own
  decoder cross-attention distribution over the passage. There's no single
  matrix to take entropy of without first choosing how to aggregate across
  generation steps -- that's a real design decision, not implemented here.

Decoder-only (Qwen/Llama-style + AutoModelForCausalLM):
  no encoder at all -- "question attending to passage" only exists during
  PREFILL, the one forward pass that encodes passage+question+"Answer:"
  before any generation happens. Causal masking means an earlier token
  (passage, question) can never attend to a later one (anything generated
  afterward), so this attention is fixed the instant prefill finishes --
  same "one static matrix" shape as the other two paradigms, just produced
  by a causal model instead of a bidirectional one.

All three subclasses share entropy math (pure math, paradigm-independent)
via the QAModel base class. Only attention extraction and answer decoding
differ. See question_answering/attention_inspection.py for the model-
agnostic manual-inspection helpers (HTML attention highlighting) that work
against any QAModel subclass.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from question_answering.qa_evaluator import _STOPWORDS

_SENT_RE = re.compile(r"[.!?]+")


class QAModel(ABC):
    """One QA model, wrapping how it predicts an answer and how attention
    is extracted from it. Subclasses implement the paradigm-specific parts;
    entropy math is shared here since it doesn't depend on paradigm."""

    @abstractmethod
    def predict_answer(self, passage: str, question: str) -> tuple[str, float]:
        """Returns (answer_text, confidence)."""
        ...

    @abstractmethod
    def get_attention_distribution(self, passage: str, question: str, layer: int | None = None) -> dict:
        """Returns at least {"sentences": [...], "distribution": [...],
        "entropy": float, "entropy_norm": float} for one layer (extractive)
        or one aggregation strategy (generative)."""
        ...

    @staticmethod
    def entropy(dist: list[float]) -> float:
        return -sum(p * math.log(p) for p in dist if p > 0)

    @classmethod
    def normalized_entropy(cls, dist: list[float]) -> float:
        """Raw entropy / log(N) (N = number of outcomes). Raw entropy is
        bounded by log(N), so distributions over more outcomes (longer
        passages) have a higher ceiling regardless of how peaked/spread
        attention actually is -- this rescales to [0, 1] so passages of
        different lengths are comparable."""
        n = len(dist)
        if n <= 1:
            return 0.0
        return cls.entropy(dist) / math.log(n)

    @staticmethod
    def normalize(raw: list[float]) -> list[float]:
        total = sum(raw) or 1.0
        return [v / total for v in raw]

    @staticmethod
    def participation_ratio(dist: list[float]) -> float:
        """1 / sum(p_i^2) -- the Renyi entropy (order 2) "effective count":
        how many outcomes are effectively sharing the mass. Unlike Shannon
        entropy, squaring the weights makes big terms dominate and small
        (noise-floor) terms nearly vanish, so a distribution with a couple
        of real peaks plus a long thin tail scores close to the peak count,
        not inflated by the tail the way entropy would be. Ranges [1, N]."""
        denom = sum(p * p for p in dist)
        return 1.0 / denom if denom > 0 else 0.0

    @classmethod
    def normalized_participation_ratio(cls, dist: list[float]) -> float:
        """(PR - 1) / (N - 1), bounded [0, 1], comparable across passages
        of different length -- same rescaling purpose as normalized_entropy."""
        n = len(dist)
        if n <= 1:
            return 0.0
        return (cls.participation_ratio(dist) - 1) / (n - 1)

    @staticmethod
    def is_punctuation_only(token_text: str) -> bool:
        """True if this (sub)token has no letters or digits -- i.e. it's
        pure punctuation/whitespace. Used to exclude "attention sink"
        tokens (periods etc. that absorb a large, near-identical share of
        attention regardless of the question -- a well-documented
        transformer artifact) from entropy/participation-ratio math.
        Stopwords are deliberately NOT filtered this way -- only
        punctuation -- since words like "not"/"no" matter for negation
        questions and shouldn't be silently dropped. See is_content_word
        for the stricter (punctuation + stopword) filter, offered as a
        separate variant precisely because it CAN blind the signal to
        negation-driven attention shifts."""
        return not re.search(r"[a-zA-Z0-9]", token_text)

    @staticmethod
    def is_content_word(token_text: str) -> bool:
        """True if this (sub)token is neither punctuation-only NOR a
        stopword (reuses QAEvaluator's stopword list, which includes
        "not"/"no"/"and"/"or"/"but" -- so this filter is stricter than
        is_punctuation_only and can suppress genuine negation signal;
        offered as a separate, comparable variant, not a replacement)."""
        return (not QAModel.is_punctuation_only(token_text)
                and token_text.strip().lower() not in _STOPWORDS)

    @staticmethod
    def sentence_spans(passage: str) -> list[tuple[int, int]]:
        """Character (start, end) spans for each sentence in the passage."""
        spans = []
        start = 0
        for m in _SENT_RE.finditer(passage):
            end = m.end()
            spans.append((start, end))
            start = end
        if start < len(passage):
            spans.append((start, len(passage)))
        return spans or [(0, len(passage))]


class ExtractiveQAModel(QAModel):
    """BERT/RoBERTa/DeBERTa-style: AutoModelForQuestionAnswering, one
    encoder forward pass, self-attention sliced to question-rows x
    passage-cols, start/end logits point at a passage span."""

    def __init__(self, model_name: str, max_length: int = 512, device: str | None = None):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def predict_answer(self, passage: str, question: str) -> tuple[str, float]:
        enc = self.tokenizer(
            question, passage, max_length=self.max_length,
            truncation="only_second", return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)
        passage_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        enc = enc.to(self.device)

        with self.torch.no_grad():
            out = self.model(**enc)

        start_probs = self.torch.softmax(out.start_logits[0], dim=-1)
        end_probs = self.torch.softmax(out.end_logits[0], dim=-1)
        start_idx = max(passage_positions, key=lambda i: start_probs[i].item())
        end_idx = max(passage_positions, key=lambda i: end_probs[i].item())
        if end_idx < start_idx:
            end_idx = start_idx
        char_start, char_end = offsets[start_idx][0], offsets[end_idx][1]
        confidence = start_probs[start_idx].item() * end_probs[end_idx].item()
        return passage[char_start:char_end].strip(), confidence

    def get_attention_distribution(self, passage: str, question: str, layer: int | None = None,
                                    head: int | None = None) -> dict:
        """One layer's sentence-level attention distribution, for
        visualization/manual review -- not the full multi-layer sweep
        AttentionDispersionSignal.compute() does for feature extraction.

        layer: None (default) uses layer 11. Pass an explicit int to
        override (e.g. distilbert-squad only has 6 layers).
        head: if None (default), averages across all attention heads in
        this layer. If given an int, uses that ONE head's raw attention
        instead -- see get_per_question_token_attention for why (heads
        specialize; averaging can wash out per-token differentiation a
        single head might show). Check model.config.num_attention_heads
        for the valid range."""
        layer = layer if layer is not None else 11
        enc = self.tokenizer(
            question, passage, max_length=self.max_length,
            truncation="only_second", return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)
        enc = enc.to(self.device)

        with self.torch.no_grad():
            out = self.model(**enc, output_attentions=True)

        sent_spans = self.sentence_spans(passage)
        sentence_texts = [passage[s:e].strip() for s, e in sent_spans]

        question_positions = [i for i, s in enumerate(sequence_ids) if s == 0]
        passage_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not question_positions or not passage_positions:
            return {"sentences": sentence_texts, "distribution": [], "entropy": None}

        layer_heads = out.attentions[layer][0]
        selected_heads = layer_heads.mean(dim=0) if head is None else layer_heads[head]
        sub = selected_heads[question_positions][:, passage_positions]
        per_passage_token = sub.mean(dim=0).tolist()

        token_dist = self.normalize(per_passage_token)

        token_texts = [passage[offsets[pos][0]:offsets[pos][1]] for pos in passage_positions]
        is_punct = [self.is_punctuation_only(t) for t in token_texts]
        is_content = [self.is_content_word(t) for t in token_texts]

        sent_sum = [0.0] * len(sent_spans)
        sent_sum_no_punct = [0.0] * len(sent_spans)
        sent_sum_content = [0.0] * len(sent_spans)
        for tok_idx, pos in enumerate(passage_positions):
            char_start, char_end = offsets[pos]
            if char_start == char_end:
                continue
            for s_idx, (s_start, s_end) in enumerate(sent_spans):
                if s_start <= char_start < s_end:
                    sent_sum[s_idx] += per_passage_token[tok_idx]
                    if not is_punct[tok_idx]:
                        sent_sum_no_punct[s_idx] += per_passage_token[tok_idx]
                    if is_content[tok_idx]:
                        sent_sum_content[s_idx] += per_passage_token[tok_idx]
                    break

        dist = self.normalize(sent_sum)
        dist_no_punct = self.normalize(sent_sum_no_punct)
        dist_content = self.normalize(sent_sum_content)
        token_weights_no_punct = [w for w, p in zip(per_passage_token, is_punct) if not p]
        token_dist_no_punct = self.normalize(token_weights_no_punct)
        token_weights_content = [w for w, c in zip(per_passage_token, is_content) if c]
        token_dist_content = self.normalize(token_weights_content)
        return {
            "sentences": sentence_texts,
            "distribution": dist,
            "entropy": self.entropy(dist),
            "entropy_norm": self.normalized_entropy(dist),
            # "_no_punct" variants: attention-sink tokens (periods etc. --
            # see is_punctuation_only) excluded before summing/normalizing,
            # since they absorb a large, near-identical share of attention
            # regardless of the question and dominate the raw versions above.
            "distribution_no_punct": dist_no_punct,
            "entropy_no_punct": self.entropy(dist_no_punct),
            "entropy_norm_no_punct": self.normalized_entropy(dist_no_punct),
            # "_content_only" variants: punctuation AND stopwords excluded
            # (see is_content_word) -- stricter than "_no_punct"; can
            # suppress negation signal ("not"/"no" are stopwords), so this
            # is offered as a third, separately comparable variant, not a
            # default replacement for the other two.
            "distribution_content_only": dist_content,
            "entropy_content_only": self.entropy(dist_content),
            "entropy_norm_content_only": self.normalized_entropy(dist_content),
            "num_tokens": len(passage_positions),
            "token_distribution": token_dist,
            "token_char_spans": [offsets[pos] for pos in passage_positions],
            "tok_entropy": self.entropy(token_dist),
            "tok_entropy_norm": self.normalized_entropy(token_dist),
            "token_distribution_no_punct": token_dist_no_punct,
            "num_tokens_no_punct": len(token_weights_no_punct),
            "tok_entropy_no_punct": self.entropy(token_dist_no_punct),
            "tok_entropy_norm_no_punct": self.normalized_entropy(token_dist_no_punct),
            "token_distribution_content_only": token_dist_content,
            "num_tokens_content_only": len(token_weights_content),
            "tok_entropy_content_only": self.entropy(token_dist_content),
            "tok_entropy_norm_content_only": self.normalized_entropy(token_dist_content),
            "layer": layer,
        }

    def get_per_question_token_attention(self, passage: str, question: str, layer: int | None = None,
                                          head: int | None = None) -> dict:
        """Like get_attention_distribution, but does NOT average across
        question tokens -- returns one passage-token distribution PER
        question token instead of one aggregate. Use this to find which
        specific question word is pulling attention toward an unexpected
        part of the passage (the aggregate view can't distinguish this).
        Passage-side punctuation tokens are excluded and each row
        renormalized over the rest, same convention as the "_no_punct"
        fields in get_attention_distribution; stopwords are kept.

        layer: None (default) uses layer 11, same as get_attention_distribution.
        head: if None (default), averages across all attention heads in
        this layer, same as get_attention_distribution. If given an int,
        uses that ONE head's raw attention instead -- heads are known to
        specialize (syntax, position, semantics...), so averaging all of
        them can wash out per-question-token differentiation that a
        single head might actually show. See model config for the valid
        head range (e.g. num_attention_heads)."""
        layer = layer if layer is not None else 11
        enc = self.tokenizer(
            question, passage, max_length=self.max_length,
            truncation="only_second", return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)

        question_positions = [i for i, s in enumerate(sequence_ids) if s == 0]
        passage_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not question_positions or not passage_positions:
            return {"question_tokens": [], "token_char_spans": [], "distributions": []}
        enc = enc.to(self.device)

        with self.torch.no_grad():
            out = self.model(**enc, output_attentions=True)

        layer_heads = out.attentions[layer][0]
        selected = layer_heads.mean(dim=0) if head is None else layer_heads[head]
        sub = selected[question_positions][:, passage_positions]

        passage_char_spans = [offsets[pos] for pos in passage_positions]
        is_punct = [self.is_punctuation_only(passage[s:e]) for s, e in passage_char_spans]
        keep_idx = [i for i, p in enumerate(is_punct) if not p]
        spans_no_punct = [passage_char_spans[i] for i in keep_idx]

        question_token_texts = [question[offsets[pos][0]:offsets[pos][1]] for pos in question_positions]
        distributions = [self.normalize([row[i] for i in keep_idx]) for row in sub.tolist()]

        return {
            "question_tokens": question_token_texts,
            "token_char_spans": spans_no_punct,
            "distributions": distributions,
        }

    def get_all_layers_attention_distribution(self, passage: str, question: str) -> dict:
        """Same as get_attention_distribution, but for EVERY layer in one
        forward pass."""
        enc = self.tokenizer(
            question, passage, max_length=self.max_length,
            truncation="only_second", return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)
        enc = enc.to(self.device)

        with self.torch.no_grad():
            out = self.model(**enc, output_attentions=True)

        sent_spans = self.sentence_spans(passage)
        sentence_texts = [passage[s:e].strip() for s, e in sent_spans]

        question_positions = [i for i, s in enumerate(sequence_ids) if s == 0]
        passage_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not question_positions or not passage_positions:
            return {"sentences": sentence_texts, "distributions": [], "entropies": []}

        distributions, entropies, entropies_norm = [], [], []
        for layer_attn in out.attentions:
            avg_heads = layer_attn[0].mean(dim=0)
            sub = avg_heads[question_positions][:, passage_positions]
            per_passage_token = sub.mean(dim=0).tolist()

            sent_sum = [0.0] * len(sent_spans)
            for tok_idx, pos in enumerate(passage_positions):
                char_start, char_end = offsets[pos]
                if char_start == char_end:
                    continue
                for s_idx, (s_start, s_end) in enumerate(sent_spans):
                    if s_start <= char_start < s_end:
                        sent_sum[s_idx] += per_passage_token[tok_idx]
                        break

            dist = self.normalize(sent_sum)
            distributions.append(dist)
            entropies.append(self.entropy(dist))
            entropies_norm.append(self.normalized_entropy(dist))

        return {
            "sentences": sentence_texts,
            "distributions": distributions,
            "entropies": entropies,
            "entropies_norm": entropies_norm,
        }


class GenerativeQAModel(QAModel):
    """T5/BART-style: AutoModelForSeq2SeqLM.

    The attention signal deliberately uses ONLY the encoder, never the
    decoder. Decoder cross-attention is computed fresh at each generation
    step (one matrix per generated token), so using it would require
    picking how to collapse N per-step matrices into one -- a real
    methodological choice, and one that's entangled with generated-answer
    length (a confound, since answer length correlates with question type,
    the exact thing this signal is trying to measure).

    "Question attending to passage" only exists in the ENCODER's
    self-attention, computed in one forward pass over the input BEFORE any
    generation happens -- identical in kind to what ExtractiveQAModel
    already does, just requires locating question/passage token positions
    via character offsets against a known prompt template instead of
    sequence_ids() (T5's tokenizer has no [SEP]/segment-ID concept -- the
    input is one flat string). Generation (predict_answer) is a separate,
    unrelated concern: only needed to produce the answer TEXT for
    correctness evaluation, not for the entropy signal.
    See question_answering/docs/qa_model_architecture.md.
    """

    QUESTION_PREFIX = "question: "
    CONTEXT_PREFIX = " context: "

    def __init__(self, model_name: str, max_length: int = 512):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length

    def _prompt(self, question: str, passage: str) -> tuple[str, int, int]:
        """Returns (full_prompt, question_char_start, passage_char_start).
        Question text is bracketed by QUESTION_PREFIX/CONTEXT_PREFIX;
        passage text runs from passage_char_start to the end."""
        question_char_start = len(self.QUESTION_PREFIX)
        passage_char_start = question_char_start + len(question) + len(self.CONTEXT_PREFIX)
        full_prompt = f"{self.QUESTION_PREFIX}{question}{self.CONTEXT_PREFIX}{passage}"
        return full_prompt, question_char_start, passage_char_start

    def predict_answer(self, passage: str, question: str) -> tuple[str, float]:
        full_prompt, _, _ = self._prompt(question, passage)
        enc = self.tokenizer(full_prompt, max_length=self.max_length, truncation=True, return_tensors="pt")

        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=64,
                output_scores=True, return_dict_in_generate=True,
            )

        answer = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True).strip()
        # confidence: mean per-step token probability of the generated sequence
        step_probs = [self.torch.softmax(score[0], dim=-1).max().item() for score in out.scores]
        confidence = sum(step_probs) / len(step_probs) if step_probs else 0.0
        return answer, confidence

    def get_attention_distribution(self, passage: str, question: str, layer: int | None = None,
                                    head: int | None = None) -> dict:
        """Encoder-only self-attention, question-rows x passage-cols, one
        layer -- same shape/keys as ExtractiveQAModel.get_attention_distribution
        so callers don't need to branch on paradigm.

        layer: None (default) uses layer 11. Pass an explicit int to override.
        head: if None (default), averages across all attention heads in
        this layer; if given an int, uses that ONE head's raw attention
        instead. See ExtractiveQAModel.get_attention_distribution."""
        layer = layer if layer is not None else 11
        full_prompt, question_char_start, passage_char_start = self._prompt(question, passage)
        enc = self.tokenizer(full_prompt, max_length=self.max_length, truncation=True,
                              return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()

        question_positions = [i for i, (s, e) in enumerate(offsets)
                               if e > 0 and question_char_start <= s < passage_char_start - len(self.CONTEXT_PREFIX)]
        passage_positions = [i for i, (s, e) in enumerate(offsets) if e > 0 and s >= passage_char_start]
        sent_spans = self.sentence_spans(passage)
        sentence_texts = [passage[s:e].strip() for s, e in sent_spans]
        if not question_positions or not passage_positions:
            return {"sentences": sentence_texts, "distribution": [], "entropy": None}

        with self.torch.no_grad():
            encoder_out = self.model.get_encoder()(
                input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"),
                output_attentions=True,
            )

        layer_heads = encoder_out.attentions[layer][0]
        selected_heads = layer_heads.mean(dim=0) if head is None else layer_heads[head]
        sub = selected_heads[question_positions][:, passage_positions]
        per_passage_token = sub.mean(dim=0).tolist()
        token_dist = self.normalize(per_passage_token)

        token_char_spans = [(offsets[pos][0] - passage_char_start, offsets[pos][1] - passage_char_start)
                             for pos in passage_positions]
        token_texts = [passage[s:e] for s, e in token_char_spans]
        is_punct = [self.is_punctuation_only(t) for t in token_texts]
        is_content = [self.is_content_word(t) for t in token_texts]

        sent_sum = [0.0] * len(sent_spans)
        sent_sum_no_punct = [0.0] * len(sent_spans)
        sent_sum_content = [0.0] * len(sent_spans)
        for tok_idx, pos in enumerate(passage_positions):
            char_start, char_end = offsets[pos]
            passage_local_start = char_start - passage_char_start
            for s_idx, (s_start, s_end) in enumerate(sent_spans):
                if s_start <= passage_local_start < s_end:
                    sent_sum[s_idx] += per_passage_token[tok_idx]
                    if not is_punct[tok_idx]:
                        sent_sum_no_punct[s_idx] += per_passage_token[tok_idx]
                    if is_content[tok_idx]:
                        sent_sum_content[s_idx] += per_passage_token[tok_idx]
                    break

        dist = self.normalize(sent_sum)
        dist_no_punct = self.normalize(sent_sum_no_punct)
        dist_content = self.normalize(sent_sum_content)
        token_weights_no_punct = [w for w, p in zip(per_passage_token, is_punct) if not p]
        token_dist_no_punct = self.normalize(token_weights_no_punct)
        token_weights_content = [w for w, c in zip(per_passage_token, is_content) if c]
        token_dist_content = self.normalize(token_weights_content)
        return {
            "sentences": sentence_texts,
            "distribution": dist,
            "entropy": self.entropy(dist),
            "entropy_norm": self.normalized_entropy(dist),
            "distribution_no_punct": dist_no_punct,
            "entropy_no_punct": self.entropy(dist_no_punct),
            "entropy_norm_no_punct": self.normalized_entropy(dist_no_punct),
            "distribution_content_only": dist_content,
            "entropy_content_only": self.entropy(dist_content),
            "entropy_norm_content_only": self.normalized_entropy(dist_content),
            "num_tokens": len(passage_positions),
            "token_distribution": token_dist,
            "token_char_spans": token_char_spans,
            "tok_entropy": self.entropy(token_dist),
            "tok_entropy_norm": self.normalized_entropy(token_dist),
            "token_distribution_no_punct": token_dist_no_punct,
            "num_tokens_no_punct": len(token_weights_no_punct),
            "tok_entropy_no_punct": self.entropy(token_dist_no_punct),
            "tok_entropy_norm_no_punct": self.normalized_entropy(token_dist_no_punct),
            "token_distribution_content_only": token_dist_content,
            "num_tokens_content_only": len(token_weights_content),
            "tok_entropy_content_only": self.entropy(token_dist_content),
            "tok_entropy_norm_content_only": self.normalized_entropy(token_dist_content),
            "layer": layer,
        }


class DecoderOnlyQAModel(QAModel):
    """Qwen/Llama-style: AutoModelForCausalLM.

    "Question attending to passage" only exists during PREFILL -- the one
    forward pass that encodes passage+question+"Answer:" before any
    generation happens (see module docstring). Query rows = question token
    positions, located by character offset against the prompt template
    (same approach as GenerativeQAModel -- causal LM tokenizers have no
    [SEP]/segment-ID concept either). Passage/question positions can never
    attend to anything that comes after them in the prompt (e.g. multiple-
    choice options, if present) -- causal masking guarantees it, so the
    entropy signal here is unaffected by whether options are included.

    predict_answer() DOES require a real `.generate()` call, same
    separation of concerns as GenerativeQAModel: unrelated to the entropy
    signal, only needed to produce answer text for correctness evaluation.

    Defaults to float32, not float16: float16 + eager attention silently
    produces NaN logits for at least Qwen2.5-1.5B-Instruct (confirmed on
    both MPS and CPU, so it's a numerics issue in the eager reference
    implementation at this model's scale, not an MPS-specific bug) --
    Qwen2.5-0.5B-Instruct happened not to hit it, which is how this went
    unnoticed until a 1.5B run produced letter=None on every single item.
    float32 does not reproduce the NaN at either size. Generation
    (predict_answer/predict_multiple_choice_answer) uses `sdpa`, which is
    faster and doesn't have this numerical issue at all -- attention
    extraction (get_attention_distribution/get_per_question_token_attention)
    switches to `eager` only for its one forward pass, via
    `set_attn_implementation`, since that's the only way to get
    output_attentions back at all (sdpa silently returns None for it).
    NOTE: attention extraction has only been verified NaN-free at
    float32 -- if you construct with precision="fp16"/"int8"/"int4" for a
    cascade experiment (answering only, via LocalDecoderAnswerer), treat
    get_attention_distribution/get_per_question_token_attention on that
    same instance as unverified for now.

    precision: "fp32" (default, safe everywhere) | "fp16" (sdpa-only path
    confirmed NaN-free for generation, but see the eager/attention caveat
    above) | "int8" | "int4" (both via bitsandbytes -- CUDA-only, will
    raise clearly if no CUDA device is available; useful for a precision-
    vs-accuracy cascade sweep on a real GPU, e.g. CSC Roihu, not on this
    project's Apple Silicon dev machines)."""

    def __init__(self, model_name: str, device: str | None = None, precision: str = "fp32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert precision in ("fp32", "fp16", "int8", "int4"), f"unknown precision {precision!r}"
        self.torch = torch
        self.model_name = model_name
        self.precision = precision
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs: dict = {"attn_implementation": "sdpa"}
        if precision == "fp32":
            load_kwargs["dtype"] = torch.float32
        elif precision == "fp16":
            load_kwargs["dtype"] = torch.float16
        else:
            assert torch.cuda.is_available(), (
                f"precision={precision!r} needs bitsandbytes + a CUDA GPU (e.g. CSC Roihu) -- "
                f"this device ({self.device}) has no CUDA support, so int8/int4 can't run here."
            )
            from transformers import BitsAndBytesConfig

            if precision == "int8":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if precision in ("fp32", "fp16"):  # quantized models are placed on-device by bitsandbytes itself
            self.model = self.model.to(self.device)
        self.model.eval()

    def _default_layer(self) -> int:
        return self.model.config.num_hidden_layers - 1

    def _templated_prompt(self, user_text: str, question: str, passage: str) -> tuple[str, int, int, int, int]:
        """Returns (full_text, q_start, q_end, p_start, p_end) -- character
        offsets of `question` and `passage` within the chat-templated
        prompt, located by substring search (both are unique enough in
        practice; same assumption GenerativeQAModel makes with its own
        fixed prefixes)."""
        full_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True,
        )
        q_start = full_text.find(question)
        p_start = full_text.find(passage)
        assert q_start != -1, "question text not found verbatim in the templated prompt"
        assert p_start != -1, "passage text not found verbatim in the templated prompt"
        return full_text, q_start, q_start + len(question), p_start, p_start + len(passage)

    def predict_answer(self, passage: str, question: str) -> tuple[str, float]:
        user_text = f"Passage: {passage}\n\nQuestion: {question}\nAnswer concisely."
        full_text, *_ = self._templated_prompt(user_text, question, passage)
        enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=64, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )

        new_tokens = out.sequences[0][enc["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # confidence: mean per-step max-token probability of the generated sequence
        step_probs = [self.torch.softmax(score[0], dim=-1).max().item() for score in out.scores]
        confidence = sum(step_probs) / len(step_probs) if step_probs else 0.0
        return answer, confidence

    def predict_multiple_choice_answer(self, passage: str, question: str,
                                        options: list[str]) -> tuple[str | None, float]:
        """Decoder-specific extra, not part of the shared QAModel interface
        (which only knows free-text QA via predict_answer): generates a
        single answer LETTER for a 4-option multiple-choice question, so a
        correctness/accuracy signal can sit alongside attention entropy
        without conflating the two. Returns (letter or None if the model's
        output didn't contain A-D, confidence)."""
        user_text = (
            f"Passage: {passage}\n\nQuestion: {question}\nOptions:\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n"
            "Answer with the letter only.\nAnswer:"
        )
        full_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True,
        )
        enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=4, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )

        new_tokens = out.sequences[0][enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        letter = next((c for c in text.upper() if c in "ABCD"), None)
        step_probs = [self.torch.softmax(s[0], dim=-1).max().item() for s in out.scores]
        confidence = sum(step_probs) / len(step_probs) if step_probs else 0.0
        return letter, confidence

    def get_attention_distribution(self, passage: str, question: str,
                                    layer: int | None = None, head: int | None = None) -> dict:
        """Same return shape/keys as ExtractiveQAModel/GenerativeQAModel so
        callers (including attention_inspection.py's inspect_attention)
        don't need to branch on paradigm.

        layer: None (default) uses the model's LAST layer -- unlike the
        other two subclasses, which hardcode 11 for a specific known
        12-layer architecture, decoder model depth varies a lot by size
        (Qwen2.5-0.5B vs. -1.5B have different layer counts), so this
        can't be a fixed constant. Pass an explicit int to override.
        head: None (default) averages across all attention heads in the
        chosen layer, same convention as the other two subclasses."""
        user_text = f"Passage: {passage}\n\nQuestion: {question}\nAnswer concisely."
        full_text, q_start, q_end, p_start, p_end = self._templated_prompt(user_text, question, passage)

        enc_full = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc_full.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc_full.items()}

        question_positions = [i for i, (s, e) in enumerate(offsets) if e > 0 and q_start <= s < q_end]
        passage_positions = [i for i, (s, e) in enumerate(offsets) if e > 0 and p_start <= s < p_end]
        sent_spans = self.sentence_spans(passage)
        sentence_texts = [passage[s:e].strip() for s, e in sent_spans]
        if not question_positions or not passage_positions:
            return {"sentences": sentence_texts, "distribution": [], "entropy": None}

        l = layer if layer is not None else self._default_layer()
        # sdpa (the default -- fast, used for generation) silently returns
        # None for output_attentions; eager is required to get real
        # attention weights back. Switch back to sdpa afterward so the
        # next predict_answer/predict_multiple_choice_answer call stays fast.
        self.model.set_attn_implementation("eager")
        with self.torch.no_grad():
            out = self.model(**enc, output_attentions=True)
        self.model.set_attn_implementation("sdpa")

        layer_heads = out.attentions[l][0].float().cpu()  # (heads, seq, seq)
        selected_heads = layer_heads.mean(dim=0) if head is None else layer_heads[head]
        sub = selected_heads[question_positions][:, passage_positions]
        per_passage_token = sub.mean(dim=0).tolist()
        token_dist = self.normalize(per_passage_token)

        token_char_spans = [(offsets[pos][0] - p_start, offsets[pos][1] - p_start) for pos in passage_positions]
        token_texts = [passage[s:e] for s, e in token_char_spans]
        is_punct = [self.is_punctuation_only(t) for t in token_texts]
        is_content = [self.is_content_word(t) for t in token_texts]

        sent_sum = [0.0] * len(sent_spans)
        sent_sum_no_punct = [0.0] * len(sent_spans)
        sent_sum_content = [0.0] * len(sent_spans)
        for tok_idx, (char_start, _) in enumerate(token_char_spans):
            for s_idx, (s_start, s_end) in enumerate(sent_spans):
                if s_start <= char_start < s_end:
                    sent_sum[s_idx] += per_passage_token[tok_idx]
                    if not is_punct[tok_idx]:
                        sent_sum_no_punct[s_idx] += per_passage_token[tok_idx]
                    if is_content[tok_idx]:
                        sent_sum_content[s_idx] += per_passage_token[tok_idx]
                    break

        dist = self.normalize(sent_sum)
        dist_no_punct = self.normalize(sent_sum_no_punct)
        dist_content = self.normalize(sent_sum_content)
        token_weights_no_punct = [w for w, p in zip(per_passage_token, is_punct) if not p]
        token_dist_no_punct = self.normalize(token_weights_no_punct)
        token_weights_content = [w for w, c in zip(per_passage_token, is_content) if c]
        token_dist_content = self.normalize(token_weights_content)
        return {
            "sentences": sentence_texts,
            "distribution": dist,
            "entropy": self.entropy(dist),
            "entropy_norm": self.normalized_entropy(dist),
            "distribution_no_punct": dist_no_punct,
            "entropy_no_punct": self.entropy(dist_no_punct),
            "entropy_norm_no_punct": self.normalized_entropy(dist_no_punct),
            "distribution_content_only": dist_content,
            "entropy_content_only": self.entropy(dist_content),
            "entropy_norm_content_only": self.normalized_entropy(dist_content),
            "num_tokens": len(passage_positions),
            "token_distribution": token_dist,
            "token_char_spans": token_char_spans,
            "tok_entropy": self.entropy(token_dist),
            "tok_entropy_norm": self.normalized_entropy(token_dist),
            "token_distribution_no_punct": token_dist_no_punct,
            "num_tokens_no_punct": len(token_weights_no_punct),
            "tok_entropy_no_punct": self.entropy(token_dist_no_punct),
            "tok_entropy_norm_no_punct": self.normalized_entropy(token_dist_no_punct),
            "token_distribution_content_only": token_dist_content,
            "num_tokens_content_only": len(token_weights_content),
            "tok_entropy_content_only": self.entropy(token_dist_content),
            "tok_entropy_norm_content_only": self.normalized_entropy(token_dist_content),
            "layer": l,
        }

    def get_per_question_token_attention(self, passage: str, question: str,
                                          layer: int | None = None, head: int | None = None) -> dict:
        """Like get_attention_distribution, but does NOT average across
        question tokens -- returns one passage-token distribution PER
        question token instead of one aggregate. Same purpose/convention
        as ExtractiveQAModel.get_per_question_token_attention."""
        user_text = f"Passage: {passage}\n\nQuestion: {question}\nAnswer concisely."
        full_text, q_start, q_end, p_start, p_end = self._templated_prompt(user_text, question, passage)

        enc_full = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc_full.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc_full.items()}

        question_positions = [i for i, (s, e) in enumerate(offsets) if e > 0 and q_start <= s < q_end]
        passage_positions = [i for i, (s, e) in enumerate(offsets) if e > 0 and p_start <= s < p_end]
        if not question_positions or not passage_positions:
            return {"question_tokens": [], "token_char_spans": [], "distributions": []}

        l = layer if layer is not None else self._default_layer()
        # sdpa (the default -- fast, used for generation) silently returns
        # None for output_attentions; eager is required to get real
        # attention weights back. Switch back to sdpa afterward so the
        # next predict_answer/predict_multiple_choice_answer call stays fast.
        self.model.set_attn_implementation("eager")
        with self.torch.no_grad():
            out = self.model(**enc, output_attentions=True)
        self.model.set_attn_implementation("sdpa")

        layer_heads = out.attentions[l][0].float().cpu()
        selected = layer_heads.mean(dim=0) if head is None else layer_heads[head]
        sub = selected[question_positions][:, passage_positions]

        passage_char_spans = [(offsets[pos][0] - p_start, offsets[pos][1] - p_start) for pos in passage_positions]
        is_punct = [self.is_punctuation_only(passage[s:e]) for s, e in passage_char_spans]
        keep_idx = [i for i, p in enumerate(is_punct) if not p]
        spans_no_punct = [passage_char_spans[i] for i in keep_idx]

        question_token_texts = [full_text[offsets[pos][0]:offsets[pos][1]] for pos in question_positions]
        distributions = [self.normalize([row[i] for i in keep_idx]) for row in sub.tolist()]

        return {
            "question_tokens": question_token_texts,
            "token_char_spans": spans_no_punct,
            "distributions": distributions,
        }
