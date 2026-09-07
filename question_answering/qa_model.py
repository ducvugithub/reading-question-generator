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

Both subclasses share entropy math (pure math, paradigm-independent) via the
QAModel base class. Only attention extraction and answer decoding differ.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

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

    def __init__(self, model_name: str, max_length: int = 512):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
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

    def get_attention_distribution(self, passage: str, question: str, layer: int = 11) -> dict:
        """One layer's sentence-level attention distribution, for
        visualization/manual review -- not the full multi-layer sweep
        AttentionDispersionSignal.compute() does for feature extraction."""
        enc = self.tokenizer(
            question, passage, max_length=self.max_length,
            truncation="only_second", return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)

        with self.torch.no_grad():
            out = self.model(**enc, output_attentions=True)

        sent_spans = self.sentence_spans(passage)
        sentence_texts = [passage[s:e].strip() for s, e in sent_spans]

        question_positions = [i for i, s in enumerate(sequence_ids) if s == 0]
        passage_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not question_positions or not passage_positions:
            return {"sentences": sentence_texts, "distribution": [], "entropy": None}

        avg_heads = out.attentions[layer][0].mean(dim=0)
        sub = avg_heads[question_positions][:, passage_positions]
        per_passage_token = sub.mean(dim=0).tolist()

        token_dist = self.normalize(per_passage_token)

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
        return {
            "sentences": sentence_texts,
            "distribution": dist,
            "entropy": self.entropy(dist),
            "entropy_norm": self.normalized_entropy(dist),
            "num_tokens": len(passage_positions),
            "tok_entropy": self.entropy(token_dist),
            "tok_entropy_norm": self.normalized_entropy(token_dist),
            "layer": layer,
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

    def get_attention_distribution(self, passage: str, question: str, layer: int = 11) -> dict:
        """Encoder-only self-attention, question-rows x passage-cols, one
        layer -- same shape/keys as ExtractiveQAModel.get_attention_distribution
        so callers don't need to branch on paradigm."""
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

        avg_heads = encoder_out.attentions[layer][0].mean(dim=0)
        sub = avg_heads[question_positions][:, passage_positions]
        per_passage_token = sub.mean(dim=0).tolist()
        token_dist = self.normalize(per_passage_token)

        sent_sum = [0.0] * len(sent_spans)
        for tok_idx, pos in enumerate(passage_positions):
            char_start, char_end = offsets[pos]
            passage_local_start = char_start - passage_char_start
            for s_idx, (s_start, s_end) in enumerate(sent_spans):
                if s_start <= passage_local_start < s_end:
                    sent_sum[s_idx] += per_passage_token[tok_idx]
                    break

        dist = self.normalize(sent_sum)
        return {
            "sentences": sentence_texts,
            "distribution": dist,
            "entropy": self.entropy(dist),
            "entropy_norm": self.normalized_entropy(dist),
            "num_tokens": len(passage_positions),
            "tok_entropy": self.entropy(token_dist),
            "tok_entropy_norm": self.normalized_entropy(token_dist),
            "layer": layer,
        }
