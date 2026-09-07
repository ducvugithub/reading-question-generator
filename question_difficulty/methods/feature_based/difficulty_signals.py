"""
Per-question difficulty signals — see question_difficulty/docs/cognitive_difficulty_estimation.md
("Method 4") for the full plan and rationale. None of these use the RACE
subset-inherited label; they compute a signal purely from the individual
(passage, question, answer) triple.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "that", "this", "it", "its",
    "and", "or", "but", "not", "no", "what", "which", "who", "whom",
    "when", "where", "why", "how",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _content_words(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


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
    """Feeds (question, passage) into a SQuAD-finetuned QA model, extracts the
    question->passage attention sub-block per layer, and summarizes how
    concentrated vs. spread out it is — at both the TOKEN level and the
    SENTENCE level (two independent breakdowns of the same underlying
    per-passage-token attention vector).

    Returns features for EVERY layer plus an all-layer average, since there's
    no ground truth to pick a single "correct" layer in advance — layer
    selection happens empirically downstream, by checking which layer's
    values show the most within-passage variance (see validate_difficulty_signals.py).

    Token-level features (prefix "tok_"): computed directly on the normalized
    per-passage-token attention vector, before any sentence grouping.
      - tok_entropy / tok_max / tok_min
      - tok_top5 / tok_top10 / tok_top15: share of total attention captured by
        the top 5%/10%/15% of tokens (by count, scaled to passage length —
        NOT a fixed token count, which wouldn't compare across passage lengths)

    Sentence-level features (prefix "sent_"): tokens are grouped into
    sentences two ways before computing entropy/max/min on each:
      - "total": sum of attention across a sentence's tokens (favors longer
        sentences — more tokens to accumulate mass over)
      - "avg": that sum divided by the sentence's token count (length-
        normalized — a short sentence and a long one with the same per-word
        intensity score the same)
    """
    name = "attention_dispersion"

    TOP_PCTS = (5, 10, 15)

    def __init__(self, qa_model_name: str = "deepset/roberta-base-squad2", max_length: int = 512):
        from question_answering.qa_model import ExtractiveQAModel

        self.qa_model = ExtractiveQAModel(qa_model_name, max_length)

    def _top_pct_mass(self, dist: list[float], pct: int) -> float:
        n = max(1, round(len(dist) * pct / 100))
        return sum(sorted(dist, reverse=True)[:n])

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        enc = self.qa_model.tokenizer(
            question, passage,
            max_length=self.qa_model.max_length, truncation="only_second",
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)  # None=special, 0=question, 1=passage

        with self.qa_model.torch.no_grad():
            out = self.qa_model.model(**enc, output_attentions=True)

        sent_spans = self.qa_model.sentence_spans(passage)
        n_sents = len(sent_spans)

        question_positions = [i for i, s in enumerate(sequence_ids) if s == 0]
        passage_positions = [i for i, s in enumerate(sequence_ids) if s == 1]
        if not question_positions or not passage_positions:
            return {}

        features: dict[str, float] = {}
        num_layers = len(out.attentions)
        all_layer_token_dists: list[list[float]] = []
        all_layer_sent_total_dists: list[list[float]] = []
        all_layer_sent_avg_dists: list[list[float]] = []

        for layer_idx, layer_attn in enumerate(out.attentions):
            # layer_attn: (batch=1, num_heads, seq, seq) -> average over heads
            avg_heads = layer_attn[0].mean(dim=0)  # (seq, seq)
            # rows = question tokens, cols = passage tokens
            sub = avg_heads[question_positions][:, passage_positions]  # (n_q, n_p)
            per_passage_token = sub.mean(dim=0).tolist()  # (n_p,) — avg over question tokens

            # --- token-level ---
            token_dist = self.qa_model.normalize(per_passage_token)
            all_layer_token_dists.append(token_dist)
            self._add_token_features(features, token_dist, f"layer{layer_idx}")

            # --- sentence-level (two breakdowns of the same token vector) ---
            sent_sum = [0.0] * n_sents
            sent_token_count = [0] * n_sents
            for tok_idx, pos in enumerate(passage_positions):
                char_start, char_end = offsets[pos]
                if char_start == char_end:  # special token
                    continue
                for s_idx, (s_start, s_end) in enumerate(sent_spans):
                    if s_start <= char_start < s_end:
                        sent_sum[s_idx] += per_passage_token[tok_idx]
                        sent_token_count[s_idx] += 1
                        break

            sent_total_dist = self.qa_model.normalize(sent_sum)
            sent_avg_raw = [s / c if c else 0.0 for s, c in zip(sent_sum, sent_token_count)]
            sent_avg_dist = self.qa_model.normalize(sent_avg_raw)
            all_layer_sent_total_dists.append(sent_total_dist)
            all_layer_sent_avg_dists.append(sent_avg_dist)
            self._add_sentence_features(features, sent_total_dist, sent_avg_dist, f"layer{layer_idx}")

        # all-layers-averaged distributions, as one more candidate row
        def _macro_avg(dists: list[list[float]], length: int) -> list[float]:
            avg = [sum(d[i] for d in dists) / num_layers for i in range(length)]
            return self.qa_model.normalize(avg)

        token_alllayers = _macro_avg(all_layer_token_dists, len(all_layer_token_dists[0]))
        self._add_token_features(features, token_alllayers, "alllayers")

        sent_total_alllayers = _macro_avg(all_layer_sent_total_dists, n_sents)
        sent_avg_alllayers = _macro_avg(all_layer_sent_avg_dists, n_sents)
        self._add_sentence_features(features, sent_total_alllayers, sent_avg_alllayers, "alllayers")

        return features

    def get_sentence_distribution(self, passage: str, question: str, layer: int = 11) -> dict:
        """For visualization/manual review: delegates to
        ExtractiveQAModel.get_attention_distribution -- one layer, not the
        full multi-layer sweep compute() does. Used by
        question_difficulty/notebooks/ for eyeballing real examples, not
        for the production feature-extraction pipeline."""
        return self.qa_model.get_attention_distribution(passage, question, layer=layer)

    def get_all_layers_distribution(self, passage: str, question: str) -> dict:
        """Delegates to ExtractiveQAModel.get_all_layers_attention_distribution
        -- every layer in one forward pass. Used for the "compare all
        layers" view in question_difficulty/notebooks/."""
        return self.qa_model.get_all_layers_attention_distribution(passage, question)

    def _add_token_features(self, features: dict[str, float], dist: list[float], suffix: str) -> None:
        features[f"tok_entropy_{suffix}"] = self.qa_model.entropy(dist)
        features[f"tok_entropy_norm_{suffix}"] = self.qa_model.normalized_entropy(dist)
        features[f"tok_max_{suffix}"] = max(dist)
        features[f"tok_min_{suffix}"] = min(dist)
        for pct in self.TOP_PCTS:
            features[f"tok_top{pct}_{suffix}"] = self._top_pct_mass(dist, pct)

    def _add_sentence_features(self, features: dict[str, float], total_dist: list[float],
                                avg_dist: list[float], suffix: str) -> None:
        features[f"sent_total_entropy_{suffix}"] = self.qa_model.entropy(total_dist)
        features[f"sent_avg_entropy_{suffix}"] = self.qa_model.entropy(avg_dist)
        features[f"sent_total_entropy_norm_{suffix}"] = self.qa_model.normalized_entropy(total_dist)
        features[f"sent_avg_entropy_norm_{suffix}"] = self.qa_model.normalized_entropy(avg_dist)
        features[f"sent_total_max_{suffix}"] = max(total_dist)
        features[f"sent_avg_max_{suffix}"] = max(avg_dist)
        features[f"sent_total_min_{suffix}"] = min(total_dist)
        features[f"sent_avg_min_{suffix}"] = min(avg_dist)


class QAPassRateSignal(DifficultySignal):
    """Runs a battery of SQuAD-finetuned QA models against the gold answer;
    returns the fraction that answer correctly. Excludes non-finetuned models
    (e.g. microsoft/deberta-v3-base) -- see question_answering/docs/qa_model_battery.md.

    Correctness is QAEvaluator's recall_overlap metric (content-word,
    stopword-filtered recall against gold, threshold 0.5) -- see
    question_answering/qa_evaluator.py for why this lives there, not here."""
    name = "qa_pass_rate"

    _DEFAULT_MODELS = [
        "deepset/roberta-base-squad2",
        "google-bert/bert-base-uncased-finetuned-squad",
        "mrm8488/distilroberta-base-finetuned-squad",
    ]

    def __init__(self, qa_model_names: list[str] | None = None):
        import sys
        from transformers import pipeline

        from question_answering.qa_evaluator import QAEvaluator

        self._evaluator = QAEvaluator()
        self.pipelines = {}
        for name in (qa_model_names or self._DEFAULT_MODELS):
            try:
                self.pipelines[name] = pipeline("question-answering", model=name)
            except Exception as e:
                print(f"  [!] Skipping {name}, failed to load: {e}", file=sys.stderr)
        if not self.pipelines:
            raise RuntimeError("No QA models could be loaded")

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        n_correct = 0
        for name, pipe in self.pipelines.items():
            try:
                result = pipe(question=question, context=passage)
                if self._evaluator.is_correct(result["answer"], answer, metric="recall_overlap"):
                    n_correct += 1
            except Exception:
                continue
        return {"qa_pass_rate": n_correct / len(self.pipelines)}


class AnswerExtractivenessSignal(DifficultySignal):
    """Plain text overlap between the gold answer and the passage. No model."""
    name = "answer_extractiveness"

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        answer_words = _content_words(_tokenize(answer))
        passage_words = _content_words(_tokenize(passage))
        overlap = len(answer_words & passage_words) / len(answer_words) if answer_words else 0.0
        return {"answer_extractiveness_overlap": overlap}


class QuestionAnswerSimilaritySignal(DifficultySignal):
    """Plain text/lexical similarity between question and answer. No model.

    Uses the overlap coefficient (|A∩B| / min(|A|,|B|)), not Jaccard
    (|A∩B| / |A∪B|) -- questions and answers are usually very different
    lengths (a question might have 10-15 content words, an answer just 2-3),
    and Jaccard's union-based denominator gets dominated by the longer set's
    extra vocabulary, understating similarity even when the shorter set is
    fully contained in the longer one. Overlap coefficient normalizes by the
    smaller set instead, so full containment always scores 1.0."""
    name = "question_answer_similarity"

    def compute(self, passage: str, question: str, answer: str) -> dict[str, float]:
        q_words = _content_words(_tokenize(question))
        a_words = _content_words(_tokenize(answer))
        if not q_words or not a_words:
            return {"question_answer_overlap_coef": 0.0}
        overlap_coef = len(q_words & a_words) / min(len(q_words), len(a_words))
        return {"question_answer_overlap_coef": overlap_coef}


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
