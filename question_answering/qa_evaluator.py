"""
Evaluate a QA model's predicted answer against a gold answer.

Pure text-scoring only -- no model loading or inference here; that stays
wherever it already lives (question_answering/scripts/run_qa_models.py,
question_difficulty's AttentionDispersionSignal, or ad-hoc notebook code).
This module answers one question: given a predicted answer span and a gold
answer, how correct is it?

Consolidates three metrics that were each duplicated elsewhere in the
project before this class existed:
  - QAPassRateSignal._is_correct (question_difficulty/methods/feature_based/
    difficulty_signals.py) -- now recall_overlap.
  - _token_f1 (question_difficulty/scripts/add_cascade_difficulty.py) --
    now token_f1.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "that", "this", "it", "its",
    "and", "or", "but", "not", "no", "what", "which", "who", "whom",
    "when", "where", "why", "how",
}


class QAEvaluator:
    """Scores a predicted answer against a gold answer, by three metrics:
      - exact_match: strict, case/whitespace-normalized string equality.
      - token_f1: precision/recall harmonic mean over raw token overlap
        (standard SQuAD-style F1) -- penalizes a verbose prediction that
        contains the gold words but pads them with irrelevant text.
      - recall_overlap: content-word (stopword-filtered) recall only, no
        precision term -- a verbose prediction that happens to contain
        every gold content word scores 1.0 regardless of length.
    """

    @staticmethod
    def exact_match(predicted: str, gold: str) -> bool:
        return predicted.strip().lower() == gold.strip().lower()

    @staticmethod
    def token_f1(predicted: str, gold: str) -> float:
        pred_tokens = re.sub(r"[^\w\s]", "", predicted.lower()).split()
        gold_tokens = re.sub(r"[^\w\s]", "", gold.lower()).split()
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def recall_overlap(predicted: str, gold: str) -> float:
        pred_words = QAEvaluator._content_words(predicted)
        gold_words = QAEvaluator._content_words(gold)
        if not gold_words:
            return 1.0 if QAEvaluator.exact_match(predicted, gold) else 0.0
        return len(pred_words & gold_words) / len(gold_words)

    @staticmethod
    def _content_words(text: str) -> set[str]:
        tokens = re.findall(r"[a-z']+", text.lower())
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}

    def is_correct(self, predicted: str, gold: str, metric: str = "token_f1",
                    threshold: float = 0.5) -> bool:
        """Convenience dispatcher. metric: "token_f1" | "recall_overlap" | "exact_match"."""
        if metric == "exact_match":
            return self.exact_match(predicted, gold)
        if metric == "recall_overlap":
            return self.recall_overlap(predicted, gold) >= threshold
        if metric == "token_f1":
            return self.token_f1(predicted, gold) >= threshold
        raise ValueError(f"Unknown metric: {metric!r}")
