from __future__ import annotations

from abc import ABC, abstractmethod

from knowledge_graph.extractor import Triple
from ..models import Question

# CEFR-inspired 8-level scale
LEVELS = ["preA1", "A1", "A2", "B1", "B2", "C1", "C2", "C2+"]
LEVEL_ORDER = {level: i for i, level in enumerate(LEVELS)}

# Thresholds are float scores in [0, 1].
# Each side normalises its raw total by its own max (defined in the subclass),
# so both text_side and question_side output a comparable [0, 1] value.

_TEXT_THRESHOLDS = [
    (0.75, "C2+"),
    (0.55, "C2"),
    (0.40, "C1"),
    (0.25, "B2"),
    (0.20, "B1"),
    (0.12, "A2"),
    (0.04, "A1"),
    (0.00, "preA1"),
]

_QUESTION_THRESHOLDS = [
    (0.90, "C2"),   # chain passive (max score) lands here
    (0.70, "C1"),   # chain active
    (0.55, "B2"),
    (0.38, "B1"),   # wh- passive, comparison, which
    (0.30, "A2"),
    (0.10, "A1"),   # wh- active
    (0.00, "preA1"),
]


def _level(score: float, thresholds: list[tuple[float, str]]) -> str:
    for threshold, label in thresholds:
        if score >= threshold:
            return label
    return "preA1"


class DifficultyEstimator(ABC):
    """
    Estimates text-side and question-side difficulty on an 8-level CEFR-inspired scale.

    Each side sums three integer sub-scores (structure + grammar + lexical), then
    normalises the total by the subclass-defined max to produce a float in [0, 1].
    This ensures all three signals contribute proportionally to their natural range
    rather than on incompatible raw scales.

    Subclasses must define:
      text_max      — sum of maximum possible text-side sub-scores
      question_max  — sum of maximum possible question-side sub-scores
    """

    text_max: int = 1       # override in subclass
    question_max: int = 1   # override in subclass

    def text_side(self, triple: Triple, hop_count: int = 1) -> str:
        raw = (
            self.text_structure(triple, hop_count)
            + self.text_grammar(triple)
            + self.text_lexical(triple)
        )
        return _level(raw / self.text_max, _TEXT_THRESHOLDS)

    def question_side(self, question: Question) -> str:
        raw = (
            self.question_structure(question)
            + self.question_grammar(question)
            + self.question_lexical(question)
        )
        return _level(raw / self.question_max, _QUESTION_THRESHOLDS)

    # ── Text-side ─────────────────────────────────────────────────────────────

    @abstractmethod
    def text_structure(self, triple: Triple, hop_count: int) -> int:
        """Reasoning depth: hop count, source clause depth, answer tree depth."""

    @abstractmethod
    def text_grammar(self, triple: Triple) -> int:
        """Grammatical complexity of the source sentence: passive voice."""

    @abstractmethod
    def text_lexical(self, triple: Triple) -> int:
        """Semantic distance of the answer: coreference resolution required."""

    # ── Question-side ─────────────────────────────────────────────────────────

    @abstractmethod
    def question_structure(self, question: Question) -> int:
        """Form complexity: yes/no < wh- < comparison/which < chain."""

    @abstractmethod
    def question_grammar(self, question: Question) -> int:
        """Grammatical complexity of the question: passive voice inversion."""

    @abstractmethod
    def question_lexical(self, question: Question) -> int:
        """Lexical distance from source: nominalization in chain questions."""
