from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from knowledge_graph.extractor import Triple
from ..models import Question

# CEFR-inspired 8-level scale
LEVELS = ["preA1", "A1", "A2", "B1", "B2", "C1", "C2", "C2+"]
LEVEL_ORDER = {level: i for i, level in enumerate(LEVELS)}

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

    Text-side combines three independent dimensions:
      1. Local extraction  — how hard it is to find/parse the answer in its sentence
                             (hop count, clause depth, passive, coreference)
      2. Global readability — how hard the passage is overall (LIX score)
      3. Distractor density — how many similar plausible wrong answers exist in the KG

    Combined score:  0.50 × local + 0.25 × readability + 0.25 × distractor

    Question-side combines: form complexity + passive grammar + lexical abstraction.

    Subclasses must implement the six abstract text/question sub-score methods.
    text_readability() and text_distractor() are optional overrides (default: 0.0).
    """

    text_max: int = 1       # max raw score for local extraction sub-scores
    question_max: int = 1

    def text_side(
        self,
        triple: Triple,
        hop_count: int = 1,
        passage: str = "",
        kg: Optional[Any] = None,
    ) -> str:
        local_raw = (
            self.text_structure(triple, hop_count)
            + self.text_grammar(triple)
            + self.text_lexical(triple)
        )
        local = local_raw / self.text_max
        readability = self.text_readability(passage) if passage else 0.0
        distractor = self.text_distractor(triple, kg) if kg is not None else 0.0
        combined = 0.50 * local + 0.25 * readability + 0.25 * distractor
        return _level(combined, _TEXT_THRESHOLDS)

    def question_side(self, question: Question) -> str:
        raw = (
            self.question_structure(question)
            + self.question_grammar(question)
            + self.question_lexical(question)
        )
        return _level(raw / self.question_max, _QUESTION_THRESHOLDS)

    # ── Text-side: local extraction (abstract) ────────────────────────────────

    @abstractmethod
    def text_structure(self, triple: Triple, hop_count: int) -> int:
        """Reasoning depth: hop count, source clause depth, answer tree depth."""

    @abstractmethod
    def text_grammar(self, triple: Triple) -> int:
        """Grammatical complexity of the source sentence: passive voice."""

    @abstractmethod
    def text_lexical(self, triple: Triple) -> int:
        """Semantic distance of the answer: coreference resolution required."""

    # ── Text-side: global + distractor (optional overrides) ──────────────────

    def text_readability(self, passage: str) -> float:
        """Global passage difficulty normalised to [0, 1]. Override to implement."""
        return 0.0

    def text_distractor(self, triple: Triple, kg: Any) -> float:
        """Distractor density normalised to [0, 1]. Override to implement."""
        return 0.0

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