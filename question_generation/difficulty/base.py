from __future__ import annotations

from abc import ABC, abstractmethod

from knowledge_graph.extractor import Triple

LEVELS = ["preA1", "A1", "A2", "B1", "B2", "C1", "C2", "C2+"]
LEVEL_ORDER = {level: i for i, level in enumerate(LEVELS)}

# Combined score [0, 1] → CEFR level
# Thresholds are evenly spaced at 1/7 intervals, aligned to the ModernBERT CEFR scale:
#   s_read = idx/7 (e.g. B2 = 4/7 = 0.571), boundaries at midpoints between levels
_THRESHOLDS = [
    (6.5 / 7, "C2+"),   # 0.929
    (5.5 / 7, "C2"),    # 0.786
    (4.5 / 7, "C1"),    # 0.643
    (3.5 / 7, "B2"),    # 0.500
    (2.5 / 7, "B1"),    # 0.357
    (1.5 / 7, "A2"),    # 0.214
    (0.5 / 7, "A1"),    # 0.071
    (0.000,   "preA1"),
]


def _level(score: float) -> str:
    for threshold, label in _THRESHOLDS:
        if score >= threshold:
            return label
    return "preA1"


class DifficultyEstimator(ABC):
    """
    Four-component difficulty estimator mapping to an 8-level CEFR-inspired scale.

    Components (each [0, 1]):
      score_type        — question form complexity (yesno < wh < chain)
      score_local       — answer extraction difficulty (depth, passive, coref)
      score_vocab       — question phrasing complexity (passive, nominalization)
      score_readability — passage readability (LIX)

    Combined: 0.5×type + 0.3×local + 0.1×vocab + 0.1×readability → estimate()
    """

    @abstractmethod
    def score_type(self, masked: str, hop_count: int = 1) -> float:
        """Question form complexity [0, 1]."""

    @abstractmethod
    def score_local(self, triple: Triple) -> float:
        """Answer extraction difficulty [0, 1]: clause depth, passive, coreference."""

    @abstractmethod
    def score_vocab(self, is_passive: bool, masked: str) -> float:
        """Question phrasing complexity [0, 1]: passive inversion, nominalization."""

    @abstractmethod
    def score_readability(self, passage: str) -> float:
        """Passage readability [0, 1]: LIX normalised to [20, 65]."""

    def estimate(
        self, s_type: float, s_local: float, s_vocab: float, s_read: float
    ) -> str:
        combined = (s_type + s_local + s_vocab + s_read) / 4
        return _level(combined)