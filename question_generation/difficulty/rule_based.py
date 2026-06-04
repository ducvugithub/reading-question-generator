from __future__ import annotations

from typing import Optional

from knowledge_graph.extractor import Triple
from .base import DifficultyEstimator

# ── Question-type base scores [0, 1] ─────────────────────────────────────────
_TYPE_SCORE: dict[str, float] = {
    "yesno":       0.20,
    "object":      0.20,
    "subject":     0.20,
    "comparison":  0.35,
    "which":       0.35,
    "aggregation": 0.30,
    "count":       0.30,
    "subgraph":       0.25,
    "chain_subgraph": 0.38,
    "chain":          0.55,
}

# Extra type score added for chain questions with > 2 hops
_HOP_BONUS: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.20, 4: 0.40}

# LIX normalisation: practical range 20 (very easy) → 65 (very hard)
_LIX_MIN = 20.0
_LIX_RANGE = 45.0


class RuleBasedEstimator(DifficultyEstimator):
    """
    Hand-crafted four-component difficulty estimator.

    score_type:        yesno(0.10) < wh-(0.20) < count/aggr(0.30) < anchor(0.25)
                       < comparison/which(0.35) < chain(0.55, +hop bonus)
    score_local:       average of four normalised sub-signals from the KG triple
    score_vocab:       passive inversion (0.5) + chain nominalization (0.5)
    score_readability: CEFR model if provided, else LIX normalised from [20, 65] → [0, 1]
    """

    def __init__(self, cefr_readability: Optional[object] = None) -> None:
        self._cefr = cefr_readability

    def score_type(self, masked: str, hop_count: int = 1) -> float:
        base = _TYPE_SCORE.get(masked, 0.20)
        hop = _HOP_BONUS.get(hop_count, 0.40) if masked == "chain" else 0.0
        return min(base + hop, 1.0)

    def score_local(self, triple: Triple) -> float:
        """Average of three normalised extraction-difficulty signals."""
        depth = min(triple.source_depth, 3) / 3               # [0, 1]
        answer = min(max(triple.answer_depth - 1, 0), 2) / 2  # [0, 1]
        coref = min(triple.coref_distance, 3) / 3              # [0, 1]
        return (depth + answer + coref) / 3

    def score_vocab(self, is_passive: bool, masked: str) -> float:
        return min(0.5 * int(is_passive) + 0.5 * int(masked == "chain"), 1.0)

    def score_readability(self, passage: str) -> float:
        if self._cefr is not None:
            return self._cefr.score(passage)
        # Fallback: LIX (Läsbarhetsindex) — language-agnostic, works for EN and FI
        sentences = [s.strip() for s in passage.split(".") if s.strip()]
        words = passage.split()
        if not sentences or not words:
            return 0.0
        long_words = sum(1 for w in words if len(w.rstrip(".,!?;:")) > 6)
        lix = len(words) / len(sentences) + long_words * 100 / len(words)
        return min(max((lix - _LIX_MIN) / _LIX_RANGE, 0.0), 1.0)