from __future__ import annotations

from knowledge_graph.extractor import Triple
from ..models import Question
from .base import DifficultyEstimator

# ── Max raw scores (used by base class for normalisation) ─────────────────────
# text_max  = max_structure + max_grammar + max_lexical
#           = (18 + 3 + 2)  + 2           + 3           = 28
# question_max = max_structure + max_grammar + max_lexical
#              = 6             + 2           + 2           = 10

# Question form → base structure score
# Calibrated so that:
#   wh- active (2+0+0=2)     → 2/10=0.20 → A1
#   comparison/which (4+0+0) → 4/10=0.40 → B1
#   chain active (6+0+2=8)   → 8/10=0.80 → C1
#   chain passive (6+2+2=10) → 10/10=1.0 → C2
_FORM_SCORE: dict[str, int] = {
    "yesno":       0,   # binary, no retrieval needed
    "object":      2,   # direct wh- lookup
    "subject":     2,   # direct wh- lookup (subject reversed)
    "comparison":  4,   # must compare two dated entities
    "which":       4,   # must disambiguate among candidates
    "aggregation": 5,   # exhaustive scan — harder than single which, simpler than chain
    "chain":       6,   # nominalized multi-hop wh-
}

# hop_count → base structure score for text-side
# 2-hop simple → 6 → 6/28 = 0.21 → B1 baseline
# 3-hop simple → 12 → 12/28 = 0.43 → C1 baseline
_HOP_SCORE: dict[int, int] = {1: 0, 2: 6, 3: 12, 4: 18}


class RuleBasedEstimator(DifficultyEstimator):
    """
    Hand-crafted additive scoring normalised to [0, 1] by the base class.

    text_max = 28  (4-hop + depth=3 + answer=2 + passive + coref=3)
    question_max = 10  (chain + passive + nominalization)
    """

    text_max = 28
    question_max = 10

    # ── Text-side ─────────────────────────────────────────────────────────────

    def text_structure(self, triple: Triple, hop_count: int = 1) -> int:
        # hop: primary signal (0, 6, 12, 18 for 1–4 hops)
        base = _HOP_SCORE.get(hop_count, 18)
        depth_score = min(triple.source_depth, 3)                    # 0–3
        answer_score = min(max(triple.answer_depth - 1, 0), 2)       # 0 for ≤1, 1–2 beyond
        return base + depth_score + answer_score                      # max: 23

    def text_grammar(self, triple: Triple) -> int:
        return 2 if triple.is_passive else 0                         # 0 or 2

    def text_lexical(self, triple: Triple) -> int:
        return min(triple.coref_distance, 3)                         # 0–3

    # ── Question-side ─────────────────────────────────────────────────────────

    def question_structure(self, question: Question) -> int:
        return _FORM_SCORE.get(question.masked, 2)                   # 0–6

    def question_grammar(self, question: Question) -> int:
        return 2 if question.is_passive else 0                       # 0 or 2

    def question_lexical(self, question: Question) -> int:
        # Chain uses nominalized form ("the founder of X") → max abstraction
        # Passive adds a smaller form-distance (past participle + auxiliary)
        if question.masked == "chain":
            return 2
        if question.is_passive:
            return 1
        return 0                                                      # 0–2
