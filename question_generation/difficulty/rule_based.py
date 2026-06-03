from __future__ import annotations

import networkx as nx

from knowledge_graph.extractor import Triple
from ..models import Question
from .base import DifficultyEstimator

# ── Max raw scores (used by base class for normalisation) ─────────────────────
# text_max  = max_structure + max_grammar + max_lexical
#           = (18 + 3 + 2)  + 2           + 3           = 28
# question_max = max_structure + max_grammar + max_lexical
#              = 6             + 2           + 2           = 10

_FORM_SCORE: dict[str, int] = {
    "yesno":       0,
    "object":      2,
    "subject":     2,
    "comparison":  4,
    "which":       4,
    "aggregation": 5,
    "anchor":      5,
    "chain":       6,
}

_HOP_SCORE: dict[int, int] = {1: 0, 2: 6, 3: 12, 4: 18}

# LIX normalisation: practical range 20 (very easy) → 65 (very hard)
_LIX_MIN = 20.0
_LIX_RANGE = 45.0  # 65 - 20



class RuleBasedEstimator(DifficultyEstimator):
    """
    Hand-crafted additive scoring with three text-side dimensions:

      1. Local extraction  (normalised by text_max=28)
      2. Global readability via LIX formula — language-agnostic, works for EN + FI
      3. Distractor density — sibling edges + same-type entity count in the KG

    Combined text score: 0.50 × local + 0.25 × readability + 0.25 × distractor

    question_max = 10  (chain + passive + nominalization)
    """

    text_max = 28
    question_max = 10

    # ── Text-side: local extraction ───────────────────────────────────────────

    def text_structure(self, triple: Triple, hop_count: int = 1) -> int:
        base = _HOP_SCORE.get(hop_count, 18)
        depth_score = min(triple.source_depth, 3)
        answer_score = min(max(triple.answer_depth - 1, 0), 2)
        return base + depth_score + answer_score          # max: 23

    def text_grammar(self, triple: Triple) -> int:
        return 2 if triple.is_passive else 0              # 0 or 2

    def text_lexical(self, triple: Triple) -> int:
        return min(triple.coref_distance, 3)              # 0–3

    # ── Text-side: global readability (LIX) ──────────────────────────────────

    def text_readability(self, passage: str) -> float:
        """
        LIX (Läsbarhetsindex) — language-agnostic readability formula.
        Works for both English and Finnish.

        LIX = words/sentences + long_words*100/words
        Normalised from [20, 65] → [0, 1].
        """
        sentences = [s.strip() for s in passage.split(".") if s.strip()]
        words = passage.split()
        if not sentences or not words:
            return 0.0
        long_words = sum(1 for w in words if len(w.rstrip(".,!?;:")) > 6)
        lix = len(words) / len(sentences) + long_words * 100 / len(words)
        return min(max((lix - _LIX_MIN) / _LIX_RANGE, 0.0), 1.0)

    # ── Text-side: distractor density ─────────────────────────────────────────

    def text_distractor(self, triple: Triple, kg) -> float:
        """
        Counts two types of distractors in the KG:

        Siblings  — other objects sharing the same (subject, verb_base):
                    "Nokia acquired Mobira" and "Nokia acquired Alcatel-Lucent"
                    → answering "What did Nokia acquire?" has 1 sibling distractor.
                    Weighted ×2 because they are direct competing answers.

        Same-type — other entities of the same NER type as the answer,
                    weighted by graph distance using exponential decay: 0.5^dist.
                    Close same-type entities (dist=1: 0.5, dist=2: 0.25, dist=3: 0.125 …)
                    contribute strongly; distant unrelated ones contribute near-zero.
                    Cutoff at 6 hops (weight < 0.03 beyond that).

        Normalised via raw / (1 + raw) — maps [0, ∞) → [0, 1) without a hard cap,
        so hub-heavy passages don't prematurely saturate at 1.0.
        """
        answer_type = kg.entity_type(triple.object)

        same_type_score = 0.0
        if answer_type and triple.object in kg._g:
            lengths = nx.single_source_shortest_path_length(
                kg._g.to_undirected(), triple.object, cutoff=6
            )
            same_type_score = sum(
                0.5 ** dist
                for node, dist in lengths.items()
                if dist > 0 and kg.entity_type(node) == answer_type
            )

        return same_type_score / (1 + same_type_score)

    # ── Question-side ─────────────────────────────────────────────────────────

    def question_structure(self, question: Question) -> int:
        return _FORM_SCORE.get(question.masked, 2)        # 0–6

    def question_grammar(self, question: Question) -> int:
        return 2 if question.is_passive else 0            # 0 or 2

    def question_lexical(self, question: Question) -> int:
        if question.masked == "chain":
            return 2
        if question.is_passive:
            return 1
        return 0                                          # 0–2