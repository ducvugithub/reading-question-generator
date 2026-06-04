from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_graph.graph import KnowledgeGraph
    from question_generation.models import Question
    from question_generation.difficulty.base import DifficultyEstimator

# Shared filter sets used across question types
SKIP_VERB_BASES = {"become", "be"}
SKIP_MASK_SUBJECT_TYPES = {"DATE", "TIME", "MONEY", "CARDINAL", "PERCENT", "QUANTITY"}
SKIP_MASK_OBJECT_TYPES = {"MONEY", "CARDINAL", "PERCENT", "QUANTITY"}
LOC_TYPES = {"LOC", "GPE", "FAC"}
SKIP_YESNO_TYPES = {"DATE", "TIME", "LOC", "GPE", "FAC"}


@dataclass
class GenerationContext:
    kg: "KnowledgeGraph"
    verb_index: dict
    passive_index: dict
    surface_index: dict
    triple_index: dict
    passage: str
    lang: str
    estimator: "DifficultyEstimator"
    min_level: str = "preA1"   # inclusive lower bound for difficulty filtering
    max_level: str = "C2+"     # inclusive upper bound for difficulty filtering


class QuestionType(ABC):
    tier: str = "retrieval"

    def __init__(self, lang: str) -> None:
        self.lang = lang

    @abstractmethod
    def generate(self, ctx: GenerationContext) -> list["Question"]:
        pass
