from __future__ import annotations
from abc import ABC, abstractmethod
from question_generation.models import Question


class QGMethod(ABC):
    """Base class for all question generation methods."""

    @abstractmethod
    def generate(
        self,
        text: str,
        anchor: str,
        cefr: str,
        lang: str,
        subgraph=None,
    ) -> list[Question]:
        """
        Generate questions.

        Args:
            text: Raw passage text (all methods receive this)
            anchor: Anchor entity name
            cefr: Target CEFR level (A1–C2)
            lang: Language code ("en" or "fi")
            subgraph: Optional KG subgraph (template/seq2seq/gnn use it; llm ignores it)
        """
        ...
