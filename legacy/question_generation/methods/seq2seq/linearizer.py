"""
Linearize a KG subgraph into a flat text string for T5 input.

Format: "generate question level=B1: Entity | relation | Entity . Entity | relation | Entity"

Example:
  triples = [Triple(subject="Microsoft", relation="founded_in", object="1975"), ...]
  linearize(triples, cefr="B1")
  → "generate question level=B1: Microsoft | founded_in | 1975 . Microsoft | founded_by | Bill Gates"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_graph.extractor import Triple


def linearize(triples: list["Triple"], cefr: str) -> str:
    """Serialize triples to flat text with CEFR control prefix.

    Args:
        triples: KG triples representing the answer subgraph.
        cefr: Target CEFR level for conditioning (e.g. "B1", "C1").

    Returns:
        Model input string ready for T5/FinT5.
    """
    parts = [f"{t.subject} | {t.relation} | {t.object}" for t in triples]
    return f"generate question level={cefr}: {' . '.join(parts)}"
