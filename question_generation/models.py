from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    text: str
    answer: str
    answer_type: Optional[str]
    difficulty: str              # combined CEFR level: preA1 … C2+
    lang: str
    is_passive: bool = False
    source: str = ""
    hop_count: int = 1
    masked: str = "object"       # "object", "subject", "true_claim", "false_claim",
                                 # "comparison", "which", "aggregation", "count",
                                 # "subgraph", "chain", "chain_subgraph", "bridge"
    chain_path: str = ""
    answer_list: list = field(default_factory=list)   # aggregation: all answer entities
    answer_facts: list = field(default_factory=list)  # anchor: all event sentences
    tier: str = "retrieval"
    # Component difficulty scores [0.0, 1.0] — for debugging and reporting
    score_type: float = 0.0
    score_local: float = 0.0
    score_vocab: float = 0.0
    score_readability: float = 0.0

    def __repr__(self) -> str:
        return (
            f"[{self.difficulty}] "
            f"{self.text!r}  →  {self.answer!r}"
        )