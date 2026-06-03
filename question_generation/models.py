from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    text: str
    answer: str
    answer_type: Optional[str]
    text_difficulty: str       # preA1 … C2+  — text-side signals
    question_difficulty: str   # preA1 … C2+  — question-side signals
    lang: str
    is_passive: bool = False
    source: str = ""
    hop_count: int = 1         # 1 = single edge, 2 = multi-hop chain
    masked: str = "object"     # "object", "subject", "chain", "yesno", "comparison", "which"
    chain_path: str = ""       # C-level only: "anchor →[rel1]→ bridge →[rel2]→ target"
    answer_list: list = field(default_factory=list)   # cat-1 aggregation: all entity answers
    answer_facts: list = field(default_factory=list)  # cat-2 anchor: all event sentences

    def __repr__(self) -> str:
        return (
            f"[T:{self.text_difficulty}/Q:{self.question_difficulty}] "
            f"{self.text!r}  →  {self.answer!r}"
        )
