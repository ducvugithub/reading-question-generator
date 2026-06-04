from __future__ import annotations

from question_generation.difficulty.base import LEVEL_ORDER, LEVELS

_MODEL_ID = "AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2"
_LEVEL_MAX = len(LEVELS) - 1  # 7


class CefrReadability:
    """
    Wraps the ModernBERT CEFR text classifier as a drop-in score_readability source.

    Returns a float [0, 1] where 0 = A1 and 1 = C2 (our scale extended to C2+).
    Result is cached per passage string so the model runs at most once per text.
    """

    def __init__(self) -> None:
        import sys, torch
        # torch.compile raises on Python 3.12 + PyTorch < 2.3 — patch to no-op
        if sys.version_info >= (3, 12) and tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 3):
            torch.compile = lambda fn=None, **kwargs: (fn if fn is not None else lambda f: f)
        from transformers import pipeline
        self._pipe = pipeline("text-classification", model=_MODEL_ID)
        self._cache: dict[str, float] = {}

    def score(self, passage: str) -> float:
        if passage not in self._cache:
            result = self._pipe(passage, truncation=True, max_length=512)
            label = result[0]["label"]       # e.g. "B1"
            idx = LEVEL_ORDER.get(label, 0)  # 0–7 on our scale
            self._cache[passage] = idx / _LEVEL_MAX
        return self._cache[passage]