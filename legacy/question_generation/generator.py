from __future__ import annotations

from question_generation.methods.template.generator import TemplateMethod


class QuestionGenerator:
    """
    Orchestrator for question generation. Selects the generation method.
    Currently delegates to TemplateMethod. Future: seq2seq, gnn, llm methods.
    """

    def __init__(self, lang: str = "en", method: str = "template", cefr_readability=None) -> None:
        self.lang = lang
        self.method = method
        if method == "template":
            self._impl = TemplateMethod(lang=lang, cefr_readability=cefr_readability)
        else:
            raise ValueError(f"Unknown method: {method!r}. Available: 'template'")

    def generate(self, *args, **kwargs) -> list:
        return self._impl.generate(*args, **kwargs)
