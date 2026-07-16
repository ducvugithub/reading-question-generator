from __future__ import annotations

from knowledge_graph.extractor import Triple
from knowledge_graph.graph import KnowledgeGraph
from question_generation.methods.template.question_types import ALL_QUESTION_TYPES
from question_generation.methods.template.question_types.base import GenerationContext
from question_generation.difficulty import RuleBasedEstimator, LEVEL_ORDER


class TemplateMethod:
    """Template-based question generation using KG triples and rule-based question types."""

    def __init__(self, lang: str = "en", cefr_readability=None) -> None:
        self.lang = lang
        self._estimator = RuleBasedEstimator(cefr_readability=cefr_readability)
        self._handlers = [cls(lang) for cls in ALL_QUESTION_TYPES]

    def generate(
        self,
        triples: list[Triple],
        kg: KnowledgeGraph,
        num_questions: int = 10,
        passage: str = "",
        min_level: str = "preA1",
        max_level: str = "C2+",
        target_template_cefr: str = "B1",
        create_variants: bool = False,
    ) -> list:
        ctx = GenerationContext(
            kg=kg,
            verb_index=_build_verb_index(triples),
            passive_index=_build_passive_index(triples),
            surface_index=_build_surface_index(triples),
            triple_index=_build_triple_index(triples),
            case_index=_build_case_index(triples),
            passage=passage,
            lang=self.lang,
            estimator=self._estimator,
            min_level=min_level,
            max_level=max_level,
        )

        min_ord = LEVEL_ORDER.get(min_level, 0)
        max_ord = LEVEL_ORDER.get(max_level, len(LEVEL_ORDER) - 1)

        seen: set[str] = set()
        questions = []
        for handler in self._handlers:
            gen = handler.generate(ctx, target_template_cefr)
            for q in gen:
                if q.text not in seen:
                    if min_ord <= LEVEL_ORDER.get(q.difficulty, 0) <= max_ord:
                        questions.append(q)
                    seen.add(q.text)

        questions.sort(key=lambda q: LEVEL_ORDER.get(q.difficulty, 0), reverse=True)

        if create_variants:
            from question_generation.methods.template.variant_processor import expand_questions_with_variants
            questions = expand_questions_with_variants(questions, lang=self.lang)
            questions.sort(key=lambda q: LEVEL_ORDER.get(q.difficulty, 0), reverse=True)

        return questions[:num_questions]


def _build_verb_index(triples):
    return {(t.subject, t.relation, t.object): t.verb_text for t in triples if t.verb_text}

def _build_passive_index(triples):
    return {(t.subject, t.relation, t.object): t.is_passive for t in triples}

def _build_surface_index(triples):
    return {(t.subject, t.relation, t.object): t.object_surface for t in triples if t.object_surface}

def _build_triple_index(triples):
    index: dict = {}
    for t in triples:
        index.setdefault((t.subject, t.relation, t.object), t)
    return index

def _build_case_index(triples):
    return {(t.subject, t.relation, t.object): t.object_case for t in triples if t.object_case}
