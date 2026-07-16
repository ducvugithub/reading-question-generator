from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext, SKIP_VERB_BASES
from question_generation.models import Question
from question_generation.templates import build_count_question, build_count_variant, TYPE_NOUNS_PLURAL


class CountQuestion(QuestionType):
    tier = "retrieval"

    def generate(self, ctx: GenerationContext, target_cefr: str = "B1") -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        s_type = ctx.estimator.score_type("count")
        s_vocab = ctx.estimator.score_vocab(False, "count")

        groups: dict = defaultdict(list)
        for src, dst, data in kg.edges:
            relation = data["relation"]
            if "_" in relation or relation.split("_")[0] in SKIP_VERB_BASES:
                continue
            dst_type = kg.entity_type(dst)
            if dst_type in TYPE_NOUNS_PLURAL:
                groups[(src, relation, dst_type)].append(dst)

        questions = []
        for (subject, relation, dst_type), objs in groups.items():
            if len(objs) < 2:
                continue
            if dst_type not in TYPE_NOUNS_PLURAL:
                continue
            result = build_count_variant(subject, relation, dst_type, self.lang, target_cefr)
            if not result:
                continue
            text, vocab_score = result
            triple = ctx.triple_index.get((subject, relation, objs[0]))
            s_local = ctx.estimator.score_local(triple) if triple else 0.1
            q = Question(
                text=text, answer=str(len(objs)), answer_type="CARDINAL",
                difficulty=ctx.estimator.estimate(s_type, s_local, vocab_score, s_read),
                lang=self.lang, source="",
                is_passive=False, hop_count=1, masked="count",
                tier="retrieval",
                score_type=s_type, score_local=s_local,
                score_vocab=vocab_score, score_readability=s_read,
            )
            questions.append(q)
        return questions
