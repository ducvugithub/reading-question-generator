from __future__ import annotations

from question_generation.question_types.base import QuestionType, GenerationContext, SKIP_VERB_BASES, SKIP_YESNO_TYPES
from question_generation.models import Question
from question_generation.templates import build_yesno_question


class YesNoQuestion(QuestionType):
    tier = "retrieval"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        questions = []

        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in SKIP_VERB_BASES:
                continue
            parts = relation.split("_", 1)
            has_prep = len(parts) > 1
            prep = parts[1] if has_prep else None
            is_p = ctx.passive_index.get((src, relation, dst), False)
            src_type = kg.entity_type(src)
            dst_type = kg.entity_type(dst)
            if src_type is None or dst_type is None:
                continue
            source = data.get("source", "")
            vt = ctx.verb_index.get((src, relation, dst), "")

            if is_p:
                if prep in ("during", "into"):
                    continue
            else:
                if has_prep and prep != "by":
                    continue
                if dst_type in SKIP_YESNO_TYPES:
                    continue

            text = build_yesno_question(src, relation, vt, dst, self.lang, is_passive=is_p)
            if text:
                triple = ctx.triple_index.get((src, relation, dst))
                s_type = ctx.estimator.score_type("yesno")
                s_local = ctx.estimator.score_local(triple) if triple else 0.1
                s_vocab = ctx.estimator.score_vocab(is_p, "yesno")
                q = Question(
                    text=text, answer="Yes", answer_type=dst_type,
                    difficulty=ctx.estimator.estimate(s_type, s_local, s_vocab, s_read),
                    lang=self.lang, source=source,
                    is_passive=is_p, hop_count=1, masked="yesno",
                    tier="retrieval",
                    score_type=s_type, score_local=s_local,
                    score_vocab=s_vocab, score_readability=s_read,
                )
                questions.append(q)
        return questions
