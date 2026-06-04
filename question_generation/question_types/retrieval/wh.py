from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext
from question_generation.question_types.base import SKIP_VERB_BASES, SKIP_MASK_SUBJECT_TYPES, SKIP_MASK_OBJECT_TYPES, LOC_TYPES
from question_generation.models import Question
from question_generation.templates import build_question, build_aggregation_question, TYPE_NOUNS_PLURAL


class WHQuestion(QuestionType):
    tier = "retrieval"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        kg = ctx.kg

        sibling_index: dict = defaultdict(list)
        for src2, dst2, data2 in kg.edges:
            rel2 = data2["relation"]
            if "_" in rel2:
                continue
            dst_type2 = kg.entity_type(dst2)
            if dst_type2 in TYPE_NOUNS_PLURAL:
                sibling_index[(src2, rel2, dst_type2)].append(dst2)

        aggregation_seen: set = set()
        s_read = ctx.estimator.score_readability(ctx.passage)
        questions = []

        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in SKIP_VERB_BASES:
                continue
            vt = ctx.verb_index.get((src, relation, dst), "")
            is_p = ctx.passive_index.get((src, relation, dst), False)
            source = data.get("source", "")
            src_type = kg.entity_type(src)
            dst_type = kg.entity_type(dst)
            if src_type is None or dst_type is None:
                continue
            triple = ctx.triple_index.get((src, relation, dst))
            s_local = ctx.estimator.score_local(triple) if triple else 0.1

            for mask, answer, answer_type, subject in (
                ("object",  dst, dst_type, src),
                ("subject", src, src_type, dst),
            ):
                if mask == "object" and dst_type in SKIP_MASK_OBJECT_TYPES:
                    continue
                if mask == "subject" and dst_type in SKIP_MASK_SUBJECT_TYPES:
                    continue
                if mask == "subject" and answer_type in LOC_TYPES and dst_type in LOC_TYPES:
                    continue
                if relation.endswith("_as"):
                    continue

                if mask == "object" and "_" not in relation:
                    sib_key = (src, relation, dst_type)
                    siblings = sibling_index.get(sib_key, [])
                    if len(siblings) >= 2:
                        if sib_key not in aggregation_seen:
                            aggregation_seen.add(sib_key)
                            type_plural = TYPE_NOUNS_PLURAL.get(dst_type)
                            if type_plural:
                                agg_text = build_aggregation_question(src, relation, type_plural, self.lang)
                                if agg_text:
                                    s_type = ctx.estimator.score_type("aggregation")
                                    s_vocab = ctx.estimator.score_vocab(False, "aggregation")
                                    q = Question(
                                        text=agg_text, answer=", ".join(siblings), answer_type=dst_type,
                                        difficulty=ctx.estimator.estimate(s_type, s_local, s_vocab, s_read),
                                        lang=self.lang, source=source,
                                        is_passive=False, hop_count=1, masked="aggregation",
                                        answer_list=siblings, tier="retrieval",
                                        score_type=s_type, score_local=s_local,
                                        score_vocab=s_vocab, score_readability=s_read,
                                    )
                                    questions.append(q)
                        continue

                subject_surface = ctx.surface_index.get((src, relation, dst), "") if mask == "subject" else ""
                text = build_question(
                    subject=subject, relation=relation, verb_text=vt,
                    answer_type=answer_type, mask=mask, lang=self.lang,
                    is_passive=is_p, subject_surface=subject_surface,
                )
                if text:
                    s_type = ctx.estimator.score_type(mask)
                    s_vocab = ctx.estimator.score_vocab(is_p, mask)
                    q = Question(
                        text=text, answer=answer, answer_type=answer_type,
                        difficulty=ctx.estimator.estimate(s_type, s_local, s_vocab, s_read),
                        lang=self.lang, source=source,
                        is_passive=is_p, hop_count=1, masked=mask,
                        tier="retrieval",
                        score_type=s_type, score_local=s_local,
                        score_vocab=s_vocab, score_readability=s_read,
                    )
                    questions.append(q)
        return questions
