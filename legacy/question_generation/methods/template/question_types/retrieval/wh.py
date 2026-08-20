from __future__ import annotations

from collections import defaultdict

from question_generation.methods.template.question_types.base import QuestionType, GenerationContext
from question_generation.methods.template.question_types.base import SKIP_VERB_BASES, SKIP_MASK_SUBJECT_TYPES, SKIP_MASK_OBJECT_TYPES, LOC_TYPES
from question_generation.models import Question
from question_generation.methods.template.templates import build_question, build_aggregation_question, build_single_variant, build_aggregation_variant, TYPE_NOUNS_PLURAL, question_word_by_case


class WHQuestion(QuestionType):
    tier = "retrieval"

    def generate(self, ctx: GenerationContext, target_cefr: str = "B1") -> list[Question]:
        kg = ctx.kg

        sibling_index: dict = defaultdict(list)
        subject_index: dict = defaultdict(list)
        for src2, dst2, data2 in kg.edges:
            rel2 = data2["relation"]
            # Include both active and passive relations for aggregation detection
            if "_" in rel2 and not rel2.endswith("_by"):
                continue
            dst_type2 = kg.entity_type(dst2)
            src_type2 = kg.entity_type(src2)
            if dst_type2 in TYPE_NOUNS_PLURAL:
                sibling_index[(src2, rel2, dst_type2)].append(dst2)
            if src_type2 in TYPE_NOUNS_PLURAL:
                subject_index[(dst2, rel2, src_type2)].append(src2)

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
                if mask == "subject" and "_" in relation and not relation.endswith("_by"):
                    continue

                # Check for multiple objects with same subject & relation
                object_answer_list = []
                if mask == "object":
                    sib_key = (src, relation, dst_type)
                    siblings = sibling_index.get(sib_key, [])
                    if len(siblings) >= 2:
                        if sib_key not in aggregation_seen:
                            aggregation_seen.add(sib_key)
                            # For passive relations (found_by), just populate answer_list
                            # For active relations (founded), try to create aggregation question
                            if "_" not in relation and dst_type in TYPE_NOUNS_PLURAL:
                                agg_result = build_aggregation_variant(src, relation, dst_type, self.lang, target_cefr)
                                if agg_result:
                                    agg_text, vocab_score = agg_result
                                    s_type = ctx.estimator.score_type("aggregation")
                                    q = Question(
                                        text=agg_text, answer=", ".join(siblings), answer_type=dst_type,
                                        difficulty=ctx.estimator.estimate(s_type, s_local, vocab_score, s_read),
                                        lang=self.lang, source=source,
                                        is_passive=False, hop_count=1, masked="aggregation",
                                        answer_list=siblings, tier="retrieval",
                                        score_type=s_type, score_local=s_local,
                                        score_vocab=vocab_score, score_readability=s_read,
                                    )
                                    questions.append(q)
                                    continue
                            # For passive relations, populate answer_list instead
                            object_answer_list = siblings

                # Check for multiple subjects with same relation & object
                subj_key = None
                subject_answer_list = []
                if mask == "subject" and "_" not in relation:
                    subj_key = (dst, relation, src_type)
                    subjects = subject_index.get(subj_key, [])
                    if len(subjects) >= 2:
                        if subj_key in aggregation_seen:
                            continue  # Already generated this question
                        aggregation_seen.add(subj_key)
                        subject_answer_list = subjects  # Populate answer_list with all subjects

                subject_surface = ctx.surface_index.get((src, relation, dst), "") if mask == "subject" else ""
                obj_case = ctx.case_index.get((src, relation, dst), "") if mask == "object" else ""

                if obj_case and mask == "object":
                    extracted_qw = question_word_by_case(self.lang, obj_case, answer_type)
                else:
                    from question_generation.methods.template.templates import question_word as get_question_word
                    extracted_qw = get_question_word(self.lang, answer_type)

                result = build_single_variant(
                    subject=subject, relation=relation, vt=vt,
                    qw=extracted_qw, mask=mask, lang=self.lang,
                    is_passive=is_p, subject_surface=subject_surface,
                    target_cefr=target_cefr,
                )
                if result:
                    text, vocab_score = result
                    s_type = ctx.estimator.score_type(mask)
                    # Use object_answer_list or subject_answer_list if available
                    final_answer_list = object_answer_list or subject_answer_list
                    q = Question(
                        text=text, answer=answer, answer_type=answer_type,
                        difficulty=ctx.estimator.estimate(s_type, s_local, vocab_score, s_read),
                        lang=self.lang, source=source,
                        is_passive=is_p, hop_count=1, masked=mask,
                        tier="retrieval",
                        answer_list=final_answer_list,  # Include multiple answers if applicable
                        score_type=s_type, score_local=s_local,
                        score_vocab=vocab_score, score_readability=s_read,
                    )
                    questions.append(q)
        return questions
