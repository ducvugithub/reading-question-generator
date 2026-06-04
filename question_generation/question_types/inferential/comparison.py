from __future__ import annotations

import re
from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext
from question_generation.models import Question
from question_generation.templates import build_comparison_question


class ComparisonQuestion(QuestionType):
    tier = "inferential"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        s_type = ctx.estimator.score_type("comparison")
        s_vocab = ctx.estimator.score_vocab(False, "comparison")
        # Comparison requires integrating two separate facts — moderate local difficulty
        s_local = 0.25

        def _year(s: str):
            m = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', s)
            return int(m.group(1)) if m else None

        by_source: dict = defaultdict(list)
        for src, dst, data in kg.edges:
            if data.get("source"):
                by_source[data["source"]].append((src, dst, data))

        by_verb: dict = defaultdict(list)
        for source_text, edges in by_source.items():
            for src, dst, data in edges:
                relation = data["relation"]
                parts = relation.split("_")
                if len(parts) > 1:
                    continue
                verb_base = parts[0].replace("co-", "")
                dst_type = kg.entity_type(dst)
                if dst_type not in ("ORG", "PERSON", "PER", "GPE", "LOC"):
                    continue
                year = None
                verb_past = verb_base
                for src2, dst2, data2 in edges:
                    if src2 != src:
                        continue
                    rel2 = data2["relation"]
                    parts2 = rel2.split("_")
                    if parts2[0].replace("co-", "") != verb_base or len(parts2) < 2:
                        continue
                    if kg.entity_type(dst2) not in ("DATE", "TIME"):
                        continue
                    y = _year(dst2)
                    if y:
                        year = y
                        verb_past = ctx.verb_index.get((src2, rel2, dst2), "") or verb_base
                        break
                if year is not None:
                    by_verb[(src, verb_base)].append((dst, dst_type, year, verb_past, source_text))

        questions = []
        seen_pairs: set = set()
        for (src, verb_base), entries in by_verb.items():
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    a_entity, a_type, a_year, a_verb, a_src = entries[i]
                    b_entity, b_type, b_year, b_verb, _ = entries[j]
                    if a_type != b_type or a_type is None:
                        continue
                    if a_year == b_year or a_entity == b_entity:
                        continue
                    pair = tuple(sorted([a_entity, b_entity]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    verb_past = a_verb or b_verb
                    for earlier in (True, False):
                        text = build_comparison_question(a_entity, b_entity, verb_past, self.lang, earlier)
                        if text:
                            answer = a_entity if (earlier == (a_year < b_year)) else b_entity
                            q = Question(
                                text=text, answer=answer, answer_type=a_type,
                                difficulty=ctx.estimator.estimate(s_type, s_local, s_vocab, s_read),
                                lang=self.lang, source=a_src,
                                is_passive=False, hop_count=1, masked="comparison",
                                tier="inferential",
                                score_type=s_type, score_local=s_local,
                                score_vocab=s_vocab, score_readability=s_read,
                            )
                            questions.append(q)
        return questions
