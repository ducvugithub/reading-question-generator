from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext
from question_generation.models import Question
from question_generation.templates import build_which_question

_TYPE_NOUNS = {
    "ORG": "organization", "PERSON": "person", "PER": "person",
    "GPE": "place", "LOC": "place", "FAC": "facility", "WORK_OF_ART": "work",
}
_SKIP_VERB_BASES = {"become", "be"}


class WhichQuestion(QuestionType):
    tier = "retrieval"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        s_type = ctx.estimator.score_type("which")
        s_vocab = ctx.estimator.score_vocab(False, "which")

        by_source: dict = defaultdict(list)
        for src, dst, data in kg.edges:
            if data.get("source"):
                by_source[data["source"]].append((src, dst, data))

        entries: dict = defaultdict(lambda: defaultdict(list))
        for source_text, edges in by_source.items():
            for src, dst, data in edges:
                relation = data["relation"]
                parts = relation.split("_")
                verb_base = parts[0].replace("co-", "")
                if len(parts) > 1:
                    continue
                if verb_base in _SKIP_VERB_BASES:
                    continue
                dst_type = kg.entity_type(dst)
                if dst_type not in _TYPE_NOUNS:
                    continue
                hint = None
                for src2, dst2, data2 in edges:
                    if src2 != src:
                        continue
                    rel2 = data2["relation"]
                    parts2 = rel2.split("_")
                    if parts2[0].replace("co-", "") != verb_base or len(parts2) < 2:
                        continue
                    dst2_type = kg.entity_type(dst2)
                    if dst2_type not in ("DATE", "TIME", "GPE", "LOC"):
                        continue
                    prep2 = "_".join(parts2[1:])
                    if hint is None or dst2_type in ("DATE", "TIME"):
                        hint = (prep2, dst2)
                if hint:
                    entries[src][verb_base].append((dst, dst_type, hint[0], hint[1], source_text))

        questions = []
        for subj, verb_dict in entries.items():
            for verb_base, obj_list in verb_dict.items():
                if len(obj_list) < 2:
                    continue
                obj_types = {otype for _, otype, _, _, _ in obj_list}
                if len(obj_types) != 1:
                    continue
                type_noun = _TYPE_NOUNS.get(obj_types.pop())
                if not type_noun:
                    continue
                for obj, otype, hint_prep, hint_val, source_text in obj_list:
                    text = build_which_question(subj, verb_base, type_noun, hint_prep, hint_val, self.lang)
                    if text:
                        q = Question(
                            text=text, answer=obj, answer_type=otype,
                            difficulty=ctx.estimator.estimate(s_type, 0.1, s_vocab, s_read),
                            lang=self.lang, source=source_text,
                            is_passive=False, hop_count=1, masked="which",
                            tier="retrieval",
                            score_type=s_type, score_local=0.1,
                            score_vocab=s_vocab, score_readability=s_read,
                        )
                        questions.append(q)
        return questions
