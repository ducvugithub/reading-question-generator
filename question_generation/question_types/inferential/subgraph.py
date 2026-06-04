from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext, SKIP_VERB_BASES
from question_generation.models import Question
from question_generation.templates import build_anchor_question, build_multi_anchor_question

_SUBJECT_ANCHORS = {"PERSON", "PER", "ORG"}
_OBJECT_ANCHORS = {"DATE", "TIME", "GPE", "LOC", "FAC"}

# Rule-based: formula gives wrong results for subgraph (event count drives difficulty)
_EVENT_DIFFICULTY = {1: "A1", 2: "A2"}
_EVENT_LOCAL = {1: 0.0, 2: 0.5}


class SubgraphQuestion(QuestionType):
    tier = "inferential"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        s_type = ctx.estimator.score_type("subgraph")
        s_vocab = ctx.estimator.score_vocab(False, "subgraph")

        seen_texts: set = set()
        questions: list[Question] = []

        # ── Single-anchor: group edges by anchor node ─────────────────────────
        anchor_events: dict = defaultdict(dict)
        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in SKIP_VERB_BASES or relation.endswith("_as"):
                continue
            source_text = data.get("source", "")
            if not source_text:
                continue
            if kg.entity_type(src) in _SUBJECT_ANCHORS:
                anchor_events[src][source_text] = True
            if kg.entity_type(dst) in _OBJECT_ANCHORS:
                anchor_events[dst][source_text] = True

        for anchor, src_map in anchor_events.items():
            events = list(src_map.keys())
            if not (1 <= len(events) <= 2):
                continue
            anchor_type = kg.entity_type(anchor)
            text = build_anchor_question(anchor, anchor_type, self.lang)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            n = len(events)
            questions.append(Question(
                text=text, answer=events[0], answer_type=anchor_type,
                difficulty=_EVENT_DIFFICULTY[n], lang=self.lang, source="",
                is_passive=False, hop_count=1, masked="subgraph",
                answer_facts=events, tier="retrieval" if n == 1 else "inferential",
                score_type=s_type, score_local=_EVENT_LOCAL[n],
                score_vocab=s_vocab, score_readability=s_read,
            ))

        # ── Paired-anchor: subject anchor + object anchor ─────────────────────
        pair_events: dict[tuple[str, str], dict] = defaultdict(dict)
        pair_types: dict[tuple[str, str], tuple[str, str]] = {}

        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in SKIP_VERB_BASES or relation.endswith("_as"):
                continue
            source_text = data.get("source", "")
            if not source_text:
                continue
            src_type = kg.entity_type(src)
            dst_type = kg.entity_type(dst)
            if src_type in _SUBJECT_ANCHORS and dst_type in _OBJECT_ANCHORS:
                key = (src, dst)
                pair_events[key][source_text] = True
                pair_types[key] = (src_type, dst_type)

        for (subj, obj), src_map in pair_events.items():
            events = list(src_map.keys())
            if not (1 <= len(events) <= 2):
                continue
            subj_type, obj_type = pair_types[(subj, obj)]
            text = build_multi_anchor_question(subj, subj_type, obj, obj_type, self.lang)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            n = len(events)
            questions.append(Question(
                text=text, answer=events[0], answer_type=obj_type,
                difficulty=_EVENT_DIFFICULTY[n], lang=self.lang, source="",
                is_passive=False, hop_count=1, masked="subgraph",
                answer_facts=events, tier="retrieval" if n == 1 else "inferential",
                score_type=s_type, score_local=_EVENT_LOCAL[n],
                score_vocab=s_vocab, score_readability=s_read,
            ))

        return questions
