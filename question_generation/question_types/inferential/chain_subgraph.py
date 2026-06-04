from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext, SKIP_VERB_BASES
from question_generation.models import Question
from question_generation.templates import relation_noun

_SUBJECT_ANCHORS = {"PERSON", "PER", "ORG"}
_OBJECT_ANCHORS = {"DATE", "TIME", "GPE", "LOC", "FAC"}

_OBJ_PREP = {"DATE": "in", "TIME": "in", "GPE": "in", "LOC": "at", "FAC": "at"}


class ChainSubgraphQuestion(QuestionType):
    tier = "inferential"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        if self.lang != "en":
            return []

        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        s_type = ctx.estimator.score_type("chain_subgraph")
        # Chain nominalization "the founder of X" has same vocab complexity as chain questions
        s_vocab = ctx.estimator.score_vocab(False, "chain")

        # Pass 1: build bridge → [(anchor, noun)] with correct direction
        #   Active:  bridge(src) → verb → anchor(dst)   e.g. Ollila → lead → Nokia
        #   Passive: anchor(src) → verb_by → bridge(dst) e.g. Nokia → found_by → Idestam
        bridge_to_anchors: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in SKIP_VERB_BASES:
                continue
            noun = relation_noun(self.lang, relation)
            if noun is None:
                continue
            src_type = kg.entity_type(src)
            dst_type = kg.entity_type(dst)
            if relation.endswith("_by") and src_type in _SUBJECT_ANCHORS and dst_type in {"PERSON", "PER", "ORG"}:
                bridge_to_anchors[dst].append((src, noun))
            elif src_type in {"PERSON", "PER", "ORG"} and dst_type in _SUBJECT_ANCHORS:
                bridge_to_anchors[src].append((dst, noun))

        # Pass 2: bridge → object-anchor edges
        # Only include if the anchor name appears in the source sentence — prevents
        # cross-sentence combinations like "acquirer of Mobira" paired with 1865 founding.
        group_events: dict[tuple[str, str, str], dict] = defaultdict(dict)
        group_obj_types: dict[tuple[str, str, str], str] = {}

        for src, dst, data in kg.edges:
            if src not in bridge_to_anchors:
                continue
            relation = data["relation"]
            if relation.split("_")[0] in SKIP_VERB_BASES:
                continue
            dst_type = kg.entity_type(dst)
            if dst_type not in _OBJECT_ANCHORS:
                continue
            source_text = data.get("source", "")
            if not source_text:
                continue
            for anchor, noun in bridge_to_anchors[src]:
                if anchor not in source_text:
                    continue
                key = (anchor, noun, dst)
                group_events[key][source_text] = True
                group_obj_types[key] = dst_type

        seen_texts: set = set()
        questions: list[Question] = []

        for (anchor, noun, obj), src_map in group_events.items():
            events = list(src_map.keys())
            if not (1 <= len(events) <= 2):
                continue
            obj_type = group_obj_types[(anchor, noun, obj)]
            prep = _OBJ_PREP.get(obj_type, "in")
            text = f"What did the {noun} of {anchor} do {prep} {obj}?"
            if text in seen_texts:
                continue
            seen_texts.add(text)

            n = len(events)
            s_local = 0.0 if n == 1 else 0.5
            difficulty = ctx.estimator.estimate(s_type, s_local, s_vocab, s_read)

            questions.append(Question(
                text=text, answer=events[0], answer_type=obj_type,
                difficulty=difficulty, lang=self.lang, source="",
                is_passive=False, hop_count=2, masked="chain_subgraph",
                answer_facts=events, tier="inferential",
                score_type=s_type, score_local=s_local,
                score_vocab=s_vocab, score_readability=s_read,
            ))

        return questions
