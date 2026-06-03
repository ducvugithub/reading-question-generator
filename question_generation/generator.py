from __future__ import annotations

from knowledge_graph.extractor import Triple
from knowledge_graph.graph import KnowledgeGraph
from .models import Question
from .templates import (
    build_question, build_chain_question,
    build_yesno_question, build_comparison_question, build_which_question,
    build_aggregation_question, TYPE_NOUNS_PLURAL,
)
from .difficulty import RuleBasedEstimator, LEVEL_ORDER


class QuestionGenerator:
    """
    Generate questions from a KnowledgeGraph + the source triples.

    Single-edge questions: mask the object or subject of each edge.
    Multi-hop questions: chain two edges via a bridge entity.
    """

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._estimator = RuleBasedEstimator()

    def generate(self, triples: list[Triple], kg: KnowledgeGraph, num_questions: int = 10) -> list[Question]:
        verb_index = _build_verb_index(triples)
        passive_index = _build_passive_index(triples)
        surface_index = _build_surface_index(triples)
        triple_index = _build_triple_index(triples)

        questions: list[Question] = []
        seen: set[str] = set()

        all_candidates = (
            self._multihop(kg, verb_index, passive_index, triple_index)
            + self._aggregation(kg, triple_index)
            + self._which(kg, verb_index)
            + self._comparison(kg, verb_index)
            + self._single_edge(kg, verb_index, passive_index, surface_index, triple_index)
            + self._yes_no(kg, verb_index, passive_index, triple_index)
        )
        for q in all_candidates:
            if q.text not in seen:
                questions.append(q)
                seen.add(q.text)

        questions.sort(
            key=lambda q: (
                LEVEL_ORDER.get(q.text_difficulty, 0),
                LEVEL_ORDER.get(q.question_difficulty, 0),
            ),
            reverse=True,
        )
        return questions[:num_questions]

    # ── Single-edge ───────────────────────────────────────────────────────────

    def _single_edge(
        self, kg: KnowledgeGraph, verb_index: dict, passive_index: dict,
        surface_index: dict, triple_index: dict,
    ) -> list[Question]:
        questions = []
        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in _SKIP_VERB_BASES:
                continue
            vt = verb_index.get((src, relation, dst), "")
            is_p = passive_index.get((src, relation, dst), False)
            source = data.get("source", "")
            src_type = kg.entity_type(src)
            dst_type = kg.entity_type(dst)
            if src_type is None or dst_type is None:
                continue
            triple = triple_index.get((src, relation, dst))

            for mask, answer, answer_type, subject in (
                ("object",  dst, dst_type, src),
                ("subject", src, src_type, dst),
            ):
                if mask == "subject" and dst_type in _SKIP_MASK_SUBJECT_TYPES:
                    continue
                if mask == "subject" and answer_type in _LOC_TYPES and dst_type in _LOC_TYPES:
                    continue
                # "_as" obliques mark a role context ("replaced X as CEO of Nokia"),
                # not a direct object — both mask directions produce misleading questions.
                if relation.endswith("_as"):
                    continue
                subject_surface = surface_index.get((src, relation, dst), "") if mask == "subject" else ""
                text = build_question(
                    subject=subject, relation=relation, verb_text=vt,
                    answer_type=answer_type, mask=mask, lang=self.lang,
                    is_passive=is_p, subject_surface=subject_surface,
                )
                if text:
                    text_diff = self._estimator.text_side(triple, hop_count=1) if triple else "A"
                    q = Question(
                        text=text, answer=answer, answer_type=answer_type,
                        text_difficulty=text_diff, question_difficulty="preA1",
                        lang=self.lang, source=source,
                        is_passive=is_p, hop_count=1, masked=mask,
                    )
                    q.question_difficulty = self._estimator.question_side(q)
                    questions.append(q)
        return questions

    # ── Multi-hop ─────────────────────────────────────────────────────────────

    def _multihop(self, kg: KnowledgeGraph, verb_index: dict, passive_index: dict, triple_index: dict) -> list[Question]:
        questions = []
        for node, _ in kg.nodes:
            for path in kg.multihop_paths(node, max_hops=2):
                if len(path) < 2:
                    continue
                anchor, rel1, bridge = path[0]
                _,      rel2, target = path[1]

                # Bridge must be a named person: "the {noun} of {bridge}" only makes
                # sense when bridge is the institution and the second hop describes
                # what that person (anchor) actually did independently — if bridge is
                # an ORG, rel2 describes the ORG's actions, not the anchor person's.
                bridge_type = kg.entity_type(bridge)
                if bridge_type not in {"PERSON", "PER"}:
                    continue

                # Target must be a real named entity, not a vague noun ("mill", "part")
                target_type = kg.entity_type(target)
                if target_type is None:
                    continue

                vt2 = verb_index.get((bridge, rel2, target), "")
                is_p2 = passive_index.get((bridge, rel2, target), False)
                source = ""
                for u, v, data in kg.edges:
                    if u == bridge and v == target and data["relation"] == rel2:
                        source = data.get("source", "")
                        break

                text = build_chain_question(
                    anchor=anchor, rel1=rel1, bridge=bridge, rel2=rel2,
                    verb2_text=vt2, answer_type=target_type,
                    lang=self.lang, is_passive2=is_p2,
                )
                if text:
                    triple = triple_index.get((bridge, rel2, target))
                    text_diff = self._estimator.text_side(triple, hop_count=2) if triple else "B1"
                    q = Question(
                        text=text, answer=target, answer_type=target_type,
                        text_difficulty=text_diff, question_difficulty="preA1",
                        lang=self.lang, source=source,
                        is_passive=is_p2, hop_count=2, masked="chain",
                        chain_path=f"{anchor} →[{rel1}]→ {bridge} →[{rel2}]→ {target}",
                    )
                    q.question_difficulty = self._estimator.question_side(q)
                    questions.append(q)
        return questions

    # ── Yes / No ──────────────────────────────────────────────────────────────

    def _yes_no(
        self, kg: KnowledgeGraph, verb_index: dict, passive_index: dict,
        triple_index: dict,
    ) -> list[Question]:
        questions = []
        for src, dst, data in kg.edges:
            relation = data["relation"]
            if relation.split("_")[0] in _SKIP_VERB_BASES:
                continue
            parts = relation.split("_", 1)
            has_prep = len(parts) > 1
            prep = parts[1] if has_prep else None
            is_p = passive_index.get((src, relation, dst), False)
            src_type = kg.entity_type(src)
            dst_type = kg.entity_type(dst)
            if src_type is None or dst_type is None:
                continue
            source = data.get("source", "")
            vt = verb_index.get((src, relation, dst), "")

            if is_p:
                if prep in ("during", "into"):
                    continue
            else:
                if has_prep and prep != "by":
                    continue
                if dst_type in _SKIP_YESNO_TYPES:
                    continue

            text = build_yesno_question(src, relation, vt, dst, self.lang, is_passive=is_p)
            if text:
                triple = triple_index.get((src, relation, dst))
                text_diff = self._estimator.text_side(triple, hop_count=1) if triple else "A"
                q = Question(
                    text=text, answer="Yes", answer_type=dst_type,
                    text_difficulty=text_diff, question_difficulty="preA1",
                    lang=self.lang, source=source,
                    is_passive=is_p, hop_count=1, masked="yesno",
                )
                q.question_difficulty = self._estimator.question_side(q)
                questions.append(q)
        return questions

    # ── Comparison ────────────────────────────────────────────────────────────

    def _comparison(self, kg: KnowledgeGraph, verb_index: dict) -> list[Question]:
        import re
        from collections import defaultdict

        def _year(s: str):
            m = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', s)
            return int(m.group(1)) if m else None

        by_verb: dict = defaultdict(list)
        by_source: dict = defaultdict(list)
        for src, dst, data in kg.edges:
            if data.get("source"):
                by_source[data["source"]].append((src, dst, data))

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
                        verb_past = verb_index.get((src2, rel2, dst2), "") or verb_base
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
                                text_difficulty="B1", question_difficulty="preA1",
                                lang=self.lang, source=a_src,
                                is_passive=False, hop_count=1, masked="comparison",
                            )
                            q.question_difficulty = self._estimator.question_side(q)
                            questions.append(q)
        return questions

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregation(self, kg: KnowledgeGraph, triple_index: dict) -> list[Question]:
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for src, dst, data in kg.edges:
            relation = data["relation"]
            if "_" in relation:
                continue
            dst_type = kg.entity_type(dst)
            if dst_type not in TYPE_NOUNS_PLURAL:
                continue
            groups[(src, relation)].append((dst, dst_type, data.get("source", "")))

        questions = []
        for (subject, relation), objs in groups.items():
            if len(objs) < 2:
                continue
            obj_types = {otype for _, otype, _ in objs}
            if len(obj_types) != 1:
                continue
            obj_type = obj_types.pop()
            type_plural = TYPE_NOUNS_PLURAL.get(obj_type)
            if not type_plural:
                continue

            answer_list = [obj for obj, _, _ in objs]
            source = objs[0][2]

            text = build_aggregation_question(subject, relation, type_plural, self.lang)
            if not text:
                continue

            triple = triple_index.get((subject, relation, answer_list[0]))
            text_diff = self._estimator.text_side(triple, hop_count=1) if triple else "A2"

            q = Question(
                text=text, answer=", ".join(answer_list), answer_type=obj_type,
                text_difficulty=text_diff, question_difficulty="preA1",
                lang=self.lang, source=source,
                is_passive=False, hop_count=1, masked="aggregation",
                answer_list=answer_list,
            )
            q.question_difficulty = self._estimator.question_side(q)
            questions.append(q)
        return questions

    # ── Which ─────────────────────────────────────────────────────────────────

    def _which(self, kg: KnowledgeGraph, verb_index: dict) -> list[Question]:
        from collections import defaultdict

        _TYPE_NOUNS = {
            "ORG": "organization", "PERSON": "person", "PER": "person",
            "GPE": "place", "LOC": "place", "FAC": "facility", "WORK_OF_ART": "work",
        }

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
                            text_difficulty="B1", question_difficulty="preA1",
                            lang=self.lang, source=source_text,
                            is_passive=False, hop_count=1, masked="which",
                        )
                        q.question_difficulty = self._estimator.question_side(q)
                        questions.append(q)
        return questions


# ── Helpers ───────────────────────────────────────────────────────────────────

_SKIP_MASK_SUBJECT_TYPES = {"DATE", "TIME", "MONEY", "CARDINAL", "PERCENT", "QUANTITY"}
_LOC_TYPES = {"LOC", "GPE", "FAC"}
_SKIP_YESNO_TYPES = {"DATE", "TIME", "LOC", "GPE", "FAC"}
# Copula-like verbs that lose their predicative complement during extraction —
# every question generated from them is grammatically incomplete.
_SKIP_VERB_BASES = {"become", "be"}


def _build_verb_index(triples: list[Triple]) -> dict[tuple[str, str, str], str]:
    return {(t.subject, t.relation, t.object): t.verb_text for t in triples if t.verb_text}


def _build_passive_index(triples: list[Triple]) -> dict[tuple[str, str, str], bool]:
    return {(t.subject, t.relation, t.object): t.is_passive for t in triples}


def _build_surface_index(triples: list[Triple]) -> dict[tuple[str, str, str], str]:
    return {(t.subject, t.relation, t.object): t.object_surface for t in triples if t.object_surface}


def _build_triple_index(triples: list[Triple]) -> dict[tuple[str, str, str], Triple]:
    index: dict[tuple[str, str, str], Triple] = {}
    for t in triples:
        index.setdefault((t.subject, t.relation, t.object), t)
    return index
