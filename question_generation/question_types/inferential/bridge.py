from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext, SKIP_VERB_BASES
from question_generation.models import Question

_BRIDGE_TYPE_NOUNS: dict[str, str] = {
    "ORG": "organization",
    "PERSON": "person", "PER": "person",
    "GPE": "place",
    "LOC": "location",
    "FAC": "facility",
    "WORK_OF_ART": "work",
    "PRODUCT": "product",
}

_VALID_ANCHOR_TYPES = {"PERSON", "PER", "ORG", "GPE", "LOC", "FAC"}
_SKIP_ENDPOINT_TYPES = {"DATE", "TIME", "CARDINAL", "PERCENT", "QUANTITY", "MONEY", "ORDINAL"}


class BridgeQuestion(QuestionType):
    """'Which organization did Nokia acquire that became Nokia Mobile Phones?' → Mobira"""
    tier = "inferential"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        if self.lang != "en":
            return []

        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        s_type = ctx.estimator.score_type("bridge")
        s_vocab = ctx.estimator.score_vocab(False, "bridge")

        # Build edge indices keyed by the node on each side
        in_edges: dict[str, list] = defaultdict(list)   # dst → [(src, rel, data)]
        out_edges: dict[str, list] = defaultdict(list)  # src → [(dst, rel, data)]

        for src, dst, data in kg.edges:
            rel = data["relation"]
            if rel.split("_")[0] in SKIP_VERB_BASES or rel.endswith("_as"):
                continue
            in_edges[dst].append((src, rel, data))
            out_edges[src].append((dst, rel, data))

        # Pre-compute node degree for picking the most prominent out-edge dst
        degree: dict[str, int] = defaultdict(int)
        for s, d, _ in kg.edges:
            degree[s] += 1
            degree[d] += 1

        seen_texts: set = set()
        questions: list[Question] = []

        for bridge_node, bridge_data in kg.nodes:
            bridge_type = bridge_data.get("entity_type")
            type_noun = _BRIDGE_TYPE_NOUNS.get(bridge_type)
            if type_noun is None:
                continue

            # Active in-edges only (skip passive: "was acquired by" template is awkward)
            b_in = [
                (src, rel, d) for src, rel, d in in_edges[bridge_node]
                if kg.entity_type(src) in _VALID_ANCHOR_TYPES and not rel.endswith("_by")
            ]
            # Active out-edges only to non-trivial endpoints, ranked by dst prominence
            b_out = sorted(
                [
                    (dst, rel, d) for dst, rel, d in out_edges[bridge_node]
                    if kg.entity_type(dst) not in _SKIP_ENDPOINT_TYPES and not rel.endswith("_by")
                ],
                key=lambda t: degree[t[0]], reverse=True,
            )

            if not b_in or not b_out:
                continue

            for src, rel1, data1 in b_in:
                source_text = data1.get("source", "")
                if not source_text:
                    continue
                verb1 = rel1.split("_")[0]  # base form for "did <verb>"

                # One question per (bridge, src) pair — use the highest-degree out-edge
                # that comes from a different sentence than the in-edge
                for dst, rel2, data2 in b_out:
                    if dst == src or dst == bridge_node:
                        continue
                    dst_text = data2.get("source", "")
                    if not dst_text or source_text == dst_text:
                        continue

                    verb2_surface = ctx.verb_index.get((bridge_node, rel2, dst), "") or rel2.split("_")[0]
                    rel2_parts = rel2.split("_", 1)
                    prep2 = rel2_parts[1] if len(rel2_parts) > 1 else None
                    verb2_phrase = f"{verb2_surface} {prep2}" if prep2 else verb2_surface

                    text = f"Which {type_noun} did {src} {verb1} that {verb2_phrase} {dst}?"
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)

                    questions.append(Question(
                        text=text, answer=bridge_node, answer_type=bridge_type,
                        difficulty="C1",
                        lang=self.lang, source=source_text,
                        is_passive=False, hop_count=2, masked="bridge",
                        tier="inferential",
                        score_type=s_type, score_local=0.5,
                        score_vocab=s_vocab, score_readability=s_read,
                    ))
                    break  # one question per (bridge, src) — stop after first valid out-edge

        return questions
