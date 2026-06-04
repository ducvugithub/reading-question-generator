from __future__ import annotations

from collections import defaultdict

from question_generation.question_types.base import QuestionType, GenerationContext, SKIP_VERB_BASES, SKIP_YESNO_TYPES
from question_generation.models import Question
from question_generation.templates import build_yesno_question

# Entity types eligible as false-premise distractors
_SUBSTITUTABLE_TYPES = {"ORG", "PERSON", "PER", "GPE", "LOC", "FAC", "DATE", "TIME"}


class YesNoQuestion(QuestionType):
    """
    Generates both positive (answer=Yes) and negative/false-premise (answer=No) yes/no questions.
    Positive: confirm a true KG fact.  Negative: substitute the object with a same-type decoy.
    """
    tier = "retrieval"

    def generate(self, ctx: GenerationContext) -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)

        # ── False-premise setup ───────────────────────────────────────────────
        degree: dict[str, int] = defaultdict(int)
        for src, dst, _ in kg.edges:
            degree[src] += 1
            degree[dst] += 1

        by_type: dict[str, list[str]] = defaultdict(list)
        for node, data in kg.nodes:
            etype = data.get("entity_type")
            if etype in _SUBSTITUTABLE_TYPES and degree[node] >= 3:
                by_type[etype].append(node)
        for etype in by_type:
            by_type[etype].sort(key=lambda n: degree[n], reverse=True)

        true_facts: set[tuple[str, str, str]] = {
            (src, data["relation"], dst) for src, dst, data in kg.edges
        }

        # ── Generate ──────────────────────────────────────────────────────────
        seen_texts: set = set()
        questions: list[Question] = []

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

            triple = ctx.triple_index.get((src, relation, dst))
            s_type = ctx.estimator.score_type("true_claim")
            s_local = ctx.estimator.score_local(triple) if triple else 0.1
            s_vocab = ctx.estimator.score_vocab(is_p, "true_claim")

            # Positive
            text = build_yesno_question(src, relation, vt, dst, self.lang, is_passive=is_p)
            if text and text not in seen_texts:
                seen_texts.add(text)
                questions.append(Question(
                    text=text, answer="Yes", answer_type=dst_type,
                    difficulty=ctx.estimator.estimate(s_type, s_local, s_vocab, s_read),
                    lang=self.lang, source=source,
                    is_passive=is_p, hop_count=1, masked="true_claim",
                    tier="retrieval",
                    score_type=s_type, score_local=s_local,
                    score_vocab=s_vocab, score_readability=s_read,
                ))

            # Negative (false premise) — only when a good distractor exists
            if not source or dst_type not in _SUBSTITUTABLE_TYPES:
                continue
            candidates = [
                n for n in by_type.get(dst_type, [])
                if n != dst and n != src and (src, relation, n) not in true_facts
            ]
            if not candidates:
                continue
            neg_text = build_yesno_question(src, relation, vt, candidates[0], self.lang, is_passive=is_p)
            if neg_text and neg_text not in seen_texts:
                seen_texts.add(neg_text)
                questions.append(Question(
                    text=neg_text, answer="No", answer_type=dst_type,
                    difficulty="B1",
                    lang=self.lang, source=source,
                    is_passive=is_p, hop_count=1, masked="false_claim",
                    answer_facts=[source],
                    tier="retrieval",
                    score_type=s_type, score_local=s_local,
                    score_vocab=s_vocab, score_readability=s_read,
                ))

        return questions
