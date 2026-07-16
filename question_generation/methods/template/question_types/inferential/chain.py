from __future__ import annotations

from question_generation.methods.template.question_types.base import QuestionType, GenerationContext
from question_generation.models import Question
from question_generation.methods.template.templates import build_chain_question, build_chain_variant


class ChainQuestion(QuestionType):
    tier = "inferential"

    def generate(self, ctx: GenerationContext, target_cefr: str = "B1") -> list[Question]:
        kg = ctx.kg
        s_read = ctx.estimator.score_readability(ctx.passage)
        questions = []

        for node, _ in kg.nodes:
            for path in kg.multihop_paths(node, max_hops=2):
                if len(path) < 2:
                    continue
                anchor, rel1, bridge = path[0]
                _,      rel2, target = path[1]

                bridge_type = kg.entity_type(bridge)
                if bridge_type not in {"PERSON", "PER"}:
                    continue
                target_type = kg.entity_type(target)
                if target_type is None:
                    continue

                vt2 = ctx.verb_index.get((bridge, rel2, target), "")
                is_p2 = ctx.passive_index.get((bridge, rel2, target), False)
                source = ""
                for u, v, data in kg.edges:
                    if u == bridge and v == target and data["relation"] == rel2:
                        source = data.get("source", "")
                        break

                result = build_chain_variant(
                    anchor=anchor, rel1=rel1, bridge=bridge, rel2=rel2,
                    verb2_text=vt2, answer_type=target_type,
                    lang=self.lang, is_passive2=is_p2,
                    target_cefr=target_cefr,
                )
                if result:
                    text, vocab_score = result
                    triple = ctx.triple_index.get((bridge, rel2, target))
                    s_type = ctx.estimator.score_type("chain", hop_count=2)
                    s_local = ctx.estimator.score_local(triple) if triple else 0.15
                    q = Question(
                        text=text, answer=target, answer_type=target_type,
                        difficulty=ctx.estimator.estimate(s_type, s_local, vocab_score, s_read),
                        lang=self.lang, source=source,
                        is_passive=is_p2, hop_count=2, masked="chain",
                        chain_path=f"{anchor} →[{rel1}]→ {bridge} →[{rel2}]→ {target}",
                        tier="inferential",
                        score_type=s_type, score_local=s_local,
                        score_vocab=vocab_score, score_readability=s_read,
                    )
                    questions.append(q)
        return questions
