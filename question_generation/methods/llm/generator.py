"""
LLM-based question generation: raw text + anchor → Claude/GPT → question.

Does NOT use the KG — operates directly on the passage text.
Use cases:
  - Abstract passages where KG extraction yields 0 typed-entity triples
  - C1/C2 question synthesis for training data (template method cannot reach C1/C2)
  - Faithfulness filter: verify generated question covers only content in the passage

Models: Claude API (claude-haiku-4-5 for cost, claude-sonnet-5 for quality)
Status: NOT YET IMPLEMENTED
"""
from __future__ import annotations
from question_generation.methods.base import QGMethod


class LLMMethod(QGMethod):
    def generate(self, text, anchor, cefr, lang, subgraph=None):
        raise NotImplementedError("LLMMethod is not yet implemented")
