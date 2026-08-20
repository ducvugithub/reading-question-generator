"""
Seq2Seq question generation: linearized KG triples → T5/FinT5 decoder.

Input:  KG triples serialized as text + anchor + CEFR prefix
Output: generated question

Training data: template-generated questions as weak supervision labels.
Models: T5 (English), Finnish-NLP/t5-small-nl24-finnish (Finnish)
Status: NOT YET IMPLEMENTED
"""
from __future__ import annotations
from question_generation.methods.base import QGMethod


class Seq2SeqMethod(QGMethod):
    def generate(self, text, anchor, cefr, lang, subgraph=None):
        raise NotImplementedError("Seq2SeqMethod is not yet implemented")
