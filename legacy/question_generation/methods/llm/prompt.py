"""
Prompt templates for LLM-based question generation.

Design: CEFR level and anchor are injected as control variables.
The LLM receives the raw passage and must generate a question
that is answerable from the text and grounded in the anchor entity.
Status: NOT YET IMPLEMENTED
"""


def build_prompt(text: str, anchor: str, cefr: str, lang: str) -> str:
    """Build a question generation prompt for the target CEFR level and language."""
    raise NotImplementedError
