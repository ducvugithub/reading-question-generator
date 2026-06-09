"""
Post-processing module to create difficulty variants from generated questions.

Takes a Question and creates multiple variants at different CEFR levels
by replacing main verbs using Word2Vec synonyms based on frequency-derived CEFR levels.
"""

from __future__ import annotations

from typing import Optional, Tuple
import stanza
from question_generation.models import Question
from question_generation.word2vec_variants import get_variant_generator

# Global NLP pipelines
_nlp_en = None
_nlp_fi = None


def get_nlp_pipeline(lang: str = "en"):
    """Get or initialize stanza NLP pipeline."""
    global _nlp_en, _nlp_fi

    if lang == "en":
        if _nlp_en is None:
            _nlp_en = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse", verbose=False)
        return _nlp_en
    elif lang == "fi":
        if _nlp_fi is None:
            _nlp_fi = stanza.Pipeline("fi", processors="tokenize,pos,lemma,depparse", verbose=False)
        return _nlp_fi
    return None


def extract_main_verb(question: str, lang: str = "en") -> Optional[Tuple[str, str]]:
    """
    Extract the main content verb and its lemma from a question using Stanza POS tags.

    Finds the syntactic root with upos=VERB, falling back to the first VERB token.
    Auxiliaries (upos=AUX) are never selected, so "did" in "What did X do?" is skipped.

    Returns: (inflected_form, lemma) or None

    Examples:
      "What did Microsoft do?"      → ("do", "do")
      "When was Microsoft founded?" → ("founded", "found")
      "What happened in 1975?"      → ("happened", "happen")
    """
    text = question.rstrip('?')

    try:
        nlp = get_nlp_pipeline(lang)
        doc = nlp(text)
        if not doc.sentences:
            return None

        words = doc.sentences[0].words

        # Prefer the syntactic root if it is a content verb
        for word in words:
            if word.upos == "VERB" and word.deprel == "root":
                return (word.text, word.lemma)

        # Fallback: first content verb (not auxiliary)
        for word in words:
            if word.upos == "VERB":
                return (word.text, word.lemma)

    except Exception:
        pass

    return None


def create_question_variants(
    question: Question,
    lang: str = "en",
    create_all_levels: bool = True
) -> list[Question]:
    """
    Create difficulty variants of a question by replacing the main verb lemma.

    Args:
        question: Original Question object
        lang: Language code ("en" or "fi")
        create_all_levels: If True, create all 5 CEFR level variants.
                          If False, keep original difficulty only.

    Returns:
        List of Question variants (one per CEFR level if create_all_levels=True)
    """
    if not create_all_levels:
        return [question]

    # Extract main verb and its lemma
    verb_info = extract_main_verb(question.text, lang=lang)
    if not verb_info:
        # Can't extract verb, return original
        return [question]

    inflected_form, _ = verb_info

    # Get variant generator
    gen = get_variant_generator(lang)

    # Use the inflected form for both Word2Vec lookup and regex replacement —
    # the question contains "happened" not "happen", so the lemma can't be matched.
    variants_dict = gen.create_variants(
        question=question.text,
        main_verb=inflected_form,
        cefr_levels=["A1", "A2", "B1", "B2", "C1"]
    )

    # Create Question objects for each variant (only if text changed)
    variants = []
    cefr_order = ["A1", "A2", "B1", "B2", "C1"]
    seen_texts = set()

    for cefr in cefr_order:
        variant_text = variants_dict[cefr]

        # Skip if this exact text was already created at a different CEFR level
        if variant_text in seen_texts:
            continue
        seen_texts.add(variant_text)

        # Create new Question with variant text and updated difficulty
        variant_q = Question(
            text=variant_text,
            answer=question.answer,
            answer_type=question.answer_type,
            difficulty=cefr,  # Update difficulty to match variant
            lang=question.lang,
            source=question.source,
            is_passive=question.is_passive,
            hop_count=question.hop_count,
            masked=question.masked,
            answer_list=question.answer_list,
            answer_facts=question.answer_facts,
            tier=question.tier,
            score_type=question.score_type,
            score_local=question.score_local,
            score_vocab=question.score_vocab,
            score_readability=question.score_readability,
        )
        variants.append(variant_q)

    return variants


def expand_questions_with_variants(
    questions: list[Question],
    lang: str = "en",
    max_variants_per_question: int = 5
) -> list[Question]:
    """
    Expand a list of questions by creating variants for each.

    Args:
        questions: Original questions
        lang: Language code
        max_variants_per_question: Max variants to create per question (1-5)

    Returns:
        Expanded list with variants
    """
    expanded = []

    for q in questions:
        # Create variants
        variants = create_question_variants(q, lang=lang, create_all_levels=True)

        # Limit variants if needed
        expanded.extend(variants[:max_variants_per_question])

    return expanded