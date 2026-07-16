"""
Post-processing module to create difficulty variants from generated questions.

Pipeline per question:
  1. Stanza POS+depparse finds the content VERB token (not AUX), returns (inflected, lemma, xpos)
  2. Skip light verbs (do, be, have, get) — no meaningful synonyms exist
  3. Word2Vec lookup uses the *lemma* for better coverage
  4. W2V result is lemmatized, then re-inflected to the original xpos via pyinflect
  5. Original inflected form is replaced in the question text
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import pyinflect
import stanza

from question_generation.models import Question
from question_generation.methods.template.word2vec_variants import get_variant_generator

# Verbs too generic for meaningful Word2Vec synonyms
_LIGHT_VERBS = {"do", "be", "have", "get"}

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


def extract_main_verb(question: str, lang: str = "en") -> Optional[Tuple[str, str, str]]:
    """
    Extract the main content verb from a question using Stanza POS + depparse.

    Returns (inflected_form, lemma, xpos) or None.
    Prefers the syntactic root with upos=VERB; falls back to first VERB token.
    Auxiliaries (upos=AUX) are never selected.

    Examples:
      "What did Microsoft do?"      → ("do", "do", "VB")
      "When was Microsoft founded?" → ("founded", "found", "VBN")
      "What happened in 1975?"      → ("happened", "happen", "VBD")
    """
    text = question.rstrip("?")

    try:
        nlp = get_nlp_pipeline(lang)
        doc = nlp(text)
        if not doc.sentences:
            return None

        words = doc.sentences[0].words

        # Prefer syntactic root if it is a content verb
        for word in words:
            if word.upos == "VERB" and word.deprel == "root":
                return (word.text, word.lemma, word.xpos)

        # Fallback: first content verb
        for word in words:
            if word.upos == "VERB":
                return (word.text, word.lemma, word.xpos)

    except Exception:
        pass

    return None


def _lemmatize(word: str, nlp) -> str:
    """Get the lemma of a single word using the Stanza pipeline."""
    try:
        doc = nlp(word)
        if doc.sentences:
            return doc.sentences[0].words[0].lemma
    except Exception:
        pass
    return word


def _inflect_to_form(lemma: str, xpos: str) -> Optional[str]:
    """Re-inflect a lemma to the target Penn Treebank form using pyinflect."""
    result = pyinflect.getInflection(lemma, tag=xpos)
    return result[0] if result else None


def create_question_variants(
    question: Question,
    lang: str = "en",
    create_all_levels: bool = True,
) -> list[Question]:
    """
    Create difficulty variants of a question by replacing the main verb.

    Flow:
      - Extract (inflected_form, lemma, xpos) via Stanza
      - Skip if lemma is a light verb
      - For each CEFR level: find W2V synonym of lemma, lemmatize it, re-inflect to xpos
      - Replace inflected_form in question text with re-inflected synonym
    """
    if not create_all_levels:
        return [question]

    verb_info = extract_main_verb(question.text, lang=lang)
    if not verb_info:
        return [question]

    inflected_form, lemma, xpos = verb_info

    if lemma in _LIGHT_VERBS:
        return [question]

    gen = get_variant_generator(lang)
    nlp = get_nlp_pipeline(lang)

    cefr_order = ["A1", "A2", "B1", "B2", "C1"]
    variants = []
    seen_texts: set[str] = set()

    for cefr in cefr_order:
        # W2V lookup on inflected form — the model was trained on inflected text so
        # "happened" has better coverage than "happen" (finds "occurred", "transpired").
        raw_synonym = gen.generator.find_verb_synonym(inflected_form, cefr)

        if raw_synonym and raw_synonym.lower() != inflected_form.lower():
            # Lemmatize the W2V result (may be returned in wrong form)
            syn_lemma = _lemmatize(raw_synonym, nlp)
            # Re-inflect to the original verb's morphological form
            inflected_syn = _inflect_to_form(syn_lemma, xpos) or raw_synonym

            pattern = r"\b" + re.escape(inflected_form) + r"\b"
            variant_text = re.sub(pattern, inflected_syn, question.text, flags=re.IGNORECASE)
        else:
            variant_text = question.text

        if variant_text in seen_texts:
            continue
        seen_texts.add(variant_text)

        variant_q = Question(
            text=variant_text,
            answer=question.answer,
            answer_type=question.answer_type,
            difficulty=cefr,
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

    return variants if variants else [question]


def expand_questions_with_variants(
    questions: list[Question],
    lang: str = "en",
    max_variants_per_question: int = 5,
) -> list[Question]:
    """Expand a list of questions by creating variants for each."""
    expanded = []
    for q in questions:
        variants = create_question_variants(q, lang=lang, create_all_levels=True)
        expanded.extend(variants[:max_variants_per_question])
    return expanded