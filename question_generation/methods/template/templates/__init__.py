from __future__ import annotations

from typing import Optional

from question_generation.methods.template.templates import _en, _fi

_REGISTRY = {"en": _en, "fi": _fi}


def _mod(lang: str):
    return _REGISTRY.get(lang)


# Shared constants
TYPE_NOUNS_PLURAL = _en.TYPE_NOUNS_PLURAL


# ── Public helpers ────────────────────────────────────────────────────────────

def question_word(lang: str, entity_type: Optional[str]) -> str:
    m = _mod(lang)
    return m.question_word(entity_type) if m else "What"


def question_word_by_case(lang: str, object_case: str, entity_type: Optional[str] = None) -> str:
    m = _mod(lang)
    return m.question_word_by_case(object_case, entity_type) if m else "What"


def relation_noun(lang: str, relation: str) -> Optional[str]:
    m = _mod(lang)
    return m.relation_noun(relation) if m else None


# ── Base question builders ────────────────────────────────────────────────────

def build_question(
    subject: str, relation: str, verb_text: str, answer_type: Optional[str],
    mask: str, lang: str, is_passive: bool = False, subject_surface: str = "",
) -> Optional[str]:
    m = _mod(lang)
    if m is None:
        return None
    qw = m.question_word(answer_type)
    vt = verb_text or relation.split("_")[0]
    return m.single(subject, relation, vt, qw, mask, is_passive, subject_surface)


def build_chain_question(
    anchor: str, rel1: str, bridge: str, rel2: str, verb2_text: str,
    answer_type: Optional[str], lang: str, is_passive2: bool = False,
) -> Optional[str]:
    m = _mod(lang)
    if m is None:
        return None
    noun = m.relation_noun(rel1)
    if noun is None:
        return None
    qw = m.question_word(answer_type)
    vt2 = verb2_text or rel2.split("_")[0]
    return m.chain(noun, bridge, rel2, vt2, qw, is_passive2)


def build_yesno_question(
    subject: str, relation: str, verb_text: str, obj: str,
    lang: str, is_passive: bool = False,
) -> Optional[str]:
    m = _mod(lang)
    if m is None:
        return None
    vt = verb_text or relation.split("_")[0]
    return m.yesno(subject, relation, vt, obj, is_passive)


def build_comparison_question(
    entity_a: str, entity_b: str, verb_past: str, lang: str, earlier: bool = True,
) -> Optional[str]:
    m = _mod(lang)
    return m.comparison(entity_a, entity_b, verb_past, earlier) if m else None


def build_aggregation_question(
    subject: str, verb_base: str, entity_type: str, lang: str,
) -> Optional[str]:
    m = _mod(lang)
    return m.aggregation(subject, verb_base, entity_type) if m else None


def build_count_question(
    subject: str, verb_base: str, entity_type: str, lang: str,
) -> Optional[str]:
    m = _mod(lang)
    return m.count(subject, verb_base, entity_type) if m else None


def build_multi_anchor_question(
    subject: str, subject_type: str, obj: str, obj_type: str, lang: str,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    return m.multi_anchor(subject, subject_type, obj, obj_type) if m else None


def build_anchor_question(
    anchor: str, anchor_type: Optional[str], lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    return m.anchor(anchor, anchor_type) if m else None


def build_which_question(
    subject: str, verb_base: str, entity_type: str,
    hint_prep: str, hint_val: str, lang: str,
) -> Optional[str]:
    m = _mod(lang)
    return m.which(subject, verb_base, entity_type, hint_prep, hint_val) if m else None


# ── Variant builders (lang dispatchers) ───────────────────────────────────────
# target_cefr kept in signatures for API compatibility with callers;
# difficulty variation is now handled by Word2Vec in variant_processor only.

def build_single_variant(
    subject: str, relation: str, vt: str, qw: str, mask: str, lang: str,
    is_passive: bool = False, subject_surface: str = "", target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_single_variant(subject, relation, vt, qw, mask, is_passive, subject_surface)


def build_yesno_variant(
    subject: str, relation: str, vt: str, obj: str, lang: str,
    is_passive: bool = False, is_false_claim: bool = False, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_yesno_variant(subject, relation, vt, obj, is_passive)


def build_chain_variant(
    anchor: str, rel1: str, bridge: str, rel2: str, verb2_text: str,
    answer_type: Optional[str], lang: str, is_passive2: bool = False, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_chain_variant(rel1, bridge, rel2, verb2_text, answer_type, is_passive2)


def build_comparison_variant(
    entity_a: str, entity_b: str, verb_past: str, lang: str,
    earlier: bool = True, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_comparison_variant(entity_a, entity_b, verb_past, earlier)


def build_aggregation_variant(
    subject: str, verb_base: str, entity_type: str, lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_aggregation_variant(subject, verb_base, entity_type)


def build_count_variant(
    subject: str, verb_base: str, entity_type: str, lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_count_variant(subject, verb_base, entity_type)


def build_which_variant(
    subject: str, verb_base: str, entity_type: str, hint_prep: str, hint_val: str,
    lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    m = _mod(lang)
    if m is None:
        return None
    return m.build_which_variant(subject, verb_base, entity_type, hint_prep, hint_val)