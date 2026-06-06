from __future__ import annotations

from typing import Optional

from question_generation.templates import _en, _fi

# Map CEFR levels to passage difficulty score [0, 1]
CEFR_TO_DIFFICULTY = {
    "A1": 0.00, "A2": 0.25, "B1": 0.50, "B2": 0.75, "C1": 0.85, "C2": 1.00,
}

# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY = {"en": _en, "fi": _fi}


def _mod(lang: str):
    return _REGISTRY.get(lang)


# ── Shared constants (language-agnostic keys) ─────────────────────────────────

# Entity-type keys that support plural/count questions — used as an eligibility
# set by wh.py and count.py before calling build_aggregation/count.
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


# ── Build functions ───────────────────────────────────────────────────────────

def build_question(
    subject: str,
    relation: str,
    verb_text: str,
    answer_type: Optional[str],
    mask: str,
    lang: str,
    is_passive: bool = False,
    subject_surface: str = "",
) -> Optional[str]:
    m = _mod(lang)
    if m is None:
        return None
    qw = m.question_word(answer_type)
    vt = verb_text or relation.split("_")[0]
    return m.single(subject, relation, vt, qw, mask, is_passive, subject_surface)


def build_chain_question(
    anchor: str,
    rel1: str,
    bridge: str,
    rel2: str,
    verb2_text: str,
    answer_type: Optional[str],
    lang: str,
    is_passive2: bool = False,
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
    subject: str,
    relation: str,
    verb_text: str,
    obj: str,
    lang: str,
    is_passive: bool = False,
) -> Optional[str]:
    m = _mod(lang)
    if m is None:
        return None
    vt = verb_text or relation.split("_")[0]
    return m.yesno(subject, relation, vt, obj, is_passive)


def build_comparison_question(
    entity_a: str,
    entity_b: str,
    verb_past: str,
    lang: str,
    earlier: bool = True,
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
    subject: str,
    subject_type: str,
    obj: str,
    obj_type: str,
    lang: str,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Returns (question_text, vocab_score) or None. Target CEFR determines template difficulty."""
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    m = _mod(lang)
    return m.multi_anchor(subject, subject_type, obj, obj_type, passage_difficulty) if m else None


def build_multi_anchor_question_variants(
    subject: str,
    subject_type: str,
    obj: str,
    obj_type: str,
    lang: str,
) -> Optional[list[tuple[str, float]]]:
    """Returns all difficulty variants or None."""
    m = _mod(lang)
    if m is None:
        return None
    templates = m._MULTI_ANCHOR_TEMPLATES
    from question_generation.templates._en import _OBJ_PREP
    prep = _OBJ_PREP.get(obj_type, "in")
    return [
        (template.format(subject=subject, prep=prep, obj=obj), score)
        for score, template in templates
    ]


def build_anchor_question(
    anchor: str, anchor_type: Optional[str], lang: str,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Returns (question_text, vocab_score) or None. Target CEFR determines template difficulty."""
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    m = _mod(lang)
    return m.anchor(anchor, anchor_type, passage_difficulty) if m else None


def build_anchor_question_variants(
    anchor: str, anchor_type: Optional[str], lang: str,
) -> Optional[list[tuple[str, float]]]:
    """Returns all difficulty variants: [(text_a1, 0.0), (text_b1, 0.45), ...] or None."""
    m = _mod(lang)
    if m is None:
        return None
    templates = m._ANCHOR_TEMPLATES.get(anchor_type or "")
    if not templates:
        return None
    return [
        (template.format(anchor=anchor), score)
        for score, template in templates
    ]


def build_which_question(
    subject: str,
    verb_base: str,
    entity_type: str,
    hint_prep: str,
    hint_val: str,
    lang: str,
) -> Optional[str]:
    m = _mod(lang)
    return m.which(subject, verb_base, entity_type, hint_prep, hint_val) if m else None


# ── CEFR-aware variant builders ───────────────────────────────────────────


def build_single_variant(
    subject: str,
    relation: str,
    vt: str,
    qw: str,
    mask: str,
    lang: str,
    is_passive: bool = False,
    subject_surface: str = "",
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build wh-question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_single_variant(subject, relation, vt, qw, mask, lang, is_passive, subject_surface, target_cefr)


def build_yesno_variant(
    subject: str,
    relation: str,
    vt: str,
    obj: str,
    lang: str,
    is_passive: bool = False,
    is_false_claim: bool = False,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build yes/no question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_yesno_variant(subject, relation, vt, obj, lang, is_passive, is_false_claim, target_cefr)


def build_chain_variant(
    anchor: str,
    rel1: str,
    bridge: str,
    rel2: str,
    verb2_text: str,
    answer_type: Optional[str],
    lang: str,
    is_passive2: bool = False,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build chain question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_chain_variant(anchor, rel1, bridge, rel2, verb2_text, answer_type, lang, is_passive2, target_cefr)


def build_comparison_variant(
    entity_a: str,
    entity_b: str,
    verb_past: str,
    lang: str,
    earlier: bool = True,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build comparison question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_comparison_variant(entity_a, entity_b, verb_past, lang, earlier, target_cefr)


def build_aggregation_variant(
    subject: str,
    verb_base: str,
    entity_type: str,
    lang: str,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build aggregation question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_aggregation_variant(subject, verb_base, entity_type, lang, target_cefr)


def build_count_variant(
    subject: str,
    verb_base: str,
    entity_type: str,
    lang: str,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build count question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_count_variant(subject, verb_base, entity_type, lang, target_cefr)


def build_which_variant(
    subject: str,
    verb_base: str,
    entity_type: str,
    hint_prep: str,
    hint_val: str,
    lang: str,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build which question with CEFR-based template variant. Returns (text, vocab_score) or None."""
    m = _mod(lang)
    if m is None:
        return None
    return m.build_which_variant(subject, verb_base, entity_type, hint_prep, hint_val, lang, target_cefr)


# ── Template rerolling (pick variant by target CEFR) ──────────────────────────

_CEFR_TO_VOCAB_SCORE = {
    "A1": 0.00, "A2": 0.25, "B1": 0.50, "B2": 0.75, "C1": 0.85, "C2": 1.0,
}


def reroll_anchor_template(
    anchor: str, anchor_type: Optional[str], lang: str, target_cefr: str,
) -> Optional[tuple[str, float]]:
    """Re-render an anchor question with a template matching target CEFR."""
    variants = build_anchor_question_variants(anchor, anchor_type, lang)
    if not variants:
        return None
    target_score = _CEFR_TO_VOCAB_SCORE.get(target_cefr, 0.5)
    # Pick variant closest to target score
    best = min(variants, key=lambda v: abs(v[1] - target_score))
    return best


def reroll_multi_anchor_template(
    subject: str, subject_type: str, obj: str, obj_type: str,
    lang: str, target_cefr: str,
) -> Optional[tuple[str, float]]:
    """Re-render a multi-anchor question with a template matching target CEFR."""
    variants = build_multi_anchor_question_variants(subject, subject_type, obj, obj_type, lang)
    if not variants:
        return None
    target_score = _CEFR_TO_VOCAB_SCORE.get(target_cefr, 0.5)
    best = min(variants, key=lambda v: abs(v[1] - target_score))
    return best