from __future__ import annotations

from typing import Optional

_QUESTION_WORDS: dict[str, str] = {
    "DATE": "When", "TIME": "When",
    "LOC": "Where", "GPE": "Where", "FAC": "Where",
    "PERSON": "Who", "PER": "Who",
    "ORG": "What", "WORK_OF_ART": "What", "PRO": "What",
    "LAW": "What", "EVENT": "What", "PRODUCT": "What",
}
_DEFAULT_QW = "What"

_RELATION_NOUNS: dict[str, str] = {
    "lead":      "leader",
    "found":     "founder",
    "establish": "founder",
    "discover":  "discoverer",
    "acquire":   "acquirer",
    "create":    "creator",
    "invent":    "inventor",
    "write":     "author",
    "head":      "head",
    "direct":    "director",
}

TYPE_NOUNS_PLURAL: dict[str, str] = {
    "ORG": "organizations", "PERSON": "people", "PER": "people",
    "GPE": "places", "LOC": "places", "FAC": "facilities",
    "WORK_OF_ART": "works", "PRODUCT": "products",
}

TYPE_NOUNS_SINGULAR: dict[str, str] = {
    "ORG": "organization", "PERSON": "person", "PER": "person",
    "GPE": "place", "LOC": "place", "FAC": "facility",
    "WORK_OF_ART": "work", "PRODUCT": "product",
}

# Single A1 base template per anchor type.
# Difficulty variation is handled entirely by Word2Vec verb replacement in variant_processor.
_ANCHOR_TEMPLATES: dict[str, str] = {
    "DATE":   "What happened in {anchor}?",
    "TIME":   "What happened in {anchor}?",
    "GPE":    "What happened in {anchor}?",
    "LOC":    "What happened in {anchor}?",
    "FAC":    "What happened in {anchor}?",
    "PERSON": "What did {anchor} do?",
    "PER":    "What did {anchor} do?",
    "ORG":    "What did {anchor} do?",
}

_MULTI_ANCHOR_TEMPLATE = "What did {subject} do {prep} {obj}?"

_OBJ_PREP: dict[str, str] = {
    "DATE": "in", "TIME": "in",
    "GPE": "in", "LOC": "at", "FAC": "at",
}

_QW_IMPLIED_PREPS: dict[str, set] = {
    "When":  {"in", "at", "on", "during", "from"},
    "Where": {"in", "at", "near", "from"},
}


def question_word(entity_type: Optional[str]) -> str:
    return _QUESTION_WORDS.get(entity_type or "", _DEFAULT_QW)


def question_word_by_case(object_case: str, entity_type: Optional[str] = None) -> str:
    case_map = {
        "Ine": "Where", "Ela": "Where", "Ill": "Where", "Ade": "Where",
        "Ess": "When",  "Gen": "Whose", "Par": "What",  "Nom": "What",
        "Abl": "Where", "All": "Where", "Com": "Who",   "Ins": "How",
    }
    return case_map.get(object_case, question_word(entity_type))


def relation_noun(relation: str) -> Optional[str]:
    return _RELATION_NOUNS.get(relation.split("_")[0])


def single(
    subject: str, relation: str, vt: str, qw: str,
    mask: str, is_passive: bool, subject_surface: str = "",
) -> Optional[str]:
    verb_base = relation.split("_")[0]
    parts = relation.split("_", 1)
    prep = parts[1] if len(parts) > 1 else None

    if relation in ("be", "be_of"):
        return f"What is {subject}?" if mask == "object" else None

    if mask == "object":
        if is_passive:
            if relation.endswith("_by"):
                by_qw = {"Who": "By whom", "What": "By what"}.get(qw, f"By {qw.lower()}")
                return f"{by_qw} was {subject} {vt}?"
            if prep and prep not in _QW_IMPLIED_PREPS.get(qw, set()):
                return f"{qw} was {subject} {vt} {prep}?"
            return f"{qw} was {subject} {vt}?"
        return f"{qw} did {subject} {verb_base}?"

    if mask == "subject":
        if is_passive:
            if prep:
                return f"{qw} was {vt} {prep} {subject}?"
            return f"{qw} was {vt}?"
        return f"{qw} {vt} {subject}?"

    return None


def chain(
    noun: str, bridge: str, rel2: str, vt2: str, qw: str, is_passive2: bool,
) -> str:
    verb_base2 = rel2.split("_")[0]
    if is_passive2:
        return f"{qw} was the {noun} of {bridge} {vt2}?"
    return f"{qw} did the {noun} of {bridge} {verb_base2}?"


def yesno(
    subject: str, relation: str, vt: str, obj: str, is_passive: bool,
) -> Optional[str]:
    verb_base = relation.split("_")[0]
    parts = relation.split("_", 1)
    prep = parts[1] if len(parts) > 1 else None

    if relation in ("be", "be_of"):
        return f"Is {subject} {obj}?"
    if is_passive:
        if prep == "by":
            return f"Was {subject} {vt} by {obj}?"
        if prep:
            return f"Was {subject} {vt} {prep} {obj}?"
        return f"Was {subject} {vt}?"
    return f"Did {subject} {verb_base} {obj}?"


def comparison(entity_a: str, entity_b: str, verb_past: str, earlier: bool) -> str:
    adv = "earlier" if earlier else "later"
    return f"Which was {verb_past} {adv}, {entity_a} or {entity_b}?"


def aggregation(subject: str, verb_base: str, entity_type: str) -> Optional[str]:
    noun = TYPE_NOUNS_PLURAL.get(entity_type)
    return f"What {noun} did {subject} {verb_base}?" if noun else None


def count(subject: str, verb_base: str, entity_type: str) -> Optional[str]:
    noun = TYPE_NOUNS_PLURAL.get(entity_type)
    return f"How many {noun} did {subject} {verb_base}?" if noun else None


def which(
    subject: str, verb_base: str, entity_type: str, hint_prep: str, hint_val: str,
) -> Optional[str]:
    noun = TYPE_NOUNS_SINGULAR.get(entity_type)
    return f"Which {noun} did {subject} {verb_base} {hint_prep} {hint_val}?" if noun else None


def multi_anchor(subject: str, subject_type: str, obj: str, obj_type: str, passage_difficulty: float = 0.5) -> tuple[str, float]:
    prep = _OBJ_PREP.get(obj_type, "in")
    return _MULTI_ANCHOR_TEMPLATE.format(subject=subject, prep=prep, obj=obj), 0.0


def anchor(anchor_entity: str, anchor_type: Optional[str], passage_difficulty: float = 0.5) -> Optional[tuple[str, float]]:
    template = _ANCHOR_TEMPLATES.get(anchor_type or "")
    if not template:
        return None
    return template.format(anchor=anchor_entity), 0.0


# ── build_* wrappers (called by question type handlers) ───────────────────────
# All return (text, vocab_score=0.0) — difficulty variation comes from Word2Vec only.

def build_single_variant(
    subject: str, relation: str, vt: str, qw: str,
    mask: str, is_passive: bool, subject_surface: str = "",
) -> Optional[tuple[str, float]]:
    text = single(subject, relation, vt, qw, mask, is_passive, subject_surface)
    return (text, 0.0) if text else None


def build_yesno_variant(
    subject: str, relation: str, vt: str, obj: str,
    is_passive: bool = False,
) -> Optional[tuple[str, float]]:
    text = yesno(subject, relation, vt, obj, is_passive)
    return (text, 0.0) if text else None


def build_chain_variant(
    rel1: str, bridge: str, rel2: str, verb2_text: str,
    answer_type: Optional[str], is_passive2: bool = False,
) -> Optional[tuple[str, float]]:
    noun = relation_noun(rel1)
    if noun is None:
        return None
    qw = question_word(answer_type)
    text = chain(noun, bridge, rel2, verb2_text, qw, is_passive2)
    return (text, 0.0)


def build_comparison_variant(
    entity_a: str, entity_b: str, verb_past: str, earlier: bool = True,
) -> Optional[tuple[str, float]]:
    return (comparison(entity_a, entity_b, verb_past, earlier), 0.0)


def build_aggregation_variant(
    subject: str, verb_base: str, entity_type: str,
) -> Optional[tuple[str, float]]:
    text = aggregation(subject, verb_base, entity_type)
    return (text, 0.0) if text else None


def build_count_variant(
    subject: str, verb_base: str, entity_type: str,
) -> Optional[tuple[str, float]]:
    text = count(subject, verb_base, entity_type)
    return (text, 0.0) if text else None


def build_which_variant(
    subject: str, verb_base: str, entity_type: str, hint_prep: str, hint_val: str,
) -> Optional[tuple[str, float]]:
    text = which(subject, verb_base, entity_type, hint_prep, hint_val)
    return (text, 0.0) if text else None