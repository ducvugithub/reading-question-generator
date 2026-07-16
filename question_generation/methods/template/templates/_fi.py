from __future__ import annotations

from typing import Optional

_QUESTION_WORDS: dict[str, str] = {
    "DATE": "Milloin", "TIME": "Milloin",
    "LOC": "Missä", "GPE": "Missä", "FAC": "Missä",
    "PER": "Kuka", "PERSON": "Kuka",
    "ORG": "Mikä", "PRO": "Mikä", "WORK_OF_ART": "Mikä",
}
_DEFAULT_QW = "Mitä"

_RELATION_NOUNS: dict[str, str] = {
    "perustaa":  "perustaja",
    "johtaa":    "johtaja",
    "löytää":    "löytäjä",
    "lead":      "johtaja",
    "found":     "perustaja",
    "establish": "perustaja",
    "acquire":   "hankkija",
    "invent":    "keksijä",
}

TYPE_NOUNS_PLURAL: dict[str, str] = {
    "ORG": "organisaatioita", "PERSON": "henkilöitä", "PER": "henkilöitä",
    "GPE": "paikkoja", "LOC": "paikkoja", "FAC": "tiloja",
    "WORK_OF_ART": "teoksia", "PRODUCT": "tuotteita",
}

TYPE_NOUNS_SINGULAR: dict[str, str] = {
    "ORG": "organisaatio", "PERSON": "henkilö", "PER": "henkilö",
    "GPE": "paikka", "LOC": "paikka", "FAC": "tila",
    "WORK_OF_ART": "teos", "PRODUCT": "tuote",
}

# Single A1 base template per anchor type.
# Difficulty variation is handled entirely by Word2Vec verb replacement in variant_processor.
_ANCHOR_TEMPLATES: dict[str, str] = {
    "DATE":   "Mitä tapahtui {anchor}?",
    "TIME":   "Mitä tapahtui {anchor}?",
    "GPE":    "Mitä tapahtui {anchor}?",
    "LOC":    "Mitä tapahtui {anchor}?",
    "FAC":    "Mitä tapahtui {anchor}?",
    "PER":    "Mitä {anchor} teki?",
    "PERSON": "Mitä {anchor} teki?",
    "ORG":    "Mitä {anchor} teki?",
}

_MULTI_ANCHOR_TEMPLATE = "Mitä {subject} teki {obj}?"


def _genitive(entity: str) -> str:
    """MVP Finnish genitive: append 'n'. Correct for most vowel-final proper nouns."""
    return entity + "n"


def _yesno_clitic(verb: str) -> str:
    """Append vowel-harmonic -ko/-kö clitic. Scans right-to-left for first non-neutral vowel."""
    for ch in reversed(verb):
        if ch in "aouAOU":
            return verb + "ko"
        if ch in "äöyÄÖY":
            return verb + "kö"
    return verb + "ko"


def question_word(entity_type: Optional[str]) -> str:
    return _QUESTION_WORDS.get(entity_type or "", _DEFAULT_QW)


def question_word_by_case(object_case: str, entity_type: Optional[str] = None) -> str:
    case_map = {
        "Ine": "Missä", "Ela": "Mistä", "Ill": "Mihin", "Ade": "Missä",
        "Ess": "Milloin", "Gen": "Kenen", "Par": "Mitä", "Nom": "Mikä",
        "Abl": "Mistä", "All": "Mihin", "Com": "Kenen", "Ins": "Millä",
    }
    if object_case in case_map:
        return case_map[object_case]
    return question_word(entity_type)


def relation_noun(relation: str) -> Optional[str]:
    return _RELATION_NOUNS.get(relation.split("_")[0])


def single(
    subject: str, relation: str, vt: str, qw: str,
    mask: str, is_passive: bool, subject_surface: str = "",
) -> Optional[str]:
    if relation in ("be", "be_of"):
        return f"Mikä on {subject}?" if mask == "object" else None

    if mask == "object":
        return f"{qw} {subject} {vt}?"

    if mask == "subject":
        ctx = subject_surface or subject
        return f"{qw} {vt} {ctx}?"

    return None


def chain(
    noun: str, bridge: str, rel2: str, vt2: str, qw: str, is_passive2: bool,
) -> str:
    return f"{qw} {_genitive(bridge)} {noun} {vt2}?"


def yesno(
    subject: str, relation: str, vt: str, obj: str, is_passive: bool,
) -> Optional[str]:
    clitic_verb = _yesno_clitic(vt).capitalize()
    if is_passive:
        return f"{clitic_verb} {subject}?"
    return f"{clitic_verb} {subject} {obj}?"


def comparison(entity_a: str, entity_b: str, verb_past: str, earlier: bool) -> str:
    adv = "aikaisemmin" if earlier else "myöhemmin"
    return f"Kumpi tapahtui {adv}, {entity_a} vai {entity_b}?"


def aggregation(subject: str, verb_base: str, entity_type: str) -> Optional[str]:
    noun = TYPE_NOUNS_PLURAL.get(entity_type)
    return f"Mitä {noun} {subject} {verb_base}?" if noun else None


def count(subject: str, verb_base: str, entity_type: str) -> Optional[str]:
    noun = TYPE_NOUNS_PLURAL.get(entity_type)
    return f"Montako {noun} {subject} {verb_base}?" if noun else None


def which(
    subject: str, verb_base: str, entity_type: str, hint_prep: str, hint_val: str,
) -> Optional[str]:
    noun = TYPE_NOUNS_SINGULAR.get(entity_type)
    return f"Minkä {noun} {subject} {verb_base} {hint_val}?" if noun else None


def multi_anchor(subject: str, subject_type: str, obj: str, obj_type: str, passage_difficulty: float = 0.5) -> tuple[str, float]:
    return _MULTI_ANCHOR_TEMPLATE.format(subject=subject, obj=obj), 0.0


def anchor(anchor_entity: str, anchor_type: Optional[str], passage_difficulty: float = 0.5) -> Optional[tuple[str, float]]:
    template = _ANCHOR_TEMPLATES.get(anchor_type or "")
    if not template:
        return None
    return template.format(anchor=anchor_entity), 0.0


# ── build_* wrappers ──────────────────────────────────────────────────────────
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