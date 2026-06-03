from __future__ import annotations

from typing import Optional

# ── Question word mapping ─────────────────────────────────────────────────────

_QUESTION_WORDS: dict[str, dict[str, str]] = {
    "en": {
        "DATE": "When", "TIME": "When",
        "LOC": "Where", "GPE": "Where", "FAC": "Where",
        "PERSON": "Who", "PER": "Who",
        "ORG": "What", "WORK_OF_ART": "What", "PRO": "What",
        "LAW": "What", "EVENT": "What", "PRODUCT": "What",
    },
    "fi": {
        "DATE": "Milloin", "TIME": "Milloin",
        "LOC": "Missä", "GPE": "Missä", "FAC": "Missä",
        "PER": "Kuka", "PERSON": "Kuka",
        "ORG": "Mikä", "PRO": "Mikä", "WORK_OF_ART": "Mikä",
    },
}
_DEFAULT_QW = {"en": "What", "fi": "Mitä"}

# Prepositions whose meaning is already captured by the question word —
# don't append them or the question reads "When was Nokia founded in?"
_QW_IMPLIED_PREPS: dict[str, set[str]] = {
    "When":  {"in", "at", "on", "during", "from"},
    "Where": {"in", "at", "near", "from"},
}

# ── Relation → noun phrase for multi-hop templates ────────────────────────────
# (relation_label, english_noun, finnish_noun)
# relation_label matches the verb lemma part of the relation (before first "_")

_RELATION_NOUNS: dict[str, tuple[str, str]] = {
    "lead":      ("leader",      "johtaja"),
    "found":     ("founder",     "perustaja"),
    "establish": ("founder",     "perustaja"),
    "discover":  ("discoverer",  "löytäjä"),
    "acquire":   ("acquirer",    "hankkija"),
    "create":    ("creator",     "luoja"),
    "invent":    ("inventor",    "keksijä"),
    "write":     ("author",      "kirjoittaja"),
    "head":      ("head",        "johtaja"),
    "direct":    ("director",    "johtaja"),
    "perustaa":  ("founder",     "perustaja"),
    "johtaa":    ("leader",      "johtaja"),
    "löytää":    ("discoverer",  "löytäjä"),
}


# ── Public helpers ────────────────────────────────────────────────────────────

def question_word(lang: str, entity_type: Optional[str]) -> str:
    return _QUESTION_WORDS.get(lang, {}).get(entity_type or "", _DEFAULT_QW.get(lang, "What"))


def relation_noun(lang: str, relation: str) -> Optional[str]:
    """Return the nominalized noun for a relation verb, or None if unknown."""
    verb = relation.split("_")[0]
    pair = _RELATION_NOUNS.get(verb)
    if pair is None:
        return None
    return pair[0] if lang == "en" else pair[1]


def build_question(
    subject: str,
    relation: str,
    verb_text: str,
    answer_type: Optional[str],
    mask: str,   # "object" or "subject"
    lang: str,
    is_passive: bool = False,
    subject_surface: str = "",
) -> Optional[str]:
    """Build a single-edge question string, or None if no suitable template."""
    qw = question_word(lang, answer_type)
    vt = verb_text or relation.split("_")[0]  # fallback to lemma

    if lang == "en":
        return _en_single(subject, relation, vt, qw, mask, is_passive)
    if lang == "fi":
        return _fi_single(subject, relation, vt, qw, mask, is_passive, subject_surface)
    return None


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
    """Build a C-level 2-hop question, or None if no noun available for rel1."""
    noun = relation_noun(lang, rel1)
    if noun is None:
        return None
    qw = question_word(lang, answer_type)
    vt2 = verb2_text or rel2.split("_")[0]

    if lang == "en":
        return _en_chain(noun, bridge, rel2, vt2, qw, is_passive2)
    if lang == "fi":
        return _fi_chain(noun, bridge, vt2, qw)
    return None


# ── English templates ─────────────────────────────────────────────────────────

_SKIP_MASK_SUBJECT_TYPES = {"DATE", "TIME"}  # don't ask "what was founded in 1865?"


def _en_single(
    subject: str, relation: str, vt: str, qw: str, mask: str, is_passive: bool
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
            # Append prep unless the question word already encodes it semantically
            # ("When" implies in/at/on/during, "Where" implies in/at/near)
            if prep and prep not in _QW_IMPLIED_PREPS.get(qw, set()):
                return f"{qw} was {subject} {vt} {prep}?"
            return f"{qw} was {subject} {vt}?"
        # active: use lemma after "did"
        # "What did Apple acquire?"  "When did Apple acquire?"
        return f"{qw} did {subject} {verb_base}?"

    if mask == "subject":
        # subject=<known object>, qw based on answer (hidden subject) type
        # Skip when the known object is a date/time (makes no sense as context)
        # passive: "What was founded in 1865?"  "Who was born in Warsaw?"
        if is_passive:
            if prep:
                return f"{qw} was {vt} {prep} {subject}?"
            return f"{qw} was {vt}?"
        # active: "Who leads Apple?"  "What discovered polonium?"
        return f"{qw} {vt} {subject}?"

    return None


def _en_chain(noun: str, bridge: str, rel2: str, vt2: str, qw: str, is_passive2: bool) -> str:
    verb_base2 = rel2.split("_")[0]
    if is_passive2:
        return f"{qw} was the {noun} of {bridge} {vt2}?"
    return f"{qw} did the {noun} of {bridge} {verb_base2}?"


# ── Finnish templates ─────────────────────────────────────────────────────────

def _fi_genitive(entity: str) -> str:
    """MVP Finnish genitive: append 'n'. Correct for most vowel-final proper nouns."""
    return entity + "n"


def _fi_single(
    subject: str, relation: str, vt: str, qw: str, mask: str, is_passive: bool,
    subject_surface: str = "",
) -> Optional[str]:
    if relation in ("be", "be_of"):
        return f"Mikä on {subject}?" if mask == "object" else None

    if mask == "object":
        # Finnish verb already carries tense/voice: "Milloin Nokia perustettiin?"
        return f"{qw} {subject} {vt}?"

    if mask == "subject":
        # Use case-inflected surface form as context if available: "Kuka syntyi Varsovassa?"
        ctx = subject_surface or subject
        return f"{qw} {vt} {ctx}?"

    return None


def _fi_chain(noun: str, bridge: str, vt2: str, qw: str) -> str:
    gen = _fi_genitive(bridge)
    return f"{qw} {gen} {noun} {vt2}?"


# ── Yes / No ──────────────────────────────────────────────────────────────────

def build_yesno_question(
    subject: str,
    relation: str,
    verb_text: str,
    obj: str,
    lang: str,
    is_passive: bool = False,
) -> Optional[str]:
    """Build a Yes/No question. English only for now."""
    if lang != "en":
        return None
    vt = verb_text or relation.split("_")[0]
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


# ── Comparison ────────────────────────────────────────────────────────────────

def build_comparison_question(
    entity_a: str,
    entity_b: str,
    verb_past: str,
    lang: str,
    earlier: bool = True,
) -> Optional[str]:
    """'Which was Xed earlier/later, A or B?' English only."""
    if lang != "en":
        return None
    adv = "earlier" if earlier else "later"
    return f"Which was {verb_past} {adv}, {entity_a} or {entity_b}?"


# ── Aggregation ───────────────────────────────────────────────────────────────

TYPE_NOUNS_PLURAL: dict[str, str] = {
    "ORG": "organizations", "PERSON": "people", "PER": "people",
    "GPE": "places", "LOC": "places", "FAC": "facilities",
    "WORK_OF_ART": "works", "PRODUCT": "products",
}


def build_aggregation_question(subject: str, verb_base: str, type_noun_plural: str, lang: str) -> Optional[str]:
    """'What <type_plural> did <subject> <verb>?' English only."""
    if lang != "en":
        return None
    return f"What {type_noun_plural} did {subject} {verb_base}?"


# ── Which ─────────────────────────────────────────────────────────────────────

def build_which_question(
    subject: str,
    verb_base: str,
    type_noun: str,
    hint_prep: str,
    hint_val: str,
    lang: str,
) -> Optional[str]:
    """'Which <type> did <subject> <verb> <prep> <hint>?' English only."""
    if lang != "en":
        return None
    return f"Which {type_noun} did {subject} {verb_base} {hint_prep} {hint_val}?"
