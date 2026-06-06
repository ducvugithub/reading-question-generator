from __future__ import annotations

import random
from typing import Optional

# Map CEFR levels to passage difficulty score [0, 1]
CEFR_TO_DIFFICULTY = {
    "A1": 0.00, "A2": 0.25, "B1": 0.50, "B2": 0.75, "C1": 0.85, "C2": 1.00,
}

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

# Anchor templates: (CEFR_vocab_score, template)
# Score 0.0 = A1 (high-frequency), 0.25 = A2, 0.50 = B1, 0.75 = B2, 1.0 = C1+
_ANCHOR_TEMPLATES: dict[str, list[tuple[float, str]]] = {
    "DATE": [
        (0.00, "What happened in {anchor}?"),                              # A1: simple past, common verb
        (0.10, "What events took place in {anchor}?"),                     # A2: "events", "took place"
        (0.20, "What occurred in {anchor}?"),                              # B1: "occurred" (lower frequency)
        (0.35, "What significant developments marked {anchor}?"),          # B2: "significant", "marked"
        (0.50, "What major occurrences characterized {anchor}?"),          # C1: "characterized", formal
    ],
    "TIME": [
        (0.00, "What happened in {anchor}?"),
        (0.10, "What events took place in {anchor}?"),
        (0.10, "What occurred in {anchor}?"),
        (0.35, "What significant developments marked {anchor}?"),
        (0.10, "What major occurrences characterized {anchor}?"),
    ],
    "GPE": [
        (0.00, "What happened in {anchor}?"),
        (0.10, "What events took place in {anchor}?"),
        (0.10, "What occurred in {anchor}?"),
        (0.35, "What developments shaped {anchor}?"),                      # B2: "shaped"
        (0.50, "What transformative events unfolded in {anchor}?"),        # C1: "transformative", "unfolded"
    ],
    "LOC": [
        (0.00, "What happened in {anchor}?"),
        (0.10, "What events took place in {anchor}?"),
        (0.10, "What occurred in {anchor}?"),
        (0.35, "What developments shaped {anchor}?"),
        (0.10, "What transformative events unfolded in {anchor}?"),
    ],
    "FAC": [
        (0.00, "What happened in {anchor}?"),
        (0.10, "What events took place in {anchor}?"),
        (0.10, "What occurred in {anchor}?"),
        (0.35, "What developments shaped {anchor}?"),
        (0.10, "What transformative events unfolded in {anchor}?"),
    ],
    "PERSON": [
        (0.00, "What did {anchor} do?"),                                   # A1: simple past, "do"
        (0.10, "What was {anchor} involved in?"),                          # A2: passive, "involved"
        (0.20, "What role did {anchor} play?"),                            # B1: "role", "play" (idiomatic)
        (0.35, "What contributions did {anchor} make?"),                   # B2: "contributions"
        (0.50, "What influence did {anchor} exert on contemporary events?"), # C1: "exert", "contemporary"
    ],
    "PER": [
        (0.00, "What did {anchor} do?"),
        (0.10, "What was {anchor} involved in?"),
        (0.10, "What role did {anchor} play?"),
        (0.35, "What contributions did {anchor} make?"),
        (0.10, "What influence did {anchor} exert on contemporary events?"),
    ],
    "ORG": [
        (0.00, "What did {anchor} do?"),                                   # A1
        (0.10, "What was {anchor} involved in?"),                          # A2
        (0.20, "What actions did {anchor} undertake?"),                    # B1: "undertake"
        (0.35, "What strategic initiatives did {anchor} pursue?"),         # B2: "strategic", "pursue"
        (0.50, "What transformative endeavors characterized {anchor}?"),   # C1: "endeavors", "characterized"
    ],
}

_MULTI_ANCHOR_TEMPLATES: list[tuple[float, str]] = [
    (0.00, "What did {subject} do {prep} {obj}?"),                          # A1: simple
    (0.10, "What actions did {subject} take {prep} {obj}?"),                # A2: "actions"
    (0.20, "What was {subject} doing {prep} {obj}?"),                       # B1: continuous aspect
    (0.35, "What strategic moves did {subject} make {prep} {obj}?"),        # B2: "strategic", "moves"
    (0.50, "What significant undertakings characterized {subject} {prep} {obj}?"), # C1: "undertakings"
]

# Wh-question variants: object mask (e.g., "What did X verb?")
_SINGLE_OBJECT_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{qw} did {subject} {verb_base}?"),                               # A1: basic structure
    (0.10, "{qw} did {subject} {verb_base}?"),                               # A2: same as A1 (variations come from vocab/complexity)
    (0.20, "{qw} did {subject} {verb_base}?"),                               # B1: same structure, vocabulary progression in estimator
    (0.35, "{qw} did {subject} {verb_base}?"),                               # B2: same structure
    (0.50, "{qw} did {subject} {verb_base}?"),                              # C1: same structure
]

# Wh-question variants: subject mask (e.g., "What verbed Y?")
_SINGLE_SUBJECT_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{qw} {verb_text} {subject}?"),                                   # A1: simple
    (0.10, "{qw} {verb_text} {subject}?"),                                   # A2: same structure
    (0.20, "{qw} {verb_text} {subject}?"),                                   # B1: same structure
    (0.35, "{qw} {verb_text} {subject}?"),                                   # B2: same structure
    (0.50, "{qw} {verb_text} {subject}?"),                                  # C1: same structure
]

# Yes/No question variants: for true_claim
_YESNO_TRUE_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Did {subject} {verb_base} {obj}?"),                              # A1: basic
    (0.10, "Did {subject} notably {verb_base} {obj}?"),                      # A2: "notably"
    (0.20, "Did {subject} significantly {verb_base} {obj}?"),                # B1: "significantly"
    (0.35, "Did {subject} substantially {verb_base} {obj}?"),                # B2: "substantially"
    (0.50, "Did {subject} profoundly {verb_base} {obj}?"),                  # C1: "profoundly"
]

# Yes/No question variants: for false_claim
_YESNO_FALSE_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Did {subject} {verb_base} {obj}?"),                              # A1: same surface
    (0.10, "Did {subject} {verb_base} {obj} instead?"),                      # A2: "instead" signals falsity
    (0.20, "Was it true that {subject} {verb_base} {obj}?"),                 # B1: embedded clause
    (0.35, "Could one argue that {subject} {verb_base} {obj}?"),             # B2: hedging
    (0.50, "Did {subject} not rather {verb_base} something else?"),          # C1: complex negation
]

# Chain question variants
_CHAIN_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{qw} was the {noun} of {bridge} {verb_base2}?"),                 # A1: simple
    (0.10, "{qw} was the {noun} that {bridge} {verb_base2}?"),               # A2: relative clause
    (0.20, "{qw} important {noun} did {bridge} {verb_base2}?"),              # B1: "important"
    (0.35, "{qw} significant {noun} characterized {bridge}?"),               # B2: "characterized"
    (0.50, "{qw} transformative {noun} defined {bridge}?"),                  # C1: "transformative"
]

# Comparison variants
_COMPARISON_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Which was {verb_past} {adv}, {entity_a} or {entity_b}?"),       # A1: simple
    (0.10, "Which event was {verb_past} {adv}, {entity_a} or {entity_b}?"), # A2: "event"
    (0.20, "Which development was {verb_past} {adv}, {entity_a} or {entity_b}?"), # B1: "development"
    (0.35, "Which significant occurrence was {verb_past} {adv}, {entity_a} or {entity_b}?"), # B2
    (0.10, "Which transformative moment was {verb_past} {adv}, {entity_a} or {entity_b}?"), # C1
]

# Aggregation variants
_AGGREGATION_VARIANTS: list[tuple[float, str]] = [
    (0.00, "What {noun} did {subject} {verb_base}?"),                        # A1: simple
    (0.10, "Which {noun} did {subject} {verb_base}?"),                       # A2: "which" for variety
    (0.20, "What key {noun} did {subject} {verb_base}?"),                    # B1: "key"
    (0.35, "What important {noun} did {subject} {verb_base}?"),              # B2: "important"
    (0.50, "What significant {noun} did {subject} {verb_base}?"),            # C1: "significant"
]

# Count variants - natural structure, difficulty from estimator & passage
_COUNT_VARIANTS: list[tuple[float, str]] = [
    (0.00, "How many {noun} did {subject} {verb_base}?"),
]

# Which variants
_WHICH_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Which {noun} did {subject} {verb_base} {hint_prep} {hint_val}?"), # A1: simple
    (0.10, "Which {noun} did {subject} {verb_base} {hint_prep} {hint_val}, exactly?"), # A2: "exactly"
    (0.20, "What specific {noun} did {subject} {verb_base} {hint_prep} {hint_val}?"), # B1: "specific"
    (0.35, "What particular {noun} did {subject} {verb_base} {hint_prep} {hint_val}?"), # B2: "particular"
    (0.50, "What distinct {noun} did {subject} {verb_base} {hint_prep} {hint_val}?"), # C1: "distinct"
]

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
    """Select question word based on morphological case (overrides entity_type-based selection)."""
    case_map = {
        "Ine": "Where",      # inessive: where (in)
        "Ela": "Where",      # elative: where from
        "Ill": "Where",      # illative: to where
        "Ade": "Where",      # adessive: at where
        "Ess": "When",       # essive: when (temporal state)
        "Gen": "Whose",      # genitive: whose
        "Par": "What",       # partitive: what (default)
        "Nom": "What",       # nominative: what
        "Abl": "Where",      # ablative: from where
        "All": "Where",      # allative: to where
        "Com": "Who",        # comitative: with whom
        "Ins": "How",        # instrumental: by means of
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


def _select_by_difficulty(templates: list[tuple[float, str]], passage_difficulty: float) -> tuple[float, str]:
    """Pick template matching passage difficulty. Lower passage difficulty → pick easier template."""
    target_idx = passage_difficulty * (len(templates) - 1) + random.uniform(-0.1, 0.1)
    target_idx = max(0, min(len(templates) - 1, target_idx))
    return templates[int(round(target_idx))]


def multi_anchor(subject: str, subject_type: str, obj: str, obj_type: str, passage_difficulty: float = 0.5) -> tuple[str, float]:
    prep = _OBJ_PREP.get(obj_type, "in")
    vocab_score, template = _select_by_difficulty(_MULTI_ANCHOR_TEMPLATES, passage_difficulty)
    return template.format(subject=subject, prep=prep, obj=obj), vocab_score


def anchor(anchor_entity: str, anchor_type: Optional[str], passage_difficulty: float = 0.5) -> Optional[tuple[str, float]]:
    templates = _ANCHOR_TEMPLATES.get(anchor_type or "")
    if not templates:
        return None
    vocab_score, template = _select_by_difficulty(templates, passage_difficulty)
    return template.format(anchor=anchor_entity), vocab_score


def which(
    subject: str, verb_base: str, entity_type: str, hint_prep: str, hint_val: str,
) -> Optional[str]:
    noun = TYPE_NOUNS_SINGULAR.get(entity_type)
    return f"Which {noun} did {subject} {verb_base} {hint_prep} {hint_val}?" if noun else None


# ── CEFR-aware variant selectors ──────────────────────────────────────────


def build_single_variant(
    subject: str, relation: str, vt: str, qw: str,
    mask: str, lang: str, is_passive: bool, subject_surface: str = "",
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build wh-question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    verb_base = relation.split("_")[0]
    parts = relation.split("_", 1)
    prep = parts[1] if len(parts) > 1 else None

    if relation in ("be", "be_of"):
        return (f"What is {subject}?", 0.0) if mask == "object" else None

    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)

    if mask == "object":
        if is_passive:
            if relation.endswith("_by"):
                by_qw = {"Who": "By whom", "What": "By what"}.get(qw, f"By {qw.lower()}")
                vocab_score, _ = _select_by_difficulty(_SINGLE_OBJECT_VARIANTS, passage_difficulty)
                return (f"{by_qw} was {subject} {vt}?", vocab_score)
            if prep and prep not in _QW_IMPLIED_PREPS.get(qw, set()):
                vocab_score, _ = _select_by_difficulty(_SINGLE_OBJECT_VARIANTS, passage_difficulty)
                return (f"{qw} was {subject} {vt} {prep}?", vocab_score)
            vocab_score, _ = _select_by_difficulty(_SINGLE_OBJECT_VARIANTS, passage_difficulty)
            return (f"{qw} was {subject} {vt}?", vocab_score)
        vocab_score, template = _select_by_difficulty(_SINGLE_OBJECT_VARIANTS, passage_difficulty)
        return (template.format(qw=qw, subject=subject, verb_base=verb_base), vocab_score)

    if mask == "subject":
        if is_passive:
            if prep:
                vocab_score, _ = _select_by_difficulty(_SINGLE_SUBJECT_VARIANTS, passage_difficulty)
                return (f"{qw} was {vt} {prep} {subject}?", vocab_score)
            vocab_score, _ = _select_by_difficulty(_SINGLE_SUBJECT_VARIANTS, passage_difficulty)
            return (f"{qw} was {vt}?", vocab_score)
        vocab_score, template = _select_by_difficulty(_SINGLE_SUBJECT_VARIANTS, passage_difficulty)
        ctx = subject_surface or subject
        return (template.format(qw=qw, verb_text=vt, subject=ctx), vocab_score)

    return None


def build_yesno_variant(
    subject: str, relation: str, vt: str, obj: str, lang: str,
    is_passive: bool = False, is_false_claim: bool = False,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build yes/no question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    verb_base = relation.split("_")[0]
    parts = relation.split("_", 1)
    prep = parts[1] if len(parts) > 1 else None
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)

    if relation in ("be", "be_of"):
        return (f"Is {subject} {obj}?", 0.0)

    templates = _YESNO_FALSE_VARIANTS if is_false_claim else _YESNO_TRUE_VARIANTS
    vocab_score, template = _select_by_difficulty(templates, passage_difficulty)

    if is_passive:
        if prep == "by":
            return (template.format(subject=subject, verb_base=verb_base, obj=obj), vocab_score)
        if prep:
            return (template.format(subject=subject, verb_base=verb_base, obj=obj), vocab_score)
        return (template.format(subject=subject, verb_base=verb_base, obj=obj), vocab_score)
    return (template.format(subject=subject, verb_base=verb_base, obj=obj), vocab_score)


def build_chain_variant(
    anchor: str, rel1: str, bridge: str, rel2: str, verb2_text: str,
    answer_type: Optional[str], lang: str, is_passive2: bool = False,
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build chain question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    noun = relation_noun(rel1)
    if noun is None:
        return None
    qw = question_word(answer_type)
    vt2 = verb2_text or rel2.split("_")[0]
    verb_base2 = rel2.split("_")[0]
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_CHAIN_VARIANTS, passage_difficulty)
    return (template.format(qw=qw, noun=noun, bridge=bridge, verb_base2=verb_base2), vocab_score)


def build_comparison_variant(
    entity_a: str, entity_b: str, verb_past: str, lang: str,
    earlier: bool = True, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build comparison question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    adv = "earlier" if earlier else "later"
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_COMPARISON_VARIANTS, passage_difficulty)
    return (template.format(entity_a=entity_a, entity_b=entity_b, verb_past=verb_past, adv=adv), vocab_score)


def build_aggregation_variant(
    subject: str, verb_base: str, entity_type: str, lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build aggregation question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    noun = TYPE_NOUNS_PLURAL.get(entity_type)
    if not noun:
        return None
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_AGGREGATION_VARIANTS, passage_difficulty)
    return (template.format(noun=noun, subject=subject, verb_base=verb_base), vocab_score)


def build_count_variant(
    subject: str, verb_base: str, entity_type: str, lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build count question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    noun = TYPE_NOUNS_PLURAL.get(entity_type)
    if not noun:
        return None
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_COUNT_VARIANTS, passage_difficulty)
    return (template.format(noun=noun, subject=subject, verb_base=verb_base), vocab_score)


def build_which_variant(
    subject: str, verb_base: str, entity_type: str, hint_prep: str, hint_val: str,
    lang: str, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build which question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    noun = TYPE_NOUNS_SINGULAR.get(entity_type)
    if not noun:
        return None
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_WHICH_VARIANTS, passage_difficulty)
    return (template.format(noun=noun, subject=subject, verb_base=verb_base, hint_prep=hint_prep, hint_val=hint_val), vocab_score)