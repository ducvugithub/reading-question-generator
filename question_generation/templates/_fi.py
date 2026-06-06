from __future__ import annotations

import random
from typing import Optional

# Map CEFR levels to passage difficulty score [0, 1]
CEFR_TO_DIFFICULTY = {
    "A1": 0.00, "A2": 0.25, "B1": 0.50, "B2": 0.75, "C1": 0.85, "C2": 1.00,
}

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

# Anchor templates: (CEFR_vocab_score, template)
_ANCHOR_TEMPLATES: dict[str, list[tuple[float, str]]] = {
    "DATE": [
        (0.00, "Mitä tapahtui {anchor}?"),                                  # A1: simple, common verbs
        (0.10, "Mitä merkittävää tapahtui {anchor}?"),                      # A2: "merkittävää" (significant)
        (0.20, "Mitä kehitystä tapahtui {anchor}?"),                        # B1: "kehitystä" (development)
        (0.35, "Mitkä merkittävät tapahtumat leimaisivat {anchor}?"),       # B2: "leimaisivat" (marked)
        (0.50, "Minkälaisia transformatiivisia ilmiöitä {anchor} koki?"),   # C1: "transformatiivisia", formal
    ],
    "TIME": [
        (0.00, "Mitä tapahtui {anchor}?"),
        (0.10, "Mitä merkittävää tapahtui {anchor}?"),
        (0.10, "Mitä kehitystä tapahtui {anchor}?"),
        (0.35, "Mitkä merkittävät tapahtumat leimaisivat {anchor}?"),
        (0.10, "Minkälaisia transformatiivisia ilmiöitä {anchor} koki?"),
    ],
    "GPE": [
        (0.00, "Mitä tapahtui {anchor}?"),
        (0.10, "Mitä merkittävää tapahtui {anchor}?"),
        (0.10, "Mitä kehitystä tapahtui {anchor}?"),
        (0.35, "Mitkä tapahtumat muotoilivat {anchor}?"),                   # B2: "muotoilivat" (shaped)
        (0.50, "Mitkä muuntavat ilmiöt määrittelivät {anchor}?"),           # C1: "muuntavat", "määrittelivät"
    ],
    "LOC": [
        (0.00, "Mitä tapahtui {anchor}?"),
        (0.10, "Mitä merkittävää tapahtui {anchor}?"),
        (0.10, "Mitä kehitystä tapahtui {anchor}?"),
        (0.35, "Mitkä tapahtumat muotoilivat {anchor}?"),
        (0.10, "Mitkä muuntavat ilmiöt määrittelivät {anchor}?"),
    ],
    "FAC": [
        (0.00, "Mitä tapahtui {anchor}?"),
        (0.10, "Mitä merkittävää tapahtui {anchor}?"),
        (0.10, "Mitä kehitystä tapahtui {anchor}?"),
        (0.35, "Mitkä tapahtumat muotoilivat {anchor}?"),
        (0.10, "Mitkä muuntavat ilmiöt määrittelivät {anchor}?"),
    ],
    "PER": [
        (0.00, "Mitä {anchor} teki?"),                                      # A1: simple
        (0.10, "Mihin {anchor} osallistui?"),                               # A2: "osallistui" (participated)
        (0.20, "Mitä panosta {anchor} antoi?"),                             # B1: "panosta" (contribution)
        (0.35, "Mitä merkittävää vaikutusta {anchor} oli?"),                # B2: "vaikutusta" (influence)
        (0.50, "Kuinka {anchor} vaikutti ajalleen ja perintönsä kautta?"),  # C1: complex sentence structure
    ],
    "PERSON": [
        (0.00, "Mitä {anchor} teki?"),
        (0.10, "Mihin {anchor} osallistui?"),
        (0.10, "Mitä panosta {anchor} antoi?"),
        (0.35, "Mitä merkittävää vaikutusta {anchor} oli?"),
        (0.10, "Kuinka {anchor} vaikutti ajalleen ja perintönsä kautta?"),
    ],
    "ORG": [
        (0.00, "Mitä {anchor} teki?"),                                      # A1
        (0.10, "Mihin {anchor} osallistui?"),                               # A2
        (0.20, "Mitä toimia {anchor} toteutti?"),                           # B1: "toteutti" (implemented)
        (0.35, "Mitä strategisia aloitteita {anchor} teki?"),               # B2: "strategisia", "aloitteita"
        (0.10, "Minkälaiset transformatiiviset pyrkimykset määrittelivät {anchor}?"), # C1
    ],
}

_MULTI_ANCHOR_TEMPLATES: list[tuple[float, str]] = [
    (0.00, "Mitä {subject} teki {obj}?"),                                   # A1
    (0.10, "Mihin {subject} osallistui {obj}?"),                            # A2
    (0.10, "Mitä toimia {subject} teki {obj}?"),                            # B1
    (0.35, "Mitä strategisia liikkeitä {subject} teki {obj}?"),             # B2
    (0.10, "Mitkä merkittävät aloitteet luonnehtivat {subject}? {obj}"),    # C1
]

# Wh-question variants: object mask (e.g., "Mitä X teki?")
_SINGLE_OBJECT_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{qw} {subject} {verb_base}?"),                                   # A1: basic
    (0.10, "{qw} {subject} erityisesti {verb_base}?"),                       # A2: "erityisesti" (particularly)
    (0.20, "{qw} keskeiset asiat {subject} {verb_base}?"),                   # B1: "keskeiset asiat" (key points)
    (0.35, "{qw} olennaiset näkökulmat {subject} {verb_base}?"),             # B2: "olennaiset" (essential)
    (0.50, "{qw} ratkaisevat tekijät {subject} {verb_base}?"),               # C1: "ratkaisevat" (decisive)
]

# Wh-question variants: subject mask (e.g., "Mitä verbi Y?")
_SINGLE_SUBJECT_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{qw} {verb_text} {subject}?"),                                   # A1: simple
    (0.10, "{qw} huomattavasti {verb_text} {subject}?"),                     # A2: "huomattavasti" (notably)
    (0.20, "{qw} merkittävä entiteetti {verb_text} {subject}?"),             # B1: "merkittävä" (significant)
    (0.35, "{qw} suuri voima {verb_text} {subject}?"),                       # B2: "suuri voima" (major force)
    (0.50, "{qw} muuntava tekijä {verb_text} {subject}?"),                   # C1: "muuntava" (transformative)
]

# Yes/No question variants: for true_claim
_YESNO_TRUE_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{verb_base_ko} {subject} {obj}?"),                               # A1: basic with clitic
    (0.10, "{verb_base_ko} {subject} huomattavasti {obj}?"),                 # A2: "huomattavasti"
    (0.20, "{verb_base_ko} {subject} merkittävästi {obj}?"),                 # B1: "merkittävästi"
    (0.35, "{verb_base_ko} {subject} olennaisesti {obj}?"),                  # B2: "olennaisesti"
    (0.50, "{verb_base_ko} {subject} perusteellisesti {obj}?"),             # C1: "perusteellisesti"
]

# Yes/No question variants: for false_claim
_YESNO_FALSE_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{verb_base_ko} {subject} {obj}?"),                               # A1: surface
    (0.10, "{verb_base_ko} {subject} {obj} sen sijaan?"),                    # A2: "sen sijaan" (instead)
    (0.20, "Oliko totta, että {subject} {verb_base} {obj}?"),                # B1: embedded
    (0.35, "Voitaisiko väittää, että {subject} {verb_base} {obj}?"),         # B2: hedging
    (0.50, "Eikö {subject} ennemmin {verb_base} jotain muuta?"),             # C1: complex negation
]

# Chain question variants
_CHAIN_VARIANTS: list[tuple[float, str]] = [
    (0.00, "{qw} oli {bridge_noun} {bridge} {verb_base2}?"),                 # A1: simple
    (0.10, "{qw} oli {bridge_noun} jonka {bridge} {verb_base2}?"),           # A2: relative
    (0.20, "{qw} tärkeä {bridge_noun} {bridge} {verb_base2}?"),              # B1: "tärkeä"
    (0.35, "{qw} merkittävä {bridge_noun} määritti {bridge}?"),              # B2: "määritti"
    (0.50, "{qw} muuntava {bridge_noun} määritteli {bridge}?"),              # C1: "muuntava"
]

# Comparison variants
_COMPARISON_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Kumpi {verb_past_ko}, {entity_a} vai {entity_b}?"),              # A1: simple
    (0.10, "Kumpi tapahtuma {verb_past_ko}, {entity_a} vai {entity_b}?"),    # A2: "tapahtuma"
    (0.20, "Kumpi kehitys {verb_past_ko}, {entity_a} vai {entity_b}?"),      # B1: "kehitys"
    (0.35, "Kumpi merkittävä tapahtuma {verb_past_ko}, {entity_a} vai {entity_b}?"), # B2
    (0.10, "Kumpi muuntava hetki {verb_past_ko}, {entity_a} vai {entity_b}?"), # C1
]

# Aggregation variants
_AGGREGATION_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Mitä {noun} {subject} {verb_base}?"),                            # A1: simple
    (0.10, "Mitkä {noun} {subject} {verb_base}?"),                           # A2: "mitkä" variant
    (0.20, "Mitkä keskeiset {noun} {subject} {verb_base}?"),                 # B1: "keskeiset"
    (0.35, "Mitkä tärkeät {noun} {subject} {verb_base}?"),                   # B2: "tärkeät"
    (0.50, "Mitkä merkittävät {noun} {subject} {verb_base}?"),               # C1: "merkittävät"
]

# Count variants - natural structure, difficulty from estimator & passage
_COUNT_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Montako {noun} {subject} {verb_base}?"),
]

# Which variants
_WHICH_VARIANTS: list[tuple[float, str]] = [
    (0.00, "Minkä {noun} {subject} {verb_base} {hint_prep} {hint_val}?"),    # A1: simple
    (0.10, "Minkä {noun} {subject} {verb_base} {hint_prep} {hint_val} tarkasti?"), # A2
    (0.10, "Mikä spesifinen {noun} {subject} {verb_base} {hint_prep} {hint_val}?"), # B1
    (0.35, "Mikä erityinen {noun} {subject} {verb_base} {hint_prep} {hint_val}?"), # B2
    (0.10, "Mikä erillinen {noun} {subject} {verb_base} {hint_prep} {hint_val}?"), # C1
]


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
    """Select question word based on morphological case (overrides entity_type-based selection)."""
    case_map = {
        "Ine": "Missä",      # inessive: where (in)
        "Ela": "Mistä",      # elative: where from
        "Ill": "Mihin",      # illative: to where
        "Ade": "Missä",      # adessive: at where
        "Ess": "Milloin",    # essive: when (temporal state)
        "Gen": "Kenen",      # genitive: whose
        "Par": "Mitä",       # partitive: what (default)
        "Nom": "Mikä",       # nominative: what
        "Abl": "Mistä",      # ablative: from where
        "All": "Mihin",      # allative: to where
        "Com": "Kenen",      # comitative: with whom
        "Ins": "Millä",      # instrumental: by means of
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


def _select_by_difficulty(templates: list[tuple[float, str]], passage_difficulty: float) -> tuple[float, str]:
    """Pick template matching passage difficulty."""
    target_idx = passage_difficulty * (len(templates) - 1) + random.uniform(-0.1, 0.1)
    target_idx = max(0, min(len(templates) - 1, target_idx))
    return templates[int(round(target_idx))]


def multi_anchor(subject: str, subject_type: str, obj: str, obj_type: str, passage_difficulty: float = 0.5) -> tuple[str, float]:
    vocab_score, template = _select_by_difficulty(_MULTI_ANCHOR_TEMPLATES, passage_difficulty)
    return template.format(subject=subject, obj=obj), vocab_score


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
    return f"Minkä {noun} {subject} {verb_base} {hint_val}?" if noun else None


# ── CEFR-aware variant selectors ──────────────────────────────────────────


def build_single_variant(
    subject: str, relation: str, vt: str, qw: str,
    mask: str, lang: str, is_passive: bool, subject_surface: str = "",
    target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build wh-question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    verb_base = relation.split("_")[0]

    if relation in ("be", "be_of"):
        return (f"Mikä on {subject}?" if mask == "object" else None, 0.0)

    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)

    if mask == "object":
        vocab_score, template = _select_by_difficulty(_SINGLE_OBJECT_VARIANTS, passage_difficulty)
        return (template.format(qw=qw, subject=subject, verb_base=verb_base), vocab_score)

    if mask == "subject":
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
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)

    if relation in ("be", "be_of"):
        return (f"Onko {subject} {obj}?", 0.0)

    templates = _YESNO_FALSE_VARIANTS if is_false_claim else _YESNO_TRUE_VARIANTS
    vocab_score, template = _select_by_difficulty(templates, passage_difficulty)

    clitic_verb = _yesno_clitic(verb_base)
    return (template.format(subject=subject, verb_base=verb_base, verb_base_ko=clitic_verb, obj=obj), vocab_score)


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
    verb_base2 = rel2.split("_")[0]
    bridge_noun = _genitive(bridge)
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_CHAIN_VARIANTS, passage_difficulty)
    return (template.format(qw=qw, bridge_noun=noun, bridge=bridge_noun, verb_base2=verb_base2), vocab_score)


def build_comparison_variant(
    entity_a: str, entity_b: str, verb_past: str, lang: str,
    earlier: bool = True, target_cefr: str = "B1",
) -> Optional[tuple[str, float]]:
    """Build comparison question with variant selected by target CEFR. Returns (text, vocab_score) or None."""
    adv = "aikaisemmin" if earlier else "myöhemmin"
    passage_difficulty = CEFR_TO_DIFFICULTY.get(target_cefr, 0.5)
    vocab_score, template = _select_by_difficulty(_COMPARISON_VARIANTS, passage_difficulty)
    # In Finnish, we need a simplified approach - just use the base verb form
    return (template.format(entity_a=entity_a, entity_b=entity_b, verb_past_ko=verb_past, adv=adv), vocab_score)


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