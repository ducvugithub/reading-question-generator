from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import stanza

# nsubj:cop is used in Finnish TDT for copula sentence subjects
_SUBJECT_DEPRELS = {"nsubj", "nsubj:pass", "nsubj:cop"}
_OBJECT_DEPRELS = {"obj", "iobj", "obl", "obl:agent"}
_CLAUSAL_DEPS = {"relcl", "advcl", "acl", "csubj", "ccomp", "xcomp"}


@dataclass
class Triple:
    subject: str
    subject_type: Optional[str]
    relation: str
    object: str
    object_type: Optional[str]
    source: str = ""
    verb_text: str = ""       # original inflected verb surface form (e.g. "founded", "perustettiin")
    is_passive: bool = False  # True when Voice=Pass in verb feats or nsubj:pass subject
    object_surface: str = ""  # original case-inflected surface form of object (Finnish: "Varsovassa")
    object_case: str = ""     # morphological case of object (Finnish: Nominative, Genitive, Inessive, etc.)
    sentence_idx: int = 0     # 0-based index of source sentence in document
    coref_distance: int = 0   # sentence distance between pronoun and its antecedent (0 = no coref)
    source_depth: int = 0     # number of clausal dependency relations in source sentence
    answer_depth: int = 0     # dependency tree depth of the answer (object) token in source sentence

    def __repr__(self) -> str:
        return f"({self.subject!r}, {self.relation!r}, {self.object!r})"


class KnowledgeGraphExtractor:
    """
    Extract (subject, relation, object) triples from text using Stanza.
    Lazy-loads the pipeline on first call to extract().
    """

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._nlp: Optional[stanza.Pipeline] = None

    def extract(self, text: str) -> list[Triple]:
        doc = self._pipeline()(text)
        triples: list[Triple] = []
        for sent_idx, sentence in enumerate(doc.sentences):
            entity_map, surface_map = _build_entity_map(sentence)
            depth = _sentence_depth(sentence)
            triples.extend(_extract_sentence(sentence, entity_map, surface_map, sent_idx, depth))
        return triples

    def _pipeline(self) -> stanza.Pipeline:
        if self._nlp is None:
            self._nlp = stanza.Pipeline(
                self.lang,
                processors="tokenize,ner,pos,lemma,depparse",
                use_gpu=False,
            )
        return self._nlp


# ── Entity map ────────────────────────────────────────────────────────────────

def _build_entity_map(sentence) -> tuple[dict[int, tuple], dict[int, str]]:
    """
    Returns (entity_map, surface_map).

    entity_map: word.id → (normalised_text, entity_type)
      Lemmatises case-inflected tokens so Finnish surface variants like
      "Varsovassa" → "Varsova" become canonical graph node IDs.

    surface_map: word.id → original entity surface text (e.g. "Varsovassa")
      Used to reconstruct case-inflected forms for question text.
    """
    entity_map: dict[int, tuple] = {}
    surface_map: dict[int, str] = {}
    for ent in sentence.ents:
        normalised = " ".join(
            _word_form(w) for w in sorted(ent.words, key=lambda w: w.id)
            if w.text != "'s"
        )
        for word in ent.words:
            entity_map[word.id] = (normalised, ent.type)
            surface_map[word.id] = ent.text
    return entity_map, surface_map


def _word_form(word) -> str:
    """Return lemma for case-inflected words, surface form otherwise.
    Stanza marks Finnish compound word boundaries with '#' (e.g. pää#kaupunki);
    strip those so node labels read as normal words."""
    form = word.lemma if (word.feats and "Case=" in word.feats) else word.text
    return form.replace("#", "")


# ── Sentence-level extraction ─────────────────────────────────────────────────

def _extract_sentence(
    sentence, entity_map: dict, surface_map: dict,
    sentence_idx: int = 0, source_depth: int = 0,
) -> list[Triple]:
    words_by_id = {w.id: w for w in sentence.words}
    triples: list[Triple] = []

    for verb in sentence.words:
        if verb.upos != "VERB":
            continue
        subjects, obj_fallback_ids = _find_subjects(verb, words_by_id)
        if not subjects:
            continue
        is_passive = _verb_is_passive(verb, words_by_id)
        for subject in subjects:
            subj_text, subj_type = _resolve_mention(subject, entity_map)
            for obj_word, deprel in _find_objects(verb, words_by_id, obj_fallback_ids):
                resolved = _resolve_postposition(obj_word, words_by_id, entity_map)
                obj_text, obj_type = _resolve_mention(resolved, entity_map)
                relation = _relation_label(verb, obj_word, deprel, words_by_id)
                feats = _extract_feats(resolved)
                obj_case = feats.get("Case", "")
                triples.append(Triple(
                    subject=subj_text, subject_type=subj_type,
                    relation=relation,
                    object=obj_text, object_type=obj_type,
                    source=sentence.text,
                    verb_text=verb.text,
                    is_passive=is_passive,
                    object_surface=surface_map.get(resolved.id, ""),
                    object_case=obj_case,
                    sentence_idx=sentence_idx,
                    source_depth=source_depth,
                    answer_depth=_dep_depth(resolved, words_by_id),
                ))

    triples.extend(_extract_copula(sentence, entity_map, words_by_id, sentence_idx, source_depth))
    return triples


# ── Subject finding ───────────────────────────────────────────────────────────

def _find_subjects(verb, words_by_id: dict) -> tuple[list, set[int]]:
    """
    Return (subjects, obj_fallback_ids).

    For Finnish passive verbs with no explicit nsubj, falls back to using the
    direct object as the pseudo-subject (e.g. "Nokia perustettiin" → Nokia is
    the obj but the logical subject). obj_fallback_ids is returned so those
    word IDs can be excluded from the objects list.
    """
    subjects = []
    for word in words_by_id.values():
        if word.head == verb.id and word.deprel in _SUBJECT_DEPRELS:
            subjects.extend(_expand_conjuncts(word, words_by_id))
    if subjects:
        return subjects, set()

    # Finnish passive fallback: promote obj to pseudo-subject
    obj_subjects = []
    for word in words_by_id.values():
        if word.head == verb.id and word.deprel == "obj":
            obj_subjects.extend(_expand_conjuncts(word, words_by_id))
    return obj_subjects, {w.id for w in obj_subjects}


# ── Object finding ────────────────────────────────────────────────────────────

def _find_objects(verb, words_by_id: dict, exclude_ids: set[int]) -> list[tuple]:
    """
    Return [(word, deprel)] for objects/obliques, expanding conj chains.

    exclude_ids: word IDs already used as subjects (Finnish passive fallback).
    """
    direct = [
        (word, word.deprel)
        for word in words_by_id.values()
        if word.head == verb.id
        and word.deprel in _OBJECT_DEPRELS
        and word.id not in exclude_ids
    ]
    expanded: list[tuple] = []
    seen: set[int] = set()
    for word, deprel in direct:
        for conj in _expand_conjuncts(word, words_by_id):
            if conj.id not in seen:
                expanded.append((conj, deprel))
                seen.add(conj.id)
    return expanded


def _expand_conjuncts(word, words_by_id: dict) -> list:
    """Return word and all items in its conj chain (handles coordinated NPs)."""
    result = [word]
    for w in words_by_id.values():
        if w.head == word.id and w.deprel == "conj":
            result.extend(_expand_conjuncts(w, words_by_id))
    return result


# ── Copula extraction ─────────────────────────────────────────────────────────

def _sentence_depth(sentence) -> int:
    """Count clausal dependency relations as a proxy for syntactic complexity."""
    return sum(1 for w in sentence.words if w.deprel in _CLAUSAL_DEPS)


def _dep_depth(word, words_by_id: dict) -> int:
    """Walk the dependency tree upward and return the depth of word from the root."""
    depth = 0
    seen: set[int] = set()
    current = word
    while current.head != 0 and current.id not in seen:
        seen.add(current.id)
        parent = words_by_id.get(current.head)
        if parent is None:
            break
        current = parent
        depth += 1
    return depth


def _extract_copula(sentence, entity_map: dict, words_by_id: dict,
                    sentence_idx: int = 0, source_depth: int = 0) -> list[Triple]:
    """
    Extract triples from copula sentences (X is Y).

    Handles both English ("Helsinki is the capital") and Finnish
    ("Helsinki on Suomen pääkaupunki") where the copula verb is AUX,
    not VERB, so it is missed by the main verb loop.
    """
    triples: list[Triple] = []
    cop_head_ids = {w.head for w in sentence.words if w.deprel == "cop"}

    for head_id in cop_head_ids:
        head = words_by_id.get(head_id)
        if head is None:
            continue
        subject = next(
            (w for w in sentence.words if w.head == head_id and w.deprel in _SUBJECT_DEPRELS),
            None,
        )
        if subject is None:
            continue

        subj_text, subj_type = _resolve_mention(subject, entity_map)
        pred_text, pred_type = _resolve_mention(head, entity_map)
        head_feats = _extract_feats(head)
        head_case = head_feats.get("Case", "")
        triples.append(Triple(
            subject=subj_text, subject_type=subj_type,
            relation="be",
            object=pred_text, object_type=pred_type,
            source=sentence.text,
            sentence_idx=sentence_idx,
            source_depth=source_depth,
            object_case=head_case,
            answer_depth=_dep_depth(head, words_by_id),
        ))

        # Also capture nmod of predicate nominal: "capital of Finland" → (Helsinki, be_of, Finland)
        for nmod in sentence.words:
            if nmod.head == head_id and nmod.deprel in ("nmod", "nmod:poss"):
                case = _find_case(nmod, words_by_id)
                nmod_text, nmod_type = _resolve_mention(nmod, entity_map)
                nmod_feats = _extract_feats(nmod)
                nmod_case = nmod_feats.get("Case", "")
                triples.append(Triple(
                    subject=subj_text, subject_type=subj_type,
                    relation=f"be_{case}" if case else "be_of",
                    object=nmod_text, object_type=nmod_type,
                    source=sentence.text,
                    sentence_idx=sentence_idx,
                    source_depth=source_depth,
                    object_case=nmod_case,
                    answer_depth=_dep_depth(nmod, words_by_id),
                ))

    return triples


# ── Voice detection ───────────────────────────────────────────────────────────

def _verb_is_passive(verb, words_by_id: dict) -> bool:
    """True when the verb carries passive voice (Voice=Pass in feats or nsubj:pass subject)."""
    if verb.feats and "Voice=Pass" in verb.feats:
        return True
    return any(
        w.head == verb.id and w.deprel == "nsubj:pass"
        for w in words_by_id.values()
    )


# ── Relation label ────────────────────────────────────────────────────────────

def _relation_label(verb, obj_word, deprel: str, words_by_id: dict) -> str:
    base = verb.lemma
    if deprel in ("obj", "iobj"):
        return base
    case = _find_case(obj_word, words_by_id)
    if case:
        return f"{base}_{case}"
    if deprel == "obl:agent":
        return f"{base}_by"
    return base


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_feats(word) -> dict[str, str]:
    """Parse word.feats string into a dict. E.g., "Case=Inessive|Number=Singular" → {"Case": "Inessive", "Number": "Singular"}."""
    if not word.feats:
        return {}
    features: dict[str, str] = {}
    for feat_pair in word.feats.split("|"):
        if "=" in feat_pair:
            key, value = feat_pair.split("=", 1)
            features[key] = value
    return features


def _find_case(word, words_by_id: dict) -> Optional[str]:
    for w in words_by_id.values():
        if w.head == word.id and w.deprel in ("case", "mark"):
            return w.lemma
    return None


def _resolve_postposition(word, words_by_id: dict, entity_map: dict):
    """
    For Finnish postpositions like 'toimesta' (by X's action = by X), the real
    entity is the nmod:poss child, not the postposition noun itself.
    If the word is not a named entity but has an nmod:poss/nmod child that is,
    return that child word instead.
    """
    if word.id in entity_map:
        return word
    for w in words_by_id.values():
        if w.head == word.id and w.deprel in ("nmod:poss", "nmod") and w.id in entity_map:
            return w
    return word


def _resolve_mention(word, entity_map: dict) -> tuple[str, Optional[str]]:
    """
    Return (text, entity_type) for a word.
    Uses the normalised entity span text when the word belongs to a named entity,
    otherwise falls back to the word's own lemma (stripping '#' compound markers).
    """
    if word.id in entity_map:
        return entity_map[word.id]
    return word.lemma.replace("#", ""), None
