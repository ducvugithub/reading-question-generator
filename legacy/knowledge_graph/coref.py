from __future__ import annotations

from typing import Optional

from .extractor import Triple

_PRONOUNS: dict[str, set[str]] = {
    "en": {"she", "he", "they", "it", "her", "him", "them", "we"},
    "fi": {"hän", "he", "se", "me"},
}


def resolve_coreferences(triples: list[Triple], lang: str = "en") -> list[Triple]:
    """
    Lightweight heuristic coreference resolution over a list of triples.

    Processes triples in order, resolving each mention against only the entities
    seen so far — so a pronoun or short name can only resolve to an entity that
    appeared earlier in the text, not a later one.

    Two rules:
    1. Pronoun → most recently introduced PERSON/PER entity.
    2. Single-token name whose token matches the last token of a known full
       entity name of the same type → the full name.
       e.g. 'Curie' → 'Marie Curie'

    Sets coref_distance on the resolved Triple to the number of sentences
    between the pronoun/short-name mention and its antecedent.
    """
    pronouns = {p.lower() for p in _PRONOUNS.get(lang, _PRONOUNS["en"])}

    # seen: entity_name → (entity_type, sentence_idx_first_seen)
    seen: dict[str, tuple[str, int]] = {}

    def _last_token_index() -> dict[str, list[str]]:
        idx: dict[str, list[str]] = {}
        for name in seen:
            last = name.split()[-1].lower()
            idx.setdefault(last, []).append(name)
        return idx

    def canonicalize(
        text: str, etype: Optional[str], sentence_idx: int
    ) -> tuple[str, Optional[str], int]:
        """Return (resolved_text, resolved_type, coref_distance)."""
        text_lower = text.lower()

        # Rule 1: pronoun → most recently seen PERSON/PER
        if text_lower in pronouns:
            for name, (t, sidx) in reversed(list(seen.items())):
                if t in ("PERSON", "PER"):
                    return name, t, sentence_idx - sidx
            return text, etype, 0

        # Rule 2: single-token last-name → full name of same type seen earlier
        if len(text.split()) == 1 and etype:
            idx = _last_token_index()
            candidates = [
                n for n in idx.get(text_lower, [])
                if n.lower() != text_lower and seen.get(n, (None, None))[0] == etype
            ]
            if candidates:
                best = max(candidates, key=len)
                _, sidx = seen[best]
                return best, etype, sentence_idx - sidx

        return text, etype, 0

    resolved: list[Triple] = []
    for t in triples:
        subj, st, subj_coref = canonicalize(t.subject, t.subject_type, t.sentence_idx)
        obj, ot, obj_coref = canonicalize(t.object, t.object_type, t.sentence_idx)
        resolved.append(Triple(
            subject=subj, subject_type=st,
            relation=t.relation,
            object=obj, object_type=ot,
            source=t.source,
            verb_text=t.verb_text,
            is_passive=t.is_passive,
            object_surface=t.object_surface,
            object_case=t.object_case,
            sentence_idx=t.sentence_idx,
            coref_distance=max(subj_coref, obj_coref),
            source_depth=t.source_depth,
            answer_depth=t.answer_depth,
        ))
        # Register resolved entities for future triples
        if st and subj.lower() not in pronouns:
            seen.setdefault(subj, (st, t.sentence_idx))
        if ot and obj.lower() not in pronouns:
            seen.setdefault(obj, (ot, t.sentence_idx))

    return resolved
