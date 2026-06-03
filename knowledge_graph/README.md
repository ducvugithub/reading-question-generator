# knowledge_graph — Implementation Reference

Step 1 of the pipeline: extract a directed knowledge graph from a text passage using Stanza NER and dependency parsing.

---

## Files

| File | Purpose |
|---|---|
| `extractor.py` | NER + dependency parsing → `Triple` objects |
| `graph.py` | `KnowledgeGraph` wrapping a NetworkX `MultiDiGraph` |
| `coref.py` | Post-processing: heuristic coreference resolution |
| `__init__.py` | Public exports |

---

## Quick start

```python
from knowledge_graph import KnowledgeGraphExtractor, KnowledgeGraph, resolve_coreferences

extractor = KnowledgeGraphExtractor(lang="en")   # or lang="fi"
triples   = extractor.extract("Nokia was founded in 1865 in Tampere by Fredrik Idestam.")
triples   = resolve_coreferences(triples, lang="en")

kg = KnowledgeGraph()
kg.add_triples(triples)
print(kg.summary())

# Multi-hop paths for C-level question generation
for path in kg.multihop_paths("Tim Cook", max_hops=2):
    print(path)
```

Download Stanza models before first use:
```bash
python -c "import stanza; stanza.download('en'); stanza.download('fi')"
```

---

## How it works

### 1. Entity map

`_build_entity_map(sentence)` scans Stanza's NER output and builds:

```
word.id → (normalised_text, entity_type)
```

**Normalisation:** each token in the entity span is lemmatised if it carries morphological case (`Case=` in Stanza feats), otherwise the surface form is kept. This strips Finnish case suffixes (`Varsovassa → Varsova`, `Idestamin → Idestam`) while preserving English proper nouns (`Beats` stays `Beats`, not `beat`).

Stanza's Finnish compound marker `#` (e.g. `pää#kaupunki`) is stripped to give clean labels (`pääkaupunki`).

### 2. Verb-based triple extraction

For each `VERB` in the sentence:

1. **Find subject** — words with `deprel ∈ {nsubj, nsubj:pass, nsubj:cop}`, expanding conjuncts (`conj` chain) to handle coordinated subjects like `"Germany and France signed..."`.

2. **Finnish passive fallback** — if no subject is found (Finnish passives have no explicit subject), the direct `obj` is promoted to pseudo-subject. e.g. `"Nokia perustettiin..."` → Nokia is the `obj` of the passive verb, so it becomes the subject of all extracted triples.

3. **Find objects** — words with `deprel ∈ {obj, iobj, obl, obl:agent}`, again expanding conjuncts to handle `"discovered polonium and radium"`.

4. **Finnish postposition resolution** — if an oblique object is not itself a named entity but has an `nmod:poss` child that is, use the child. This resolves the Finnish agentive construction `"Fredrik Idestamin toimesta"` (by Fredrik Idestam's action) → `Fredrik Idestam` instead of `toimi`.

5. **Relation label** — built as `verb_lemma[_preposition]`:

   | Pattern | Example | Label |
   |---|---|---|
   | `obj` / `iobj` | acquired Beats | `acquire` |
   | `obl` + case prep | founded in 1865 | `found_in` |
   | `obl:agent` | founded by Idestam | `found_by` |
   | `obl` (no case child) | `perustaa` | verb lemma only |

### 3. Copula extraction

`_extract_copula(sentence)` handles sentences where the main predicate is a noun or adjective with a copula verb (`on`, `is`, `was`). These are entirely missed by the verb loop since `is/on` is tagged `AUX` not `VERB`.

Pattern: find words that have a `cop` dependent, then extract their `nsubj:cop` / `nsubj` as subject.

```
"Helsinki on Suomen pääkaupunki."
→ (Helsinki, be, pääkaupunki)
→ (Helsinki, be_of, Suomi)      ← from nmod:poss of the predicate nominal
```

### 4. Coreference resolution (`coref.py`)

A lightweight post-processing pass over the ordered triple list. Processes triples **in order**, resolving each mention against only entities seen in earlier triples — this prevents a later entity from becoming the referent of an earlier pronoun.

**Rule 1 — Pronoun resolution:** `she/he/hän/...` → most recently introduced `PERSON`/`PER` entity.

**Rule 2 — Partial name resolution:** a single-token name whose token matches the *last token* of a known full entity name of the same type → the full name.

```
Triple order:
  (Marie Curie, bear_in, Warsaw)   ← Marie Curie added to seen
  (she, discover, polonium)        ← she → Marie Curie  ✓
  (Curie, receive, Nobel Prize)    ← Curie → Marie Curie (last token match) ✓
```

---

## Graph structure

`KnowledgeGraph` wraps `networkx.MultiDiGraph` (supports multiple distinct relations between the same node pair).

```python
kg.nodes                        # (entity_text, {entity_type: ...})
kg.edges                        # (src, dst, {relation: ..., source: ...})
kg.entity_type("Nokia")         # → "ORG"
kg.neighbors("Nokia")           # → [("found_in", "1865"), ...]
kg.multihop_paths("A", max_hops=2)  # → list of edge-path lists, cycle-safe
```

---

## Language support

| Feature | English | Finnish |
|---|---|---|
| NER | ✓ (OntoNotes) | ✓ (Turku NER) |
| Active voice triples | ✓ | ✓ |
| Passive (`was founded`) | ✓ via `nsubj:pass` | ✓ via obj fallback |
| Copula (`is`, `on`) | ✓ | ✓ via `nsubj:cop` |
| Coordinated subjects/objects | ✓ | ✓ |
| Case lemmatisation | not needed | ✓ (`Case=` feats) |
| Postposition agents (`toimesta`) | n/a | ✓ via `nmod:poss` walk |
| Coreference (pronoun) | ✓ heuristic | ✓ heuristic |
| Coreference (partial name) | ✓ heuristic | ✓ heuristic |

---

## Known limitations

| Issue | Notes |
|---|---|
| Relation labels are shallow | `found_in` for both time and place; deeper semantics need an LLM or relation classifier |
| Coreference is heuristic | Pronoun-to-most-recent-person fails with multiple persons of the same type; no gender resolution |
| Coreference only within a passage | Cross-sentence coref works but cross-passage does not |
| Non-entity objects remain | `polonium`, `radium` have no entity type; they appear as nodes with type `—` |
| `Suomi armeija` vs `Suomen armeija` | Multi-word entity lemmatisation joins tokens with spaces but loses compound morphology |
| Finnish `toimesta` pattern | Only resolved one level deep (`nmod:poss`); nested postpositional phrases are not handled |
