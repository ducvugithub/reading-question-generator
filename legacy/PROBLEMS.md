# Open Problems

Tracked issues found during output review (`data/output/nokia_en.md`).

---

## #1 — Active chain template hardcodes "What" regardless of answer type

**Example:**
```
"What did the leader of Nokia merge?" → 1967
```
Should be `"When did the leader of Nokia merge?"` since the answer is a DATE.

**Root cause:** `_en_chain` in `templates.py` always returns `f"What did the {noun} of {bridge} {verb_base2}?"` — the `qw` parameter (computed from answer type) is ignored in the active branch.

**Fix:** Replace hardcoded `"What"` with `qw`.

**Status:** Fixed — `_en_chain` in `templates.py`

---

## #2 — Circular chain questions

**Example:**
```
"When was the founder of Nokia founded?" → 1865
"Who was the founder of Nokia founded?" → Fredrik Idestam
```
These are semantically broken. The multihop is traversing Nokia as a *bridge*
(e.g. `Idestam →[establish]→ Nokia →[found_in]→ 1865`), so the noun phrase
"the founder of Nokia" refers to Idestam, but the chain then asks what Nokia
itself was "founded" as — circular and confusing.

**Root cause:** `multihop_paths` returns paths where the bridge entity's own
properties (edges that originally defined it) are reused as the second hop.
No guard prevents anchor-rel1-bridge-rel2-target where `rel2` is a property
of the bridge that semantically belongs to the same fact as `rel1`.

**Fix:** In `_multihop` (`generator.py`), filter out paths where `bridge_type` is not PERSON/PER — agent role nouns only make sense when the bridge is the person the question is about, not an ORG whose actions get misattributed to the anchor. Also filter paths where `target_type` is None (vague nouns like "mill", "part").

**Status:** Fixed — `_multihop` in `generator.py`

---

## #3 — Possessive "Nokia 's" extracted as entity

**Example:**
```
"Did Microsoft acquire Nokia 's?" → Yes
"What did Microsoft acquire?" → Nokia 's
```
The NER picks up `Nokia's` from "Nokia's mobile phone business" as an ORG entity,
including the possessive marker.

**Root cause:** Stanza tokenises `Nokia` and `'s` as separate tokens; the entity
span includes both, and the extractor joins them with a space → `"Nokia 's"`.

**Fix:** In `_build_entity_map` (`extractor.py`), skip tokens where `word.text == "'s"` when joining the entity span into a normalised name.

**Status:** Fixed — `_build_entity_map` in `extractor.py`

---

## #4 — Unresolved common-noun anaphors produce junk questions

**Examples:**
```
"What did that drive?" → expansion        ("that" = Nokia Mobile Phones division)
"Who viewing champion?" → Pekka Lundmark  (malformed + wrong referent)
"What did company lead?" → transition     ("company" = Nokia, article dropped)
```
The coreference resolver only handles personal pronouns and partial proper-name
matches. Common-noun anaphors (`that`, `the company`, `the division`,
`the headquarters`, `the government`) fall through unresolved, producing
low-quality or grammatically broken questions.

**Fix:** In `_single_edge` and `_yes_no` (`generator.py`), skip edges where either `src_type` or `dst_type` is `None` — vague common nouns have no entity type, so this blocks the entire family of junk questions at the source.

**Status:** Fixed — `_single_edge` and `_yes_no` in `generator.py`

---

## #6 — MONEY/price objects generate backwards subject-mask questions

**Examples:**
```
"What acquired 15.6 billion euros?" → Nokia
"What acquired approximately 7.2 billion dollars?" → Microsoft
```
Nokia paid 15.6B euros; Microsoft paid 7.2B dollars. The questions imply the
companies *received* money, which is the opposite of what happened.

**Root cause:** `acquire_for` relations have a MONEY-typed object. Subject-mask
questions flip subject and object, producing "What acquired $X?" where $X is a
price — semantically inverted.

**Fix:** Added `MONEY`, `CARDINAL`, `PERCENT`, `QUANTITY` to `_SKIP_MASK_SUBJECT_TYPES` in `generator.py`.

**Status:** Fixed — `_SKIP_MASK_SUBJECT_TYPES` in `generator.py`

---

## #7 — Incomplete "become" questions

**Example:**
```
"When did Rajeev Suri become?" → 2014
```
`become` is copula-like and requires a predicative complement ("became CEO").
Without it the question is grammatically broken.

**Root cause:** The extractor captures `Rajeev Suri → become_in → 2014` from
"became the CEO of Nokia in 2014", dropping the predicate nominal "CEO of Nokia".

**Fix:** Added `become` and `be` to `_SKIP_VERB_BASES` in `generator.py`; both `_single_edge` and `_yes_no` skip any edge whose verb base is in this set.

**Status:** Fixed — `_SKIP_VERB_BASES` in `generator.py`

---

## #8 — `replace_as` produces misleading object questions

**Examples:**
```
"What did Pekka Lundmark replace?" → Nokia
"Who replaced Nokia?" → Pekka Lundmark
```
From "Pekka Lundmark replaced Rajeev Suri *as CEO of Nokia*". The extractor
creates `Pekka Lundmark → replace_as → Nokia` treating Nokia as the object of
"replace as", so questions imply Nokia itself was replaced.

**Root cause:** The `as` oblique of the predicate nominal ("as CEO of Nokia") is
extracted as a prepositional object of `replace`, conflating the replaced person
(Rajeev Suri) with the role context (Nokia/CEO).

**Fix:** In the mask loop inside `_single_edge` (`generator.py`), skip both mask directions when `relation.endswith("_as")`.

**Status:** Fixed — `_single_edge` in `generator.py`

---

## #9 — Comparison questions conflate full vs. partial acquisitions

**Examples:**
```
"Which was acquired earlier, Mobira or Nokia?" → Mobira
"Which was acquired later, Alcatel-Lucent or Nokia?" → Alcatel-Lucent
```
Microsoft acquired Nokia's *mobile phone business* (a division), not Nokia the
corporation. Comparing this with Nokia's own acquisitions of Mobira and
Alcatel-Lucent is misleading — different actors, different scopes.

**Root cause:** The comparison generator groups all entities that share an
"acquired" verb regardless of who the acquirer is (Nokia vs Microsoft).

**Fix:** Changed `by_verb` key from `verb_base` to `(src, verb_base)` in `_comparison` (`generator.py`), so only entities acted on by the **same subject** are ever compared.

**Status:** Fixed — `_comparison` in `generator.py`

---

## #10 — Partial name treated as separate ORG node, not merged with full-name PERSON

**Example:**
```
"What did Idestam establish?" → Nokia
"What established Nokia?" → Idestam
```
`Idestam` (from "Idestam later established…") and `Fredrik Idestam` (from "founded
by Fredrik Idestam") are two separate KG nodes. Stanza classifies bare `Idestam`
as ORG (no first name = ambiguous), so it passes the entity-type filter and
generates a "What" question instead of "Who". The two nodes should be merged and
typed as PERSON.

**Root cause:** Two issues compound:
1. NER assigns ORG to `Idestam` in isolation (no first name context)
2. The partial-name coref heuristic in `coref.py` matches on last-token overlap
   but only updates the `coref_distance` field — it does not unify the two nodes
   into one canonical entity in the KG

**Why it's non-trivial:** Fixing requires either (a) post-NER entity typing
correction using a name gazetteer/heuristic, or (b) proper entity unification in
the graph (merging `Idestam` into `Fredrik Idestam` and re-pointing all edges),
which is a structural change to both `coref.py` and `graph.py`.

**Status:** Parked — known limitation, not blocking

---

## #11 — Morphological mismatch in Word2Vec verb replacement

**Examples:**
```
"When did FDA introducing?" → 2021     (should be "introduce")
"When did FDA approved?"   → 2020     (should be "approve")
"What happen in 2019?"                (should be "happened")
```

**Root cause:** Word2Vec replacement is purely token-level — the synonym string is substituted verbatim into the question. No morphological inflection is applied after replacement. Two sub-cases:

1. **Base-form slot after "did"**: In "What did FDA introduce?", Word2Vec finds "introducing" or "approved" as neighbors of "introduce". These are morphologically wrong in a `did + BASE` context.
2. **Tense replacement for anchor verbs**: "What happened in 2019?" — Word2Vec finds "happen" as a neighbor of "happened". Substituting it produces the uninflected base form without tense.

**Fix:** After substituting a synonym, lemmatize the W2V result (Stanza), then re-inflect to the original token's Penn Treebank xpos using `pyinflect`. E.g., "occurring" → lemma "occur" → `getInflection("occur", "VBD")` → "occurred".

**Status:** Fixed — `variant_processor.py` (`_inflect_to_form`, `_lemmatize`)

---

## #5 — Missing preposition in passive subject-mask questions

**Example:**
```
"What was Nokia founded?" → mill
```
Should be `"What was Nokia founded as?"` — the `as` preposition from the
`found_as` relation is dropped when building the passive subject-mask template.

**Root cause:** `_en_single` passive subject-mask branch uses
`f"{qw} was {vt} {prep} {subject}?"` when `prep` exists, but when `mask="object"`
with a prepositional relation (`found_as`), the prep is not appended.

**Fix:** In `_en_single` (`templates.py`), append the preposition in passive object-mask questions unless it's already semantically encoded in the question word (e.g. "When" implies "in/at/on"). Added `_QW_IMPLIED_PREPS` dict to make the rule explicit.

**Status:** Fixed — `_en_single` in `templates.py`
