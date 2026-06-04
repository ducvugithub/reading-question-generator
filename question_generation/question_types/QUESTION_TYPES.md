# Question Types

Generated from a Knowledge Graph (KG) extracted via Stanza NLP. Each question type
maps to a `masked` field on the `Question` dataclass and appears as a column in
the CLI report.

---

## Difficulty model

Every question gets four component scores, each in `[0, 1]`:

| Component | What it measures |
|---|---|
| `s_type` | Question-form complexity (see type scores below) |
| `s_local` | Answer-extraction difficulty: clause depth + answer depth + coreference distance |
| `s_vocab` | Phrasing complexity: 0.5 if passive voice, 0.5 if chain nominalization |
| `s_read` | Passage readability via ModernBERT CEFR classifier (fallback: LIX) |

**Combined score** → CEFR level via evenly-spaced thresholds aligned to ModernBERT's
`idx / 7` scale:

```
combined = (s_type + s_local + s_vocab + s_read) / 4
```

| Range | Level |
|---|---|
| ≥ 0.929 | C2+ |
| ≥ 0.786 | C2  |
| ≥ 0.643 | C1  |
| ≥ 0.500 | B2  |
| ≥ 0.357 | B1  |
| ≥ 0.214 | A2  |
| ≥ 0.071 | A1  |
| < 0.071 | preA1 |

Some types bypass the formula and use **rule-based difficulty** (marked ★ below)
because their difficulty depends on event count or structural complexity that the
formula cannot capture.

### `s_type` base scores

| Type | Score |
|---|---|
| `true_claim`, `false_claim`, `object`, `subject` | 0.20 |
| `subgraph` | 0.25 |
| `aggregation`, `count` | 0.30 |
| `comparison`, `which`, `chain`, `bridge` | 0.35 |
| `chain_subgraph` | 0.38 |

`chain` also gets a hop bonus: +0.20 for 3-hop, +0.40 for 4-hop.

---

## Retrieval tier

Questions whose answer is directly stated in the passage.

### `wh-` (object / subject masking)
Mask the object or subject of a KG edge and ask for it.

- *"When did Nokia acquire Mobira?"* → 1982
- *"Who founded Nokia?"* → Fredrik Idestam
- *"What was founded by Fredrik Idestam?"* → Nokia (subject mask)

**Difficulty:** formula → typically A1 (active) or A2 (passive).

---

### `true_claim`
Confirm a true fact as a yes/no question.

- *"Did Nokia acquire Mobira?"* → Yes
- *"Was Nokia founded in 1865?"* → Yes

**Difficulty:** formula → typically A2.

---

### `false_claim`
A yes/no question built around a **wrong** entity — a same-type distractor drawn from
the KG. The answer is always *No*; the source sentence shows the contradicted fact.
Distractors are ranked by KG degree so only prominent entities are substituted.

- *"Was Nokia founded by Jorma Ollila?"* → No *(was founded by Fredrik Idestam)*
- *"Did Pekka Lundmark replace Jorma Ollila?"* → No *(replaced Rajeev Suri)*
- *"Was The Nokia Research Center established in 2014?"* → No *(established in 1986)*

**Difficulty:** ★ rule-based → **B1** (requires knowing the correct fact to detect the error).

> **Note:** Together with a planned `not_given` type (information absent from the passage),
> these form the IELTS reading comprehension trinity: `true_claim` / `false_claim` / `not_given`.

---

### `which`
Ask which entity satisfies a relation with an additional constraint (e.g. a date hint).

- *"Which organization did Nokia acquire in 1982?"* → Mobira

**Difficulty:** formula → typically A2.

---

### `count`
Ask how many entities satisfy a relation.

- *"How many organizations did Nokia acquire?"* → 2

**Difficulty:** formula → typically A2.

---

### `aggregation`
Ask for the full set of entities satisfying a relation.

- *"What organizations did Nokia acquire?"* → Mobira / Alcatel-Lucent

**Difficulty:** formula → typically A2.

---

## Inferential tier

Questions that require combining or traversing multiple KG facts.

### `subgraph`
Ask about **1–2 facts** anchored to a single entity node or a subject+object pair.
Difficulty is rule-based on event count, not the formula.

**Single anchor** — all edges attached to an entity:
- *"What happened in 1982?"* → Nokia acquired Mobira in 1982 …
- *"What did Jorma Ollila do?"* → Jorma Ollila led Nokia from 1992 …

**Paired anchor** — edges shared by a subject anchor and an object anchor:
- *"What did Nokia do in 1982?"* → Nokia acquired Mobira in 1982 …
- *"What did Jorma Ollila do in 1992?"* → Jorma Ollila led Nokia from 1992 …

**Difficulty:** ★ rule-based → **A1** (1 fact) · **A2** (2 facts).

---

### `comparison`
Compare two entities along a shared relation (earlier/later, more/less).

- *"Which was acquired earlier, Mobira or Alcatel-Lucent?"* → Mobira

**Difficulty:** formula → typically A2.

---

### `chain`
Two-hop traversal through a PERSON/PER bridge node, using a nominalized chain phrase.

- *"What did the founder of Nokia acquire?"* → Mobira
- *"When did the leader of Nokia transform the company?"* → 1998

**Difficulty:** formula → typically **A2** (s_vocab no longer includes chain bonus,
s_type = 0.35 captures multi-hop complexity).

---

### `chain_subgraph`
Chain question anchored to a time or place — combines two-step inference with a
temporal/spatial constraint. Only fires when the bridge node's source sentence
also names the anchor entity (prevents cross-sentence spurious combinations).

- *"What did the leader of Nokia do in 1992?"* → Jorma Ollila led Nokia from 1992 …
- *"What did the acquirer of Nokia do in 2014?"* → Microsoft acquired Nokia's mobile …

**Difficulty:** formula → typically **B1**
(chain nominalization in s_vocab + moderate s_local).

---

### `bridge`
Two-hop path A → B → C where the question asks for the intermediate node B.
Only fires when the A→B and B→C source sentences are different (ensures B
genuinely bridges two separate facts). Passive out-edges are skipped to keep
templates grammatical.

- *"Which organization did Jorma Ollila lead that acquired Mobira?"* → Nokia
- *"Which organization did Microsoft acquire that partnered with Siemens?"* → Nokia

**Difficulty:** ★ rule-based → **C1** (requires holding two separate facts and
identifying the connecting entity).

---

## Adding a new question type

1. Create `retrieval/<name>.py` or `inferential/<name>.py` implementing `QuestionType`.
2. Register in the corresponding `__init__.py` `TYPES` list.
3. Add `"<name>": <score>` to `_TYPE_SCORE` in `difficulty/rule_based.py`.
4. Add `"<name>": "<display-name>"` to `_FORM_NAMES` in `scripts/question_generation_script.py`.
5. If difficulty bypasses the formula, set `difficulty=` directly in the `Question`
   constructor and document it as ★ rule-based above.
