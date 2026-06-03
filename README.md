# Knowledge Graph Question Generation

Generate CEFR-levelled questions from English or Finnish text passages. The pipeline extracts a knowledge graph, generates questions from graph structure, and estimates difficulty along two independent axes.

## Quick Start

```bash
# CLI report
make script INPUT=data/en/demo.txt OUTPUT=data/output/report.md

# Interactive UI
make streamlit
```

---

## Architecture

```
Input passage
      ↓
[1] NER + dependency parsing → Triple objects  (knowledge_graph/extractor.py)
      ↓
[2] Heuristic coreference resolution           (knowledge_graph/coref.py)
      ↓
[3] Build directed graph (nodes=entities, edges=relations)  (knowledge_graph/graph.py)
      ↓
[4] Generate candidate questions per type      (question_generation/generator.py)
      ↓
[5] Estimate difficulty (text-side + question-side)  (question_generation/difficulty/)
      ↓
Output: questions with text_difficulty, question_difficulty on preA1–C2+ scale
```

---

## Step 1 — Text → Knowledge Graph

### NER + Relation Extraction

Uses Stanza (tokenize, NER, POS, lemma, depparse). Each sentence produces `Triple` objects:

```
"Nokia was founded in 1865 in Tampere by Fredrik Idestam."

Triples:
  (Nokia, found_in,  1865)            is_passive=True
  (Nokia, found_in,  Tampere)         is_passive=True
  (Nokia, found_by,  Fredrik Idestam) is_passive=True
```

Handles: active and passive voice, coordinated subjects/objects, Finnish passive without explicit subject, copula sentences, Finnish agentive postposition (`Idestamin toimesta`), Finnish case lemmatisation.

### Triple Fields (signals carried to difficulty)

| Field | Description |
|---|---|
| `sentence_idx` | Which sentence the triple came from |
| `source_depth` | Clausal embedding depth of the source sentence |
| `answer_depth` | Dependency tree depth of the object token |
| `is_passive` | Passive voice extraction |
| `coref_distance` | Sentences between reference and antecedent (set by coref step) |

### Coreference Resolution

Heuristic resolution tracks sentence indices. Sets `coref_distance = sentence_idx − antecedent_sentence_idx` on the resolved triple, so later difficulty scoring can penalise long-range references.

---

## Step 2 — Question Generation

Questions are organised into two categories based on how they are constructed from the knowledge graph.

---

### Category 1 — Single node questions

One edge, one masked node. The answer is a single entity or a list of entities.

**Difficulty controller: hop count** — 1-hop asks directly about a single edge. 2-hop (chain) requires reasoning through a bridge entity ("What did the founder of Nokia establish?"). Higher hop count = harder reasoning path.

| Question type | Answer node type | Siblings? | Example | Answer |
|---|---|---|---|---|
| wh- object | DATE | no | "When did Nokia acquire Mobira?" | entity |
| wh- object | PERSON | no | "Who founded Nokia?" | entity |
| wh- object | GPE/LOC | no | "Where was Nokia founded?" | entity |
| wh- object | ORG/other | no | "What did Nokia acquire?" | entity |
| wh- object | ORG/other | yes | "What organizations did Nokia acquire?" | list of entities |
| wh- subject | ORG | no | "What acquired Mobira?" | entity |
| wh- subject | PERSON | no | "Who founded Nokia?" | entity |
| yes/no | any | — | "Did Nokia acquire Mobira?" | bool |
| which | ORG/PERSON/GPE | yes | "Which org did Nokia acquire in 1982?" | entity |
| temporal comparison | DATE | yes | "Which was acquired earlier, A or B?" | entity |
| number comparison | MONEY/CARDINAL | yes | "Which acquisition cost more, A or B?" | entity |

**Notes:**
- Answer node type determines the question word: DATE→When, PERSON→Who, GPE/LOC→Where, else→What
- When siblings exist (same subject + verb + answer type), wh- object promotes automatically to the aggregation template with a type noun ("What **organizations**…") and `answer_list`
- Chain questions are wh- questions with hop count ≥ 2, not a separate type. The bridge entity must be a PERSON.

---

### Category 2 — Anchor node questions

One anchor node, all connected event clusters. The answer is a list of event sentences.

**Difficulty controllers:**
- **Fact count** — how many events the learner must recall (more = harder)
- **Traversal depth** — depth 1: direct edges only; depth 2: also events from anchor's neighbors

**Construction strategy (group-first):**
1. Cluster edges by `(subject, verb_base)` to form event groups
2. For each group, identify the anchor node that grounds the cluster (subject, DATE, GPE, or PERSON in context)
3. Render each event cluster as an event sentence
4. Generate the question based on anchor type

| Question type | Anchor node type | Example | Answer |
|---|---|---|---|
| temporal | DATE | "What happened in 1982?" | list of event sentences |
| location | GPE/LOC | "What happened in Tampere?" | list of event sentences |
| person | PERSON | "What did Fredrik Idestam do?" | list of event sentences |
| org | ORG | "What has Nokia been involved in?" | list of event sentences |

Templates are in `question_generation/templates.py`. Finnish is supported for Category 1 `object`, `subject` types.

---

## Step 3 — Difficulty Estimation

### Two-Axis Model

Each question gets two independent CEFR-level estimates:

- **Text-side difficulty** — how hard is it to find / parse the answer in the source passage?
- **Question-side difficulty** — how complex is the question form itself?

Both axes use an 8-level scale: `preA1 · A1 · A2 · B1 · B2 · C1 · C2 · C2+`

### Text-Side: Three Independent Dimensions

Text-side difficulty combines three components, each normalised to [0, 1]:

```
text_score = 0.50 × local + 0.25 × readability + 0.25 × distractor
```

#### 1. Local Extraction (weight 0.50)

How hard it is to locate and parse the answer within its sentence.

| Signal | Description | Max raw |
|---|---|---|
| Hop count | 1-hop=0, 2-hop=6, 3-hop=12, 4-hop=18 | 18 |
| Source clause depth | Clausal embeddings (`relcl`, `advcl`, …) | 3 |
| Answer tree depth | Dependency depth of the answer token | 2 |
| Passive voice | Passive construction extraction | 2 |
| Coreference distance | Sentences between pronoun and antecedent | 3 |

**Total max = 28.** Normalised: `local = raw_sum / 28`

#### 2. Global Readability — LIX (weight 0.25)

How hard the passage is overall to read, using the LIX (Läsbarhetsindex) formula. LIX is language-agnostic and works for both English and Finnish (unlike Flesch-Kincaid, which is calibrated to English syllable patterns).

```
LIX = words/sentences + long_words×100/words
```

`long_words` = words longer than 6 characters (after stripping trailing punctuation). Normalised from the practical range [20, 65] → [0, 1]:

```
readability = clamp((LIX − 20) / 45, 0, 1)
```

| LIX | Typical text |
|---|---|
| 20–30 | Children's books, simple news |
| 30–40 | Popular press |
| 40–50 | Standard non-fiction |
| 50–60 | Academic / technical |
| 60+ | Legal / scientific |

#### 3. Distractor Density (weight 0.25)

How many plausible competing answers exist in the knowledge graph, making it harder to select the right one.

Same-type entities of the answer node reachable in the KG, weighted by graph distance using exponential decay:

```
weight = 0.5 ^ distance        (dist=1: 0.5,  dist=2: 0.25,  dist=3: 0.125 …)
raw    = sum of weights within 6 hops
distractor = raw / (1 + raw)   (smooth mapping [0, ∞) → [0, 1), no hard cap)
```

Close same-type entities (topically related) contribute strongly; distant unrelated ones contribute near-zero. Only same-type siblings (same subject + verb + answer type) count — cross-type objects sharing a verb base are excluded.

### Text-Side Level Thresholds

After combining the three dimensions, the score is mapped to a level:

| Combined score | Level |
|---|---|
| ≥ 0.75 | C2+ |
| ≥ 0.55 | C2 |
| ≥ 0.40 | C1 |
| ≥ 0.25 | B2 |
| ≥ 0.20 | B1 |
| ≥ 0.12 | A2 |
| ≥ 0.04 | A1 |
| ≥ 0.00 | preA1 |

### Question-Side Signals

| Signal | Description | Max raw |
|---|---|---|
| Form score | yesno=0, wh-=2, comparison/which=4, aggregation=5, chain=6 | 6 |
| Passive voice | Passive question construction | 2 |
| Nominalization | Chain uses noun phrase ("the founder of X") | 2 |

**Total max = 10.** Normalised → level mapping with question-specific thresholds.

### Calibration Examples

| Question | Text-side | Question-side |
|---|---|---|
| "What did Nokia acquire?" (1-hop active) | local≈0 → varies by LIX+distractor | A1 |
| "Who was Nokia founded by?" (1-hop passive) | local=2/28 | B1 |
| "Which org did Nokia acquire in 2016?" | — | B1 |
| 2-hop simple chain | local=6/28 | C1 |
| 2-hop passive chain | local=8/28 | C2 |

Implementation: `question_generation/difficulty/base.py` (ABC + normalisation) and `question_generation/difficulty/rule_based.py` (concrete scoring).

---

## File Structure

```
knowledge_graph/
├── extractor.py      NER + dependency parsing → Triple objects
├── graph.py          KnowledgeGraph (NetworkX MultiDiGraph) + multihop_paths
└── coref.py          Heuristic pronoun/partial-name coreference + coref_distance

question_generation/
├── generator.py      QuestionGenerator — orchestrates all question types
├── models.py         Question dataclass (text, answer, answer_list, difficulty axes, …)
├── templates.py      Question string builders (EN + FI)
└── difficulty/
    ├── base.py       DifficultyEstimator ABC + CEFR thresholds + LEVELS / LEVEL_ORDER
    └── rule_based.py RuleBasedEstimator (text_max=28, question_max=10)

scripts/
├── question_generation_script.py    CLI: .txt → Markdown report
└── question_generation_streamlit.py Interactive Streamlit UI

data/
├── en/demo.txt       English demo passage (Nokia)
├── fi/demo.txt       Finnish demo passage (Nokia)
└── output/           Generated reports (git-ignored)
```

---

## Running

```bash
# Streamlit UI (interactive)
make streamlit

# CLI: generate a Markdown report
make script INPUT=data/en/demo.txt
make script INPUT=data/en/demo.txt OUTPUT=data/output/nokia_en.md

# Direct script invocation
python scripts/question_generation_script.py data/en/demo.txt --output report.md --max-questions 20
```

---

## Known Limitations

- **Relation labels** are shallow (verb lemma + preposition only); complex semantic relations may be missed
- **Finnish question types** — only `object`, `subject`, and `chain` forms; yes/no, comparison, which, aggregation are English-only
- **Coreference heuristic** fails with multiple persons of the same gender in the same passage
- **Aggregation deduplication** — one question per (subject, verb) pair; different phrasings of the same group are collapsed
- **0-hop questions** (entity-type definitions, e.g. "What is Nokia?") are not generated — too trivial for the intended difficulty range
- **IRT calibration** — empirical β estimates require real learner response data (Phase 2 goal)
