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

Six question types are generated from graph structure:

| Type (`masked`) | Example | Graph pattern |
|---|---|---|
| `object` (wh-) | "What did Nokia acquire?" | single edge, mask object |
| `subject` (wh-) | "Who founded Nokia?" | single edge, mask subject |
| `yesno` | "Did Nokia acquire Alcatel-Lucent?" | single edge, binary |
| `comparison` | "Which was founded earlier, Nokia or Alcatel-Lucent?" | two entities + date edges |
| `which` | "Which company did Nokia acquire in 2016?" | multiple objects, same verb + hint |
| `aggregation` (list) | "What organizations did Nokia acquire?" | 2+ objects, same subject+verb |
| `chain` (multi-hop) | "What did the founder of Nokia discover?" | 2-hop path via bridge entity |

Templates are in `question_generation/templates.py`. Finnish is supported for `object`, `subject`, and `chain` types.

### Aggregation Questions

Aggregation questions are generated when the same subject performs the same verb on 2+ objects of the same entity type (scattered across different sentences in the passage). The answer is the full set: `["Alcatel-Lucent", "Mobira", ...]`.

---

## Step 3 — Difficulty Estimation

### Two-Axis Model

Each question gets two independent CEFR-level estimates:

- **Text-side difficulty** — how hard is it to find / parse the answer in the source passage?
- **Question-side difficulty** — how complex is the question form itself?

Both axes use an 8-level scale: `preA1 · A1 · A2 · B1 · B2 · C1 · C2 · C2+`

### Text-Side Signals

| Signal | Source | Max raw |
|---|---|---|
| Hop count | 1-hop=0, 2-hop=6, 3-hop=12, 4-hop=18 | 18 |
| Source clause depth | Clausal embeddings in sentence (`relcl`, `advcl`, …) | 3 |
| Answer tree depth | Dependency depth of the answer token | 2 |
| Passive voice | Triple extracted via passive construction | 2 |
| Coreference distance | Sentences between pronoun and antecedent | 3 |

**Total max = 28.** Raw sum is divided by 28 → normalised score in [0, 1], then mapped to a level:

| Score | Level |
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

| Signal | Source | Max raw |
|---|---|---|
| Form score | yesno=0, wh-=2, comparison/which=4, aggregation=5, chain=6 | 6 |
| Passive voice | Passive question construction | 2 |
| Nominalization | Chain uses noun phrase ("the founder of X") | 2 |

**Total max = 10.** Same normalisation → level mapping with question-specific thresholds.

### Calibration Examples

| Question | Text score | Question score |
|---|---|---|
| "What did Nokia acquire?" (1-hop active) | 0/28 ≈ preA1 | 2/10 = A1 |
| "Who was Nokia founded by?" (1-hop passive) | 2/28 ≈ preA1 | 4/10 = B1 |
| "Which org did Nokia acquire in 2016?" (which) | — | 4/10 = B1 |
| 2-hop simple chain | 6/28 ≈ A2 | 8/10 = C1 |
| 2-hop passive chain | 8/28 ≈ B1 | 10/10 = C2 |

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
