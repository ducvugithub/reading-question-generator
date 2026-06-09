# Knowledge Graph Question Generation with Difficulty Variants

Generate CEFR-levelled questions from English or Finnish text passages. The pipeline extracts a knowledge graph, generates questions, estimates difficulty, and creates **difficulty variants** by replacing verbs with semantically similar synonyms at different CEFR levels using Word2Vec.

## Quick Start

```bash
# Generate questions with difficulty variants
make questions INPUT=data/en/microsoft.txt TARGET_CEFR=C1

# Specify output file
make questions INPUT=data/en/microsoft.txt OUTPUT=my_report.md TARGET_CEFR=B1

# Print to terminal (verbose)
make questions INPUT=data/en/microsoft.txt VERBOSE=1

# Interactive UI (legacy)
make app
```

---

## Complete Pipeline

```
INPUT: Text passage
   ↓
[PHASE 1: KNOWLEDGE GRAPH EXTRACTION]
   ├─ [1] NER + dependency parsing → Triple objects
   ├─ [2] Heuristic coreference resolution  
   └─ [3] Build directed graph (nodes=entities, edges=relations)
   ↓
[PHASE 2: QUESTION GENERATION & SCORING]
   ├─ [4] Generate candidates across 3 tiers:
   │    ├─ Retrieval tier: single-hop single-node & subgraph candidates
   │    ├─ Inferential tier: multi-hop chain single-node & subgraph candidates
   │    └─ Critical tier: complex reasoning single-node & subgraph candidates (TBD)
   │    Then: Filter by difficulty → Deduplicate → Sort by difficulty
   └─ [5] Estimate difficulty (text-side + question-side)
   ↓
[PHASE 3: DIFFICULTY VARIANTS]
   ├─ [6a] Extract main verb from question
   ├─ [6b] Lemmatize verb → base form (e.g., "founded" → "find")
   ├─ [6c] Look up lemma frequency (wordfreq or local vocab)
   ├─ [6d] CEFR classification via percentile bins (A1=30%, A2=25%, B1=20%, B2=15%, C1=7%, C2=3%)
   ├─ [6e] Find Word2Vec synonym at target CEFR level
   ├─ [6f] Replace verb while preserving sentence structure
   └─ [6g] Deduplicate (skip if text unchanged)
   ↓
OUTPUT: Questions with variants at A1–C2 levels
```

---

## Phase 1 — Knowledge Graph Extraction

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

## Phase 2 — Question Generation & Scoring

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

### Difficulty Estimation

#### Four-Component Scoring Model

Overall difficulty combines four independent components, each normalised to [0, 1]:

```
combined = (score_type + score_local + score_vocab + score_readability) / 4
```

This simple average is then mapped to the 8-level CEFR scale (preA1 → C2+).

##### Component 1: Question Form Complexity (score_type)

How complex the question form is itself:

| Question type | Score |
|---|---|
| yes/no | 0.0 (simplest) |
| wh- questions | 0.2–0.4 |
| which/comparison | 0.4 |
| aggregation | 0.5 |
| chain (multi-hop) | 0.6 (hardest) |

##### Component 2: Answer Extraction Difficulty (score_local)

How hard it is to locate and parse the answer within the knowledge graph and sentence:

| Signal | Normalisation |
|---|---|
| Source clause depth | Clausal embeddings (`relcl`, `advcl`, …): depth ≤ 3 |
| Answer tree depth | Dependency depth of answer token: depth ≤ 2 |
| Coreference distance | Sentences between pronoun and antecedent: distance ≤ 3 |

**Formula:** `score_local = (depth + answer + coref) / 3`

##### Component 3: Question Phrasing Complexity (score_vocab)

How complex the question phrasing is:

| Feature | Score |
|---|---|
| Passive voice construction | +0.5 |
| Nominalization (chain uses noun phrase) | +0.5 |
| **Max:** | 1.0 |

##### Component 4: Passage Readability (score_readability)

How hard the passage is overall to read, using a **finetuned BERT model** for CEFR classification:

- **Primary method:** [`AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2`](https://huggingface.co/AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2) (HuggingFace transformer-based CEFR classifier, when available)
- **Fallback method:** LIX (Läsbarhetsindex) formula for unknown languages

**Fallback formula (LIX):**

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

#### CEFR Level Thresholds

After combining the four components, the averaged score [0, 1] is mapped to an 8-level CEFR scale using evenly-spaced thresholds (1/7 intervals):

| Combined score | Level |
|---|---|
| ≥ 0.929 | C2+ |
| ≥ 0.786 | C2 |
| ≥ 0.643 | C1 |
| ≥ 0.500 | B2 |
| ≥ 0.357 | B1 |
| ≥ 0.214 | A2 |
| ≥ 0.071 | A1 |
| ≥ 0.000 | preA1 |

**Implementation:** `question_generation/difficulty/base.py` (ABC + normalisation) and `question_generation/difficulty/rule_based.py` (concrete scoring).

---

## Difficulty Variant Generation (Phase 3)

After questions are generated and difficulty is estimated, **difficulty variants** are created by replacing main verbs with semantically similar synonyms at different CEFR levels.

### Pipeline

1. **Lemmatization** — Extract main verb from question and get base form (e.g., "founded" → "find")
2. **Frequency Lookup** — Look up lemma frequency using wordfreq (English) or local vocab (Finnish)
3. **CEFR Classification** — Map frequency to CEFR level using **percentile-based bins**:
   - **A1** (easiest): Top 30% most frequent words
   - **A2**: 30-55% 
   - **B1**: 55-75%
   - **B2**: 75-90%
   - **C1**: 90-97%
   - **C2** (hardest): 97-100% least frequent

4. **Word2Vec Synonym Lookup** — Find Word2Vec neighbors at target CEFR level using:
   - Google Word2Vec (English) or FastText-wiki (Finnish)
   - Similarity threshold: 0.5+
   - Accept exact CEFR match or ±2 levels if high similarity (>0.55)

5. **Verb Replacement** — Replace main verb with synonym while preserving sentence structure

6. **Deduplication** — Only keep variant if text actually changed; skip duplicates

### Example

**Original question (A1 difficulty):**
```
"When was Microsoft founded?"
```

**Generated variants:**
- **A1**: "When was Microsoft found?" (simplest form)
- **A2**: "When was Microsoft started?" (more common synonym)
- **B1**: "When was Microsoft created?" (less common)
- **B2**: "When was Microsoft established?" (rare)
- **C1**: "When was Microsoft instituted?" (very rare)

### Limitations

⚠️ **Word2Vec coverage is incomplete:**
- Only works for verbs that have high-similarity neighbors at different CEFR levels
- Some verbs (e.g., domain-specific terminology) may not have good synonyms
- When no variant is found at a target level, the original verb is kept
- If a question cannot generate any unique variants, only the original question is kept

---

## File Structure

```
knowledge_graph/
├── extractor.py      NER + dependency parsing → Triple objects
├── graph.py          KnowledgeGraph (NetworkX MultiDiGraph) + multihop_paths
└── coref.py          Heuristic pronoun/partial-name coreference + coref_distance

question_generation/
├── generator.py              QuestionGenerator (+ create_variants param)
├── models.py                 Question dataclass (text, answer, answer_list, difficulty, …)
├── templates.py              Question string builders (EN + FI)
├── variant_processor.py      Difficulty variant generation (NEW!)
│                             ├─ Verb lemmatization + extraction
│                             ├─ Question variant creation
│                             └─ Deduplication
├── word2vec_variants.py      Word2Vec synonym lookup (NEW!)
│                             ├─ Percentile-based CEFR binning
│                             ├─ Word2Vec model loading (gensim)
│                             └─ Frequency lookup (wordfreq + local files)
└── difficulty/
    ├── base.py               DifficultyEstimator ABC + CEFR thresholds
    └── rule_based.py         RuleBasedEstimator (text_max=28, question_max=10)

scripts/
├── question_generation_script.py  CLI (legacy)
├── question_generation_streamlit.py Interactive UI (legacy)
└── demo_variants.py          CLI variant generation (NEW!)

data/
├── en/
│   ├── microsoft.txt         English demo passage
│   ├── cern.txt
│   └── ... (vocab/ git-ignored)
├── fi/
│   ├── demo.txt              Finnish demo passage
│   └── ... (vocab/ git-ignored)
└── output/                   Generated reports (git-ignored)
```

---

## Running

```bash
# Generate questions with difficulty variants (default: auto-save to data/output/)
make questions INPUT=data/en/microsoft.txt TARGET_CEFR=C1

# Different target CEFR level
make questions INPUT=data/en/microsoft.txt TARGET_CEFR=B1

# Print to terminal (verbose)
make questions INPUT=data/en/microsoft.txt VERBOSE=1

# Custom output file
make questions INPUT=data/en/microsoft.txt OUTPUT=my_report.md TARGET_CEFR=B1

# Interactive UI
make app
```

**Output:** Markdown report with timestamp, input text, questions table, and knowledge graph statistics.

---

## Known Limitations

### Question Generation
- **Relation labels** are shallow (verb lemma + preposition only); complex semantic relations may be missed
- **Finnish question types** — only `object`, `subject`, and `chain` forms; yes/no, comparison, which, aggregation are English-only
- **Coreference heuristic** fails with multiple persons of the same gender in the same passage
- **Aggregation deduplication** — one question per (subject, verb) pair; different phrasings are collapsed
- **0-hop questions** (entity-type definitions, e.g. "What is Nokia?") are not generated — too trivial
- **Entity extraction noise** — NER and extraction may include non-entities (company, domain, etc. with `None` type)

### Abstract Text — 0 Questions Generated

The pipeline is **entity-centric**: it requires named entities (PERSON, ORG, DATE, GPE) as anchors to build questions. Abstract, argumentative, or conceptual passages (e.g. academic essays, opinion pieces) produce triples where all entity types are `None`, resulting in 0 questions generated.

**Example of a failing passage:**
```
"The proliferation of artificial intelligence has fundamentally altered clinical practice.
 Critics contend that algorithmic diagnoses lack the contextual sensitivity…"
```
All extracted triples have `subj_type=None, obj_type=None` → no question templates match.

**Workaround (short-term):** Ensure the passage contains concrete named entities — real organizations, people, dates, and locations — even when the topic is abstract. For example, grounding the same topic around "DeepMind developed X in 2019" enables question generation.

**Future solution — LLM fallback:**
When KG extraction yields fewer than N typed-entity triples, fall back to an LLM-based question generator that can reason over abstract concepts directly. The LLM output would be merged with any KG-derived questions, with difficulty estimated via the same readability + question-form scoring model. This keeps the KG pipeline as the primary path and only pays the LLM cost for genuinely abstract passages.

### Difficulty Variant Generation
- **Word2Vec coverage is incomplete** — Not all verbs have semantic neighbors at every CEFR level
  - If no synonym found at target level, the original verb is kept
  - Some questions may not have variants at all difficulty levels (e.g., all A1)
  - This is acceptable — better one good question than multiple duplicates with false difficulties
  - **Google News corpus bias** — Google Word2Vec is trained on news text, so neighbors reflect news/sports context. For example, `mark` (B1) returns `eclipsing`, `surpassing`, `milestone` (all C2) rather than simpler synonyms like `happen` or `occur`. General-purpose verbs in non-news contexts are poorly served.
  
- **Percentile-based CEFR binning** assumes word frequency correlates with CEFR level
  - Works well for common words (common → easy)
  - Limited coverage for specialized/technical vocabulary
  - Finnish frequency data requires local vocabulary files (git-ignored)

- **Lemmatization dependency** — Accuracy depends on Stanza NLP quality for the language
  - Edge cases (irregular verbs, phrasal verbs) may not lemmatize correctly

### Other
- **IRT calibration** — empirical β estimates require real learner response data (Phase 2 goal)
