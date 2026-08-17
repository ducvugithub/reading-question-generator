# Related Work: Question Generation Methods

Survey of published methods for automatic question generation (QG), primarily for English.
Used to position this project and identify gaps.

---

## 1. Template / Rule-based

**Heilman & Smith 2010** — *"Good Question! Statistical Ranking for Question Generation"*
- Transform declarative sentences into questions via parse tree manipulation
- Hand-crafted syntactic rules + statistical ranking
- Oldest ML approach, still used as a baseline
- Limitation: rigid, limited variety, no semantic understanding

---

## 2. Seq2seq (Passage + Answer → Question)

The dominant paradigm. Input is raw passage text + an answer span; output is a natural language question.

| Paper | Venue | Key idea |
|-------|-------|----------|
| Du et al. 2017 — *"Learning to Ask"* | ACL | First neural QG; LSTM encoder-decoder on SQuAD |
| Zhao et al. 2018 — *"Paragraph-level NQG with Maxout Pointer"* | EMNLP | Attention over full paragraph, not just answer sentence |
| Dong et al. 2019 — *UniLM* | NeurIPS | Unified LM for understanding and generation; strong QG baseline |
| Qi et al. 2020 — *ProphetNet* | EMNLP | Future n-gram prediction during training; competitive on SQuAD QG |

**Limitation:** No control over question difficulty or type. Model decides what to ask and at what level.

---

## 3. Graph-based (Graph2Seq)

Encode a graph structure of the passage (AMR, KG, dependency) via GNN, then decode a question.

| Paper | Venue | Key idea |
|-------|-------|----------|
| Pan et al. 2020 — *"Semantic Graphs for Generating Deep Questions"* | ACL | AMR graph of passage → GNN encoder → question; most cited graph QG paper |
| Bao et al. 2022 — *"Multi-hop QG with Graph Convolutional Network"* | — | KG traversal for multi-hop questions requiring cross-sentence reasoning |

**Note:** These use AMR or pre-built KGs (ConceptNet, Wikidata), not a KG extracted on-the-fly from the passage.

---

## 4. Difficulty / Controllable QG

**Directly relevant to this project.**

| Paper | Venue | Key idea |
|-------|-------|----------|
| Gao et al. 2019 — *"Difficulty Controllable QG for Reading Comprehension"* | IJCAI | Conditions T5 generation on heuristic difficulty signals (answer frequency, sentence complexity) |
| Wang et al. 2021 — *"Controllable QG"* | — | Controls question type (what / when / why / how) via prefix conditioning |

**Gap this project fills:** Gao et al. 2019 is the closest published work — difficulty-controlled QG — but uses heuristic signals, not a standardised linguistic framework (CEFR). English only, no KG extraction from text.

---

## 5. LLMs (Prompting)

| Paper / System | Key idea |
|---------------|----------|
| Brown et al. 2020 — *GPT-3* | Few-shot prompting; no fine-tuning needed |
| GPT-4 (OpenAI 2023) | Instruction prompting; currently strongest zero-shot QG |

**Limitation:** No reliable difficulty control; output quality varies; black box; expensive at scale.

---

## 6. KG → Question (from Pre-built KG)

Input is structured RDF/KG triples from a global knowledge base (Wikidata, Freebase), not extracted from a passage.

| Paper / Benchmark | Key idea |
|------------------|----------|
| Seyler et al. 2017 — *"Knowledge Questions from Knowledge Graphs"* | Wikidata triples → factoid questions |
| WebNLG Challenge (2017–2023) | RDF triples → natural language text; standard benchmark for graph-to-text |

**Difference from this project:** These use a static global KG. This project extracts a document-local KG from the reading passage on the fly.

---

## Summary: Input/Output by Method

| Method | Input | Output |
|--------|-------|--------|
| Template | Parse tree | Question |
| Seq2seq | Passage + answer span | Question |
| Graph2Seq | AMR/KG graph + anchor node | Question |
| Difficulty-controlled | Passage + answer + difficulty signal | Question |
| LLM prompting | Raw text + prompt | Question |
| KG → Question | Pre-built KG triples | Question |

---

## What This Project Does Differently

1. **Extract KG from the passage on the fly** — not a global pre-built KG
2. **CEFR level as the conditioning signal** — a standardised linguistic framework, not a heuristic
3. **Finnish** — no prior QG work exists for Finnish
4. **Multi-method under one framework** — template, seq2seq, GNN, LLM all take the same extracted KG as input

**Closest prior work:** Gao et al. 2019 (difficulty-controlled QG) + Pan et al. 2020 (graph-based QG).
The gap: no prior work combines graph-based QG with CEFR conditioning, and no work exists for Finnish.

---

## Ablation Study: Coreference Resolution in the KG

**Research question:** Does explicit coreference resolution in the KG improve question generation quality, or does the model learn implicit entity linking from co-occurring nodes?

### Setup

SQuAD-style passages often contain pronoun subjects after the first sentence:

```
Beyoncé Giselle Knowles-Carter is a singer …
She rose to fame in the late 1990s …
```

The raw KG contains `'she' | rise_in | 'the late 1990s'` alongside
`'Beyoncé Giselle Knowles - Carter' | be | 'singer'`.

With coref resolution (Stanza `coref` processor), the triple becomes
`'Beyoncé Giselle Knowles - Carter' | rise_in | 'the late 1990s'`.

### Three dataset conditions

| Condition | KG input | How to build |
|---|---|---|
| **Raw** | `she \| rise_in \| …` | `build_seq2seq_dataset.py --sources squad` |
| **Coref** | `Beyoncé \| rise_in \| …` | `build_seq2seq_dataset.py --sources squad --coref` |

### Expected outcomes and what they mean

| Model output | Interpretation |
|---|---|
| "When did **she** rise to fame?" | Coref not learned; raw KG insufficient signal |
| "When did **Beyoncé** rise to fame?" (from raw model) | Model learned implicit entity linking from KG context |
| "When did **Beyoncé** rise to fame?" (from coref model) | Explicit resolution helped; coref is a useful preprocessing step |

If raw and coref models score similarly on BLEU/BERTScore, it suggests the GNN representation encodes enough structural signal to resolve pronouns implicitly — a positive finding about graph-based input representations (not claimed by Pan et al. 2020).

### Why this matters for Finnish

Stanza coref is **English-only** (no Finnish model). If the raw model resolves pronouns implicitly from KG context, that result generalises directly to Finnish — meaning coref preprocessing is not a blocking dependency for the Finnish system.

### Implementation

- `KnowledgeGraphExtractor(coref=True)` enables Stanza coref (English only, warns otherwise).
- `Triple.coref_distance=1` marks any triple that was pronoun-resolved.
- Dataset records carry `"coref_resolved": true/false` for downstream filtering.
- Requires: `stanza.download('en', processors='coref')`
