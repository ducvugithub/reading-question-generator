# Cognitive Difficulty Estimation for Reading Comprehension Questions

## Overview

Cognitive difficulty measures **reasoning demand** — how hard it is to think through a question — as distinct from vocabulary difficulty (word frequency, CEFR level) or text readability (LIX, Flesch). A simple word appearing in a causal "why" question is cognitively harder than a rare word in a direct "where" question.

The `GraphCognitiveDifficultyEstimator` operationalises this with four signals derived from the question, its answer, and the knowledge graph (KG) extracted from the passage:

```
score = 0.45 × s_qtype + 0.30 × s_coref + 0.15 × s_coverage + 0.10 × s_density
label = easy (< 0.33) | medium (0.33–0.67) | hard (> 0.67)
```

---

## Factor 1 — Question Word Type (`s_qtype`, weight 0.45)

### Rationale

The question word is the strongest surface-level signal of required reasoning. Questions beginning with *why* or *how* demand causal or procedural reasoning; questions beginning with *when*, *where*, or *who* require simple entity retrieval.

### Scoring table

| Question word | Score | Bloom level |
|---|---|---|
| `when` | 0.10 | Remember |
| `where` | 0.15 | Remember |
| `who` | 0.20 | Remember |
| `what` | 0.25 | Understand |
| `which` | 0.35 | Understand |
| `how many / how much` | 0.40 | Apply |
| `how` | 0.75 | Analyse |
| `why` | 0.90 | Evaluate |
| other | 0.50 | fallback |

Implementation: longest-prefix match on the lowercased question string.

### Scientific basis

- **Bloom, B. S. (1956).** *Taxonomy of Educational Objectives, Handbook I: Cognitive Domain.* McKay. — The foundational six-level hierarchy (Knowledge → Comprehension → Application → Analysis → Synthesis → Evaluation) maps directly to the question-word ladder above.
- **Anderson, L. W. & Krathwohl, D. R. (2001).** *A Taxonomy for Learning, Teaching and Assessing: A Revision of Bloom's Taxonomy.* Longman. — Updated two-dimensional framework; compact overview in Krathwohl (2002), *Theory into Practice*, 41(4).
- **Empirical NLP support:** Tack et al. (2017), "Question Difficulty — How to Estimate Without Norming" (ACL Anthology W17-5001) uses question-word type as a primary difficulty signal; cross-lingual studies (MLQA) confirm WHY/HOW questions are empirically harder than WHEN/WHO across languages.

---

## Factor 2 — Coreference Dependency of the Answer (`s_coref`, weight 0.30)

### Rationale

If the KG triple that contains the answer has a pronoun (*she*, *it*, *they*…) in the subject or object position, the reader must resolve that pronoun to understand which entity the fact belongs to. Pronoun resolution is a documented source of cognitive load in reading comprehension.

### Computation

1. Find all raw-KG triples where the answer string appears in the subject or object.
2. Of those *covering triples*, count how many have a pronoun in subject or object position.
3. `s_coref = pronoun_covering / covering` (0.0 if no covering triples).

This is **answer-relative**: passage-level pronoun counts are ignored. A passage heavy with pronouns does not inflate difficulty for a question whose answer appears in a named-entity triple.

### Scientific basis

- **Just, M. A. & Carpenter, P. A. (1980).** A theory of reading: From eye fixations to comprehension. *Psychological Review*, 87(4), 329–354. — Eye-tracking evidence that pronoun-to-antecedent resolution produces measurable increases in fixation time, establishing pronoun resolution as a direct cognitive cost.
- **Gernsbacher, M. A. (1990).** *Language Comprehension as Structure Building.* Erlbaum. — The Structure Building Framework predicts that ambiguous or anaphoric references force a mental "shift" (creating a new sub-structure), increasing processing load relative to unambiguous reference.
- **PMC8946822 (2022).** EEG study measuring cognitive workload during ambiguous vs. unambiguous pronoun resolution; confirms distinct workload levels consistent with the above theories.

---

## Factor 3 — Answer Explicitness (`s_coverage`, weight 0.15)

### Rationale

If the answer can be read off directly from a KG triple, the question requires retrieval. If the answer is absent from the KG, the reader must infer, synthesise, or reason beyond explicit facts — the hardest case.

### Computation

Binary:
- `0.0` — answer string found as a substring of any triple's subject or object → directly retrievable
- `1.0` — not found → requires inference or synthesis

### Scientific basis

- **Sugawara, S., Inui, K., Sekine, S. & Aizawa, A. (2018).** What Makes Reading Comprehension Questions Easier? In *Proceedings of EMNLP 2018*, pp. 4208–4219. ([ACL D18-1453](https://aclanthology.org/D18-1453/)) — Directly studies answer-extraction vs. inference-requiring questions as the primary difficulty axis; proposes heuristics to split datasets accordingly.
- **Lai, G., Xie, Q., Liu, H., Yang, Y. & Hovy, E. (2017).** RACE: Large-scale ReAding Comprehension Dataset From Examinations. In *Proceedings of EMNLP 2017*, pp. 785–794. ([ACL D17-1082](https://aclanthology.org/D17-1082/)) — RACE shows model accuracy drops from ~95% (humans) to ~43% (models) on inference-requiring vs. extraction questions, empirically validating explicitness as a hard difficulty signal.
- **Yao, S. et al. (2022).** What Makes Reading Comprehension Questions Difficult? In *Proceedings of ACL 2022*. ([ACL 2022.acl-long.479](https://aclanthology.org/2022.acl-long.479/)) — Directly relevant follow-up on question difficulty factors.

---

## Factor 4 — KG Density (`s_density`, weight 0.10)

### Rationale

A passage that produces many KG triples is informationally denser — more entities, more relations, more facts to hold in working memory. Density serves as a lightweight proxy for passage complexity.

### Computation

`s_density = min(triple_count / 15, 1.0)` — saturates at 15 triples; constant across all questions from the same passage.

### Scientific basis

- **Graesser, A. C., McNamara, D. S., Louwerse, M. M. & Cai, Z. (2004).** Coh-Metrix: Analysis of text on cohesion and language. *Behavior Research Methods, Instruments & Computers*, 36(2), 193–202. ([Springer](https://link.springer.com/article/10.3758/BF03195564)) — The Coh-Metrix framework uses over 200 text features including referential cohesion, entity density, and syntactic complexity as passage difficulty signals. KG density is a structural proxy for the same concept.
- **McNamara, D. S., Graesser, A. C., McCarthy, P. M. & Cai, Z. (2014).** *Automated Evaluation of Text and Discourse with Coh-Metrix.* Cambridge University Press. — Book-length treatment of density and cohesion as difficulty dimensions.

---

## Weights: Heuristic, Pending Empirical Calibration

The factor scores are scientifically grounded; the **weights (0.45 / 0.30 / 0.15 / 0.10) are heuristic**. The literature supports that these factors matter but does not prescribe specific ratios between them.

Two paths to empirical calibration:

1. **Regress against human judgments** — collect difficulty ratings from annotators on a held-out QA set, fit a linear model to recover weights.
2. **Regress against learner response data** — use per-question correctness rates from Revita (real learner interactions) as a ground-truth difficulty signal and fit the weights accordingly. This is the most principled approach for the language-learning context.

---

## Thresholds

| Range | Label |
|---|---|
| `score < 0.33` | easy |
| `0.33 ≤ score < 0.67` | medium |
| `score ≥ 0.67` | hard |

Thresholds are symmetric thirds of [0, 1]. They should be re-evaluated once empirical calibration produces a score distribution over a real dataset.

---

## Implementation

- **Estimator:** `question_generation/difficulty/cognitive.py` — `GraphCognitiveDifficultyEstimator`
- **LLM baseline for ground-truth labelling:** `LLMCognitiveDifficultyEstimator` (same file) — calls Claude Haiku with the passage KG and answer; useful for auditing the rule-based estimator on a sample.
- **Demo / batch script:** `scripts/estimate_cognitive_difficulty.py`

```bash
# Local demo (extracts KG on-the-fly, requires stanza models)
python scripts/estimate_cognitive_difficulty.py --demo --mode graph --verbose

# Run on a built dataset file
python scripts/estimate_cognitive_difficulty.py \
    --input data/training/en/train.jsonl \
    --mode graph --output data/training/en/train_cog.jsonl

# LLM mode for ground-truth labels (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... python scripts/estimate_cognitive_difficulty.py \
    --demo --mode llm --verbose
```

---

## Related Work

See also `docs/related_work_qg.md` for broader question generation literature.
