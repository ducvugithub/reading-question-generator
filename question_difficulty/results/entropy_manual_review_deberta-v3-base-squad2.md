# Attention-Entropy Difficulty Signal — Experiment Report

**Model:** `deepset/deberta-v3-base-squad2` (`QA_MODEL_CANDIDATES["deberta-v3-base-squad2"]`)
**Layer:** 11 (last layer)
**Source notebook:** `question_difficulty/notebooks/entropy_manual_review.ipynb`
**Date:** 2026-09-04

This records results for one QA model before switching to another, so nothing gets lost. Re-run
the same notebook with a different `QA_MODEL` to produce a comparable report for the next model.

## Setup

- **Signal**: `AttentionDispersionSignal.get_sentence_distribution()` — averages attention over
  heads at one layer, extracts the question→passage attention sub-block, computes Shannon entropy
  over that distribution at both token level (`tok_entropy_norm`) and sentence level
  (`sent_entropy_norm`). Both are normalized by `log(N)` (N = token count or sentence count), so
  they're bounded in `[0, 1]` and (in principle) comparable across passages of different length.
- **Correctness**: `QAEvaluator.token_f1` (SQuAD-style precision/recall F1) against a gold answer,
  `is_correct` threshold `0.5`.
- **Confidence**: softmax(start_logit) × softmax(end_logit) at the model's argmax span — the
  model's own certainty, independent of whether it's actually correct.
- **Main corpus** (§1): 52 real questions across 14 passages — RACE-middle/high/C, OneStopQA
  (elementary/intermediate/advanced), SQuAD, 2 passages per source group, all non-MC questions on
  each passage included (~3.7 questions/passage average).

⚠️ **§3's main-corpus extraction (the length-confound Spearman check and the full 66-row sorted
list) is not currently cached in the notebook** — it was last run before later fixes in this
session and needs re-running to get final deberta numbers for that part. The last *reported*
numbers for `deberta-v3-base-squad2` at layer 11, from this session's chat (not re-verified against
a fresh cell run), were:

```
Spearman(n_sent, sent raw entropy)  = 0.938   Spearman(n_sent, sent norm entropy)  = 0.480
Spearman(n_tok,  tok  raw entropy)  = 0.870   Spearman(n_tok,  tok  norm entropy)  = 0.461
```

Take these as provisional. Everything below (§5–§8) is transcribed directly from the notebook's
currently-saved cell outputs, so it's solid.

## §5 — Literal vs. inferential (5 pairs, short → very long)

Same passage, one literal ("what") question and one inferential ("why") question, scored against
separate gold answers (the literal fact isn't always a complete answer to "why").

| passage | len | literal correct? | inferential correct? | Δtok_entropy_norm | Δsent_entropy_norm |
|---|---|---|---|---|---|
| Lily (charger) | short (~30w) | ✅ | ✅ | +0.100 | +0.414 |
| Maria (school) | short (~50w) | ✅ | ✅ | +0.034 | +0.121 |
| Millbrook (generator) | medium (~95w) | ✅ | ❌ (f1=0.312) | +0.034 | +0.179 |
| bandage | long (~190w) | ✅ | ❌ (f1=0.000) | +0.020 | +0.016 |
| Riverside Elementary (bus route) | very long (~230w) | ✅ | ❌ (f1=0.207) | +0.032 | +0.187 |

**Findings:**
- Entropy direction is consistent: inferential > literal in all 5 pairs, both entropy levels.
- Entropy **magnitude does not track correctness or difficulty**. The largest jump
  (`sent_entropy_norm +0.414`) is on the pair that was fully correct (Lily); the smallest
  (`+0.016`) is on a total failure (bandage, f1=0.000, model grabbed an unrelated clause). A
  signal meant to reflect "how hard/uncertain" should show the opposite ranking.
- Correctness degrades with passage length/complexity: both short pairs succeed, all three
  medium/long/very-long pairs fail the inferential question.

## §6 — Negation (5 pairs, very-short → long)

Positive ("what does X include/offer/cover") vs. negation ("what does X NOT include") — non-MC,
the negative fact is always explicitly stated in the passage.

| passage | positive correct? | negation correct? | Δtok_entropy_norm | Δsent_entropy_norm |
|---|---|---|---|---|
| handbook | ✅ | ✅ | +0.015 | +0.140 |
| gym membership | ✅ | ✅ | -0.069 | +0.000* |
| cafe menu | ✅ | ✅ (f1=0.800) | +0.081 | +0.477 |
| travel insurance | ✅ (f1=0.941) | ✅ | +0.052 | +0.201 |
| recycling program | ✅ | ✅ | +0.044 | +0.172 |

\* gym passage's `sent_entropy_norm=0.000` for both conditions — likely a degenerate case
(passage segmented to very few sentences, `_normalized_entropy` returns `0.0` when `N<=1`). Worth
checking sentence count before trusting this row.

**Findings:**
- **All 10 predictions correct** — negation, at least in this non-MC, explicitly-stated form, is
  not a hard case for this model, unlike inferential reasoning.
- Entropy direction is inconsistent this time (gym: `tok_entropy_norm` goes *down* for negation),
  unlike the always-positive pattern in §5/§7. With 100% correctness across the board there's no
  failure signal to correlate against here — this section mainly shows negation-with-explicit-text
  is easy for the model, not that entropy tracks a negation "cost."

## §7 — Comparison and superlative (5 pairs, 3 entities each)

Literal (single fact) vs. comparison (binary, named) vs. superlative (rank all 3 for the extremum),
all scored against the literal baseline.

| passage | comparison correct? | superlative correct? | Δtok (comp) | Δtok (sup) | Δsent (comp) | Δsent (sup) |
|---|---|---|---|---|---|---|
| factories | ❌ (both→'Hillcrest') | ❌ (→'Hillcrest') | +0.070 | +0.116 | +0.475 | +0.455 |
| marathon | ✅ | ❌ (→'Priya') | +0.058 | +0.099 | +0.258 | +0.294 |
| museum | ✅ | ✅ | +0.005 | +0.072 | +0.048 | +0.120 |
| phones | ❌ (→'Zenith X200') | ❌ (→'Zenith X200') | +0.010 | -0.035 | +0.127 | +0.001 |
| hospitals | ❌ (verbose span) | ✅ | -0.043 | -0.039 | +0.023 | -0.044 |

**Findings — this section changed my read of the whole exercise:**
- **Correctness doesn't rank comparison easier than superlative or vice versa** (2/5 each), and
  hospitals is a direct counterexample to "superlative is harder": comparison WRONG, superlative
  CORRECT, same passage.
- **⚠️ Known confound in the hospitals passage**: it contains the phrase *"the longest of the
  three"* directly attached to the correct entity (Eastview), which hands the model the superlative
  answer as literal text instead of requiring real numeric comparison. That's almost certainly why
  hospitals' superlative succeeded with high confidence (`1.000`) and *lower* entropy than the
  literal baseline — not evidence of genuine ranking capability. **Fix before reusing**: remove
  that phrase and re-run.
- **Repeated wrong answers suggest positional/recency heuristics, not real arithmetic**: factories'
  comparison and superlative both wrongly answer `'Hillcrest'` (the *last*-mentioned factory);
  phones' both wrongly answer `'Zenith X200'` (the *first*-mentioned phone). Neither is a
  consistent "always pick first/last" rule across pairs, but within each pair the model gives the
  *same* wrong answer to two different questions — consistent with falling back to a positional or
  lexical-overlap heuristic rather than computing which number is actually larger.
- **Token-level entropy is more consistent than sentence-level here**: superlative shows a larger
  `tok_entropy_norm` jump than comparison in 3/5 (factories, marathon, museum), tied in 1
  (hospitals), reversed in 1 (phones) — plausibly because ranking 3 entities needs attention spread
  across more distinct number-tokens than binary comparison. `sent_entropy_norm` shows no
  consistent ordering between the two operations.
- **Overall interpretation**: this model (SQuAD-only extractive, never trained on
  comparison/discrete-reasoning tasks — the same gap DROP was built to probe) likely lacks a real
  mechanism for numeric/superlative comparison. Confidently-wrong high-confidence predictions
  (factories' superlative: `confidence=0.876`, wrong) support "guessing via heuristic," not
  "uncertain but reasoning." **Entropy results from this section should be read as "model behavior
  on an out-of-distribution task," not as a graded difficulty signal** until re-tested on a model/
  task setup that can actually do comparison, and with the hospitals leak fixed.

## §8 — Vocabulary/phrasing complexity (5 pairs, cognitive operation held constant)

Same question (plain literal recall), same gold answer, only wording changes (plain vs.
dense/formal). Isolates surface complexity from reasoning demand.

| passage | plain correct? | dense correct? | Δtok_entropy_norm | Δsent_entropy_norm |
|---|---|---|---|---|
| Lily | ✅ | ✅ (f1=0.667) | -0.004 | +0.062 |
| Maria | ✅ | ✅ (f1=0.667) | +0.050 | +0.095 |
| Millbrook | ✅ | ✅ (f1=0.800) | -0.034 | -0.002 |
| bandage | ✅ | ✅ | +0.012 | +0.009 |
| Riverside Elementary | ✅ | ✅ | +0.011 | +0.047 |

**Findings:**
- **All 10 correct** — vocabulary/phrasing complexity alone doesn't break this model.
- **Entropy deltas are small and inconsistent in sign** (Millbrook's dense phrasing is actually
  *lower* entropy on both metrics) — average magnitude (~0.03–0.04) is roughly 4–5× smaller than
  the literal-vs-inferential deltas from §5 on the *same passages*. This is a useful negative
  control: entropy isn't obviously reacting to surface/linguistic complexity the way it clearly
  does react to a change in reasoning operation, which supports (with caveats — n=5, dense
  phrasings written by the assistant, not independently sourced) that §5's shift is capturing
  something about reasoning demand specifically, not just "harder to parse question."

## Cross-cutting takeaways for this model

1. **Direction over magnitude**: entropy reliably shows "inferential/comparison/superlative >
   literal" in sign, but its *size* does not rank pairs by actual difficulty or correctness —
   confirmed independently in §5 (bandage: smallest delta, total failure) and §7 (hospitals:
   negative delta, but that's the leaked-answer case).
2. **Token vs. sentence entropy diverge structurally**: §7 comparison questions show sentence
   entropy spiking while token entropy stays flat (attention moves to a second sentence but stays
   sharp within it); superlative pushes token entropy up further on top of that in most pairs. The
   two levels appear to capture different structural properties (how many locations touched vs.
   how concentrated within them), not the same thing at different granularity.
3. **This model can't reliably do numeric comparison/superlative reasoning** — it falls back to
   positional/lexical heuristics, sometimes confidently wrong (§7). Negation and vocabulary
   complexity, by contrast, are handled at 100% accuracy — those aren't meaningfully hard for it.
4. **Known test-set issues to fix before comparing against the next model**:
   - Hospitals passage (§7) leaks the superlative answer as text ("the longest of the three") —
     rewrite before reuse.
   - Gym passage (§6) has a degenerate `sent_entropy_norm=0.000` for both conditions — check
     sentence-count edge case.
   - §3's Spearman/length-confound numbers need a fresh run to be trustworthy for cross-model
     comparison (only chat-reported, not cell-cached, as of this report).
