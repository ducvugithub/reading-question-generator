# Answer-to-option matching: tiers and flow

How a blind answerer's free-text answer (generated without seeing the 4 MCQ
options) gets mapped to one of them, as implemented in
`question_answering/semantic_matcher.py`'s `best_option_index()`.

Tested on the 10K-item Haiku blind-cascade run
(`question_difficulty/scripts/cascade_results_race_pool_10k_by_passage_blind_v2_haiku.json`).

## Current flow (implemented)

1. **Exact substring match** (~15% of items)
   Normalize (lowercase, strip punctuation) the raw text and each option. If
   exactly one option is a substring of the raw text, or vice versa, pick it
   deterministically. No model call.
   Example: raw="...the first film...is Dr No." -> option "Dr No".

2. **Keyword-overlap match** (~45% of items)
   If tier 1 doesn't resolve, count shared content words (stopwords and
   interrogatives -- how/why/what/when/where/who/which -- excluded; numbers
   count) between the raw text and each option. Pick the option with the
   highest count, if it beats the runner-up by >=1 (margin).
   **Skipped entirely for main-idea/title-style questions** ("what's this
   mainly about", "best title", "purpose of writing") -- literal word
   overlap there systematically favors a wrong-but-concrete-sounding
   distractor over the correct-but-abstract gold option (the gold answer is
   an abstraction of the passage, so it naturally shares fewer literal words
   with a raw answer that restates concrete details).
   Example: raw="...220 photos and 40 items..." -> option "many photos and
   pictures..." (neither string fully contains the other, but "photos" is a
   shared distinctive word).

3. **Embedding fallback** (~40% of items, `sentence-transformers/all-mpnet-base-v2`)
   Cosine similarity between the raw text and each option; picks the
   highest. Used whenever tiers 1-2 don't resolve (including all main-idea
   questions, which skip tier 2 entirely).

### Accuracy progression (10K Haiku items)
| Matcher | Accuracy |
|---|---|
| Original (MiniLM, embedding-only) | 57.1% |
| mpnet swap (embedding-only) | 58.7% |
| 3-tier hybrid (substring -> keyword -> mpnet) | 60.3% |

## Tier 4: LLM tiebreaker (tested, not yet wired in)

Within tier 3 (embedding fallback), the top-1/top-2 similarity gap is a
reliable confidence signal:

| top1/top2 gap | items (of 4,517 reaching tier 3) | accuracy |
|---|---|---|
| < 0.02 | 895 (9% of all 10K) | 37.4% |
| < 0.05 | 1,910 (19% of all 10K) | 38.5% |
| >= 0.15 | 830 | 71.3% |

Below gap 0.05 the embedding model is essentially guessing between two
similar options (near chance). Routing just these 1,910 ambiguous items to
an LLM tiebreaker (Haiku via Bedrock, given the raw text + 4 options, asked
to pick the closest match) instead of trusting the embedding's near-tie pick:

- Old (embedding-only) accuracy on these 1,910 items: **38.5%**
- New (LLM tiebreaker) accuracy: **67.1%**
- 734 flipped wrong->correct, 189 flipped correct->wrong (net +545)

Projected effect on the full 10K set: **+5.5pp overall accuracy** (545/10,000).

Cost/latency: ~1,910 extra short Bedrock calls (4 max_tokens output) per
candidate scored -- cheap and bounded, since it only fires for the ~19% of
items that are already ambiguous, not every item.

## Known, unresolved weaknesses

- **Numeric/quantity comparisons**: e.g. raw="lasts 15 to 30 minutes" vs.
  option "less than half an hour" -- semantically identical but share no
  literal tokens, and mpnet/keyword-overlap have no notion of numeric
  magnitude; can pick an unrelated option (e.g. "three hours"). Not fixed by
  any tier here, including the LLM tiebreaker (not yet tested specifically
  on this failure mode).
- **Near-identical abstract-vs-concrete option pairs** on main-idea
  questions (e.g. "A general introduction to Clara Barton" vs. "Clara
  Barton's contribution to the Red Cross"): confirmed this fails at the
  *embedding* layer itself, not just the keyword tier -- the stem carve-out
  (tier 2 skip) can't fix a limitation that's already present in tier 3.
- **Refusals**: ~1.2% of raw answers (121/10,000 in the Haiku run) are the
  model declining to guess ("I don't see the answer options...", "the
  passage does not contain information about..."). These are filtered out
  before scoring (`cascade_common.is_refusal_answer()`, wired into
  `score_cascade_results.py`) rather than mapped to a nearest option --
  separate fix, not part of the matcher itself.
