"""
Maps a free-text answer (from a blind, options-unaware model -- decoder or
extractive) to the closest of 4 multiple-choice options.

Three-tier hybrid (see question_difficulty cascade investigation, 2026-11):
  1. Exact substring match: if exactly one option is contained in the raw
     text (or vice versa) after normalization, pick it deterministically.
     Catches cases like raw="The first film...is Dr No." -> option "Dr No"
     that a pure embedding similarity can miss in favor of a topically
     related but wrong option (e.g. "Skyfall").
  2. Keyword-overlap match: if no single substring hit, but exactly one
     option shares a distinctive content word/number with the raw text
     (and beats the runner-up option by a clear margin, not just by one
     stray word), pick it. Catches cases like raw="...220 photos and 40
     items..." -> option "many photos and pictures..." that substring
     matching misses because neither string fully contains the other, but
     which a bag-of-words human reader would catch instantly. Numbers
     count as content tokens (not stopwords). Skipped entirely for
     main-idea/title-style questions (see _is_main_idea_question) -- this
     tier is pure literal overlap, which structurally favors a wrong
     option that restates concrete passage details over the (correct but
     abstract) gold paraphrase; this is the exact bias that motivated
     moving off token_f1 in the first place (see below), so main-idea
     questions skip straight to the embedding tier instead. Found via a
     2026-11 spot-check: e.g. "A general introduction to Clara Barton"
     (gold, abstract) lost to "The Establishment of the American Red
     Cross" (wrong, but shares more literal words with a raw answer that
     mentions Clara Barton's "founding of the Red Cross").
     Requires a margin (best score beats runner-up by >=2), not just a
     unique max of 1 -- a single shared word is too easy to hit by
     chance (e.g. a stray match on a common-but-not-stopword word like
     "how" was enough to flip a title question before this fix).
  3. Embedding fallback (sentence-transformers/all-mpnet-base-v2, swapped
     from all-MiniLM-L6-v2 -- stronger on general STS/semantic-similarity
     benchmarks, confirmed via a 10K-item Haiku-answer accuracy test:
     MiniLM=57.1%, mpnet alone=58.7%, this 3-tier hybrid=~60%+).
  4. LLM tiebreaker (optional, opt-in via `llm_tiebreak_fn`): when tier 3's
     top-1/top-2 cosine similarity gap is small (< _TIEBREAK_GAP), the
     embedding is essentially guessing between two similar options (tested
     at only 38.5% accuracy on these specifically, vs 71.3% when the gap is
     large) -- so for callers that pass an `llm_tiebreak_fn` (text, options)
     -> index, this routes just that ambiguous slice (~19% of items) to it
     instead of trusting the embedding's near-tie pick. Tested on the same
     10K-item set: lifts that slice's accuracy to 67.1% (734 flipped
     wrong->correct, 189 flipped correct->wrong, net +545 -- roughly +5.5pp
     overall). See question_answering/docs/semantic_matcher_tiers.md.
     Deliberately NOT a hardcoded Bedrock call in this module -- this file
     has no AWS/network dependency and is also used by LocalDecoderAnswerer/
     LocalExtractiveAnswerer, which run offline on Roihu with no Bedrock
     credentials. Callers that already hold a Bedrock client (e.g.
     BlindClaudeBedrockAnswerer) pass their own tiebreak function; callers
     without one simply omit it and get tiers 1-3 only, unchanged.

Known remaining weakness (not fixed here, needs more thought): numeric/
quantity comparisons, e.g. raw="lasts 15 to 30 minutes" vs option "less
than half an hour" -- semantically identical but share no literal tokens,
so this still falls through to embedding similarity, which has no notion
of numeric magnitude and can pick an unrelated option (e.g. "three hours").

Replaces QAEvaluator.token_f1 for this specific job. token_f1 is literal
token overlap, which systematically penalizes a correct but PARAPHRASED
answer (e.g. a main-idea question's gold option is rarely a verbatim
passage span or a verbatim restatement) -- this was identified as a root
cause of LocalExtractiveAnswerer's near-chance accuracy on main-idea
questions specifically (see question_difficulty docs / cascade investigation,
2026-10). Embedding similarity captures "this means the same thing" instead
of "these share words", so it should treat extractive and decoder answers
fairly regardless of how literally each one's free-text answer overlaps
with the correct option's wording.

Model is loaded once per process (module-level singleton) and reused
across every answer() call -- loading it fresh per call would dominate
runtime at cascade scale (130K+ matches across a 13-candidate x 10K-item
run).
"""
from __future__ import annotations

import re

_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
_tokenizer = None
_model = None
_device = None

# Common function words excluded from keyword-overlap scoring -- deliberately
# small and conservative (a false-negative here just falls through to the
# embedding tier, same as today; a false-positive could wrongly override it).
_STOPWORDS = frozenset("""
a an the and or but of to in on at for with that this it as by from be is
was are were has have had he she they we you i his her their its not no
do does did can could will would should may might this these those there
been being so than then them him us our your my mainly main passage text
how why what when where who which whom whose
""".split())

# Question stems where the gold option is typically an ABSTRACTION of the
# passage/raw-answer (main idea, title, purpose), so literal word overlap
# systematically favors a wrong-but-concrete-sounding distractor over the
# correct-but-abstract option -- skip the keyword-overlap tier for these,
# straight to embeddings. See module docstring.
_MAIN_IDEA_STEM = re.compile(
    r"\b(?:mainly|main idea|best title|best headline|purpose of (?:writing|this)|"
    r"mainly (?:about|talk|discuss)|passage is (?:mainly|mostly) about)\b",
    re.IGNORECASE,
)


def _ensure_loaded() -> None:
    global _tokenizer, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoModel, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    _model = AutoModel.from_pretrained(_MODEL_ID).to(_device)
    _model.eval()


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed(texts: list[str]):
    """Returns an (N, hidden_dim) L2-normalized embedding tensor for the
    given texts, batched in one forward pass."""
    import torch

    _ensure_loaded()
    enc = _tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _model(**enc)
    pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _content_tokens(normalized_text: str) -> set[str]:
    return {tok for tok in normalized_text.split() if tok not in _STOPWORDS and len(tok) > 2}


def _substring_match(text_norm: str, option_norms: list[str]) -> int | None:
    hits = [i for i, o in enumerate(option_norms) if o and (o in text_norm or text_norm in o)]
    return hits[0] if len(hits) == 1 else None


_OVERLAP_MARGIN = 1  # best score must beat the runner-up by this much to override embeddings.
# Tried 2 (2026-11): fixed a stray-stopword false-tie bug but also undid a
# genuine win (a +1-margin "photos" case) for no gain -- the cases a bigger
# margin was meant to protect (main-idea/title questions) are now routed
# around the keyword tier entirely via _MAIN_IDEA_STEM, and turned out to
# fail at the embedding-similarity layer too, not just here -- so a bigger
# margin bought nothing there while losing real wins elsewhere. Back to 1,
# relying on the interrogative-word stopword fix + the stem carve-out for
# main-idea questions instead.


def _keyword_overlap_match(text_norm: str, option_norms: list[str]) -> int | None:
    text_tokens = _content_tokens(text_norm)
    if not text_tokens:
        return None
    scores = [len(text_tokens & _content_tokens(o)) for o in option_norms]
    ranked = sorted(scores, reverse=True)
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else 0
    if best == 0 or (best - runner_up) < _OVERLAP_MARGIN:
        return None
    best_idxs = [i for i, s in enumerate(scores) if s == best]
    return best_idxs[0] if len(best_idxs) == 1 else None


_TIEBREAK_GAP = 0.05  # below this top1/top2 cosine-sim gap, tier 3 is essentially
# guessing (38.5% accuracy in testing, vs 71.3% when the gap is large) -- see
# module docstring's tier 4.


def best_option_index(
    text: str,
    options: list[str],
    question: str | None = None,
    llm_tiebreak_fn=None,
) -> int:
    """Returns the index (0-3) of the option that best matches `text`
    (typically a model's free-text answer, generated without seeing the
    options). Tries exact substring, then keyword overlap (skipped for
    main-idea/title questions -- pass `question` so this carve-out can
    apply), then falls back to embedding cosine similarity -- see module
    docstring for why.

    `llm_tiebreak_fn`, if given, is `(text, options) -> int` (an index
    0-3): called instead of trusting the embedding tier's pick when its
    top1/top2 similarity gap is below _TIEBREAK_GAP (see tier 4 in the
    module docstring). Optional and opt-in -- omit it to keep this module
    free of any Bedrock/network dependency (see docstring)."""
    text_norm = _normalize(text)
    option_norms = [_normalize(o) for o in options]

    idx = _substring_match(text_norm, option_norms)
    if idx is not None:
        return idx

    if not (question and _MAIN_IDEA_STEM.search(question)):
        idx = _keyword_overlap_match(text_norm, option_norms)
        if idx is not None:
            return idx

    embeddings = embed([text] + list(options))
    text_emb, option_embs = embeddings[0], embeddings[1:]
    sims = (option_embs @ text_emb).tolist()
    ranked = sorted(range(len(sims)), key=lambda i: -sims[i])
    top1, top2 = ranked[0], ranked[1]
    gap = sims[top1] - sims[top2]

    if llm_tiebreak_fn is not None and gap < _TIEBREAK_GAP:
        return llm_tiebreak_fn(text, options)

    return top1
