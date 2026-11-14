"""
Shared item-loading for the answerer cascade (run_answerer_cascade.py) and
its scorer (score_cascade_results.py). Kept in exactly one place so both
always apply the IDENTICAL per-item option shuffle -- see load_items()'s
docstring for why that shuffle exists. If the shuffle logic drifted
between the two scripts (e.g. each reimplementing it separately), scoring
would silently misread which option text a model's letter corresponded
to, since the prompt-building side and the scoring side would disagree
on the order.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ONESTOPQA_REPO = REPO_ROOT.parent / "onestop-qa"
RASCH_DIFFICULTY_CACHE = REPO_ROOT / "question_difficulty/notebooks/irt_rasch_difficulty_cache.json"

sys.path.insert(0, str(REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"))

# Matches RACE's original MCQ-elimination stems ("Which of the following is
# NOT true/mentioned/a reason...", "...EXCEPT _ .", "Which is WRONG?") that
# only make sense with the 4 options shown. The BLIND cascade
# (question_answering/answerer.py's predict_answer(), never shown options)
# and QG (which trains to generate question text with no options ever
# attached) both structurally cannot handle these -- found via manual
# fact-check of sample_rerank_diffcond.py output on 2026-11-11.
#
# First pass only matched the literal "NOT true/correct/right" + "EXCEPT"
# stems (~3.5% of the pool); a 200-item random-sample audit of what
# SURVIVED that filter turned up 5 more genuine elimination questions it
# missed ("NOT mentioned", "NOT a reason", "NOT inferred", "is WRONG",
# "doesn't have") -- the real pattern is "which" + any negation/"wrong",
# not a fixed word list after "NOT". Broadened to require EXCEPT, or
# "which" co-occurring with not/n't/wrong/incorrect anywhere in the
# question -- checked against the full ~87.9K-item pool with zero false
# positives found in a 15-example spot-check of the newly-caught items
# (all genuine "pick the one that doesn't fit" questions). Now catches
# 5.2% of the pool (was 3.5%).
#
# Excluded here, not in a downstream filter, so every caller of
# load_race_training_pool() (run_answerer_cascade.py --item-source
# race_pool, and the IRT data-prep scripts that build on its difficulty
# cache) gets the fix automatically.
_EXCEPT_STEM = re.compile(r"EXCEPT", re.IGNORECASE)
_WHICH_STEM = re.compile(r"\bwhich\b", re.IGNORECASE)
_NEGATION_STEM = re.compile(r"\bnot\b|n't\b|\bwrong\b|\bincorrect\b", re.IGNORECASE)

# Positive elimination-style stems ("Which of the following is TRUE/correct...")
# are just as option-dependent as the negated ones above -- "the following"
# always refers to the 4 answer choices, so the question is meaningless
# without them -- but the negation-only check above missed them (found
# 2026-11-13: these kept showing up verbatim as FiLM-model HARD output,
# since they survived into QG training targets unfiltered). Matches
# augment_qg_difficulty_llm.py's _MC_PATTERNS, unified here so every
# load_race_training_pool() caller (cascade AND QG data prep) gets it.
_MC_FOLLOWING_STEM = re.compile(
    r"which (?:of the following|one of the following|of these)", re.IGNORECASE
)


def _is_blind_unanswerable(question: str) -> bool:
    return bool(_EXCEPT_STEM.search(question)) or bool(_MC_FOLLOWING_STEM.search(question)) or (
        bool(_WHICH_STEM.search(question)) and bool(_NEGATION_STEM.search(question))
    )


# A blind GENERATIVE answerer (decoder-style local models, or any
# claude_bedrock_blind candidate) sometimes declines to guess instead of
# answering. Found at 1.21% prevalence (121/10,000) in the first full Haiku
# blind run -- splits into two genuinely different cases (2026-11 audit of
# all 121), NOT one undifferentiated "refusal":
#   1. "needs options" (68/121): raw_text explicitly asks for the options/
#      choices it wasn't shown -- e.g. "I don't see the answer options,
#      please provide them". This is a PROMPT-FORMAT artifact (the question's
#      own phrasing, e.g. "which of the following...", primes the model to
#      expect a list even though it's being asked blind) -- not genuine
#      content difficulty. Scoring it as wrong would inject formatting noise
#      into the difficulty signal and unfairly penalize a model for admitting
#      uncertainty vs. a model that always guesses something. Excluded
#      entirely (is_correct=None), same convention data_processing.py uses
#      for other broken/unscoreable items.
#   2. "content gap" (53/121): raw_text says the passage doesn't contain the
#      needed info -- e.g. "the passage does not provide information about
#      X". This IS a genuine failure to find/infer the answer -- real
#      difficulty signal, not a formatting artifact -- so it should be
#      scored as WRONG (is_correct=False), not excluded, same as any other
#      incorrect answer.
#
# ONLY applicable to generative candidates -- extractive/BERT-style span
# extractors (candidate name prefix "extractive_") mechanically pull a
# substring from the passage and cannot produce free-form refusal language;
# checked (2026-11), the few apparent "refusals" there were false positives
# -- literal passage dialogue a character says ("I don't see...", "I can't
# answer your question") that happened to get extracted as the span, not the
# model declining to answer. Callers MUST skip this check entirely for
# extractive_* candidates -- see is_extractive_candidate().
_NEEDS_OPTIONS_PATTERNS = [
    r"\b(?:options?|choices?|statements?|letter)\b.*\b(?:provide|share|see|listed|given|aren'?t)\b",
    r"\bplease (?:provide|share)\b",
    r"\bI don'?t see\b",
    r"\bneed to see\b",
    r"\bno answer options\b",
    r"\bwithout (?:the|seeing|knowing) the (?:answer )?options?\b",
    r"\bno (?:answer )?options? (?:were|was|are|is) provided\b",
    r"\bcan you (?:provide|share) the options\b",
]
_NEEDS_OPTIONS_RE = re.compile("|".join(_NEEDS_OPTIONS_PATTERNS), re.IGNORECASE)

_CONTENT_GAP_PATTERNS = [
    # Subject MUST be the passage/text itself, not some entity the passage
    # describes -- "The passage does not provide information about X" is a
    # refusal, but "The Medical Center does not provide dental care
    # services" is a legitimate ANSWER with the same verb phrase. Found as
    # a false positive in a spot-check (2026-11) of a looser version of
    # this pattern that matched "does not (contain|provide|mention)"
    # regardless of subject.
    r"\b(?:the |this )?passage (?:does not|doesn'?t) (?:contain|provide|mention)\b",
    r"\b(?:the |this )?text (?:does not|doesn'?t) (?:contain|provide|mention)\b",
    r"\bnot mentioned anywhere\b",
    r"\bcannot answer this question\b",
    r"\bI (?:cannot|can'?t) answer\b",
    r"\bI (?:do not|don'?t) have (?:enough|sufficient) information\b",
    r"\bI'?m unable to\b",
]
_CONTENT_GAP_RE = re.compile("|".join(_CONTENT_GAP_PATTERNS), re.IGNORECASE)


def is_extractive_candidate(name: str) -> bool:
    return name.startswith("extractive_")


def classify_refusal(raw_text: str | None) -> str | None:
    """Returns "needs_options", "content_gap", or None (not a refusal at
    all). Only call this for generative candidates -- see module comment
    above is_extractive_candidate()."""
    if not raw_text:
        return None
    if _NEEDS_OPTIONS_RE.search(raw_text):
        return "needs_options"
    if _CONTENT_GAP_RE.search(raw_text):
        return "content_gap"
    return None


def is_refusal_answer(raw_text: str | None) -> bool:
    """Deprecated: true for EITHER refusal kind, with no "needs_options"
    vs "content_gap" distinction. Kept only so old callers/tests importing
    this name don't break -- new code should use classify_refusal()."""
    return classify_refusal(raw_text) is not None


def load_items() -> dict:
    """item_id -> {passage, question, options (shuffled), correct_answer_text, difficulty}."""
    from data_processing import HumanResponseBank

    assert ONESTOPQA_REPO.exists(), f"expected onestop-qa clone at {ONESTOPQA_REPO}"
    assert RASCH_DIFFICULTY_CACHE.exists(), (
        f"{RASCH_DIFFICULTY_CACHE} not found -- run question_difficulty_with_irt.ipynb's "
        "IRT section first, it writes this cache"
    )
    bank = HumanResponseBank(ONESTOPQA_REPO)
    item_difficulty = json.loads(RASCH_DIFFICULTY_CACHE.read_text())

    items = {}
    for item_id, rows in bank.rows_by_item().items():
        if item_id not in item_difficulty:
            continue
        r = rows[0]
        # qa.tsv's raw "options" column is NOT the order shown to human
        # participants (that's the separate "new_options" permutation) --
        # it's un-shuffled storage order, and the correct answer sits at
        # position 0 in ~100% of resolvable items (586/648 RACE, 588/648
        # OneStopQA -- confirmed empirically). Feeding that directly to a
        # model as "A) ... B) ... C) ... D) ..." would just measure each
        # model's bias toward picking "A", not reading comprehension.
        # Shuffle here, once, with a per-item seed (reproducible across
        # runs/candidates, and identical between this loader and
        # score_cascade_results.py) -- every downstream Answerer/scoring
        # step compares by option TEXT, not position, so this is the
        # only place that needs the shuffle applied.
        options = r["options"].split("|")
        random.Random(item_id).shuffle(options)
        items[item_id] = {
            "passage": r["paragraph"],
            "question": r["question"],
            "options": options,
            "correct_answer_text": r["correct_answer_text"],
            "difficulty": item_difficulty[item_id],
        }
    return items


def load_race_training_pool() -> dict:
    """item_id -> {passage, question, options (shuffled), correct_answer_text,
    difficulty: None} -- for enriching cascade-IRT coverage onto the ~87K RACE
    questions the actual QG models train on (data/qg/raw/race/train.jsonl),
    not just the 1,296 OneStopQA/RACE items with real human responses.

    Deliberately does NOT read data/qg/raw/race/train.jsonl -- that file
    (built by question_generation/scripts/prepare_qg_test_sets.py) only kept
    passage/question/difficulty/source for QG training targets and dropped
    options/answer entirely (QG's target is the question text, not a letter).
    Reads ehovy/race directly instead (same source dataset) to get those
    fields back, keyed by its own `example_id` (stable, e.g. "middle4558.txt")
    rather than trying to re-match by (passage, question) text against the
    QG pool -- a real risk, since duplicate question text across different
    articles is the exact root cause already found and fixed for the 66
    broken-gold-answer items (see redo_irt_excluding_broken_gold.py).

    Does NOT reproduce QG's own 80/10/10 train/val/test split (keyed by a
    different ID scheme) -- callers needing train/val/test consistency should
    re-split this pool the same way (MD5-hash the passage, see
    prepare_qg_test_sets.py's _split_bucket) rather than assume alignment
    with the existing data/qg/raw/race/{split}.jsonl files."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # avoid the HF freshness-check hang
                                                    # documented elsewhere in this project
    from datasets import load_dataset

    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    items = {}
    n_stem_filtered = 0
    for subset in ("middle", "high"):
        ds = load_dataset("ehovy/race", subset, split="train")
        # example_id is per-ARTICLE, not per-question (~3.65 questions share
        # one example_id, e.g. "middle4558.txt") -- using it alone as item_id
        # collapses most rows via dict-key collision (confirmed: 25,421 middle
        # rows -> only 6,409 unique example_ids). Append a running per-article
        # occurrence counter to make each question's id unique.
        seen_count: dict[str, int] = {}
        for row in ds:
            if _is_blind_unanswerable(row["question"]):
                n_stem_filtered += 1
                continue
            idx = letter_to_idx.get(row["answer"])
            if idx is None or idx >= len(row["options"]):
                continue
            occurrence = seen_count.get(row["example_id"], 0)
            seen_count[row["example_id"]] = occurrence + 1
            item_id = f"{subset}_{row['example_id']}_{occurrence}"
            options = list(row["options"])
            correct_answer_text = options[idx]
            random.Random(item_id).shuffle(options)
            items[item_id] = {
                "passage": row["article"],
                "question": row["question"],
                "options": options,
                "correct_answer_text": correct_answer_text,
                "difficulty": None,
            }
    print(f"load_race_training_pool: filtered {n_stem_filtered} blind-unanswerable "
          f"(EXCEPT/NOT true/correct/right) items, {len(items)} remain")
    return items
