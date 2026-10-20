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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ONESTOPQA_REPO = REPO_ROOT.parent / "onestop-qa"
RASCH_DIFFICULTY_CACHE = REPO_ROOT / "question_difficulty/notebooks/irt_rasch_difficulty_cache.json"

sys.path.insert(0, str(REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"))


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
    for subset in ("middle", "high"):
        ds = load_dataset("ehovy/race", subset, split="train")
        # example_id is per-ARTICLE, not per-question (~3.65 questions share
        # one example_id, e.g. "middle4558.txt") -- using it alone as item_id
        # collapses most rows via dict-key collision (confirmed: 25,421 middle
        # rows -> only 6,409 unique example_ids). Append a running per-article
        # occurrence counter to make each question's id unique.
        seen_count: dict[str, int] = {}
        for row in ds:
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
    return items
