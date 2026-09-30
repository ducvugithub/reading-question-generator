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
