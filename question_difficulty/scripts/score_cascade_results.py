#!/usr/bin/env python3
"""
Scores a cascade_results.json (from run_answerer_cascade.py -- raw
letter/confidence per candidate per item, no correctness) against the
real gold answers: adds "is_correct" to every answer, and prints an
accuracy summary per candidate. Pure string matching against data already
in this repo's onestop-qa clone -- no model loading, runs in seconds
regardless of how many candidates/items are in the file.

This is deliberately separate from run_answerer_cascade.py so scoring
logic lives in exactly one place, whether the raw answers came from a
local Mac run, a Roihu GPU run, or (most likely) several files merged
together -- see --merge.

Usage:
  python question_difficulty/scripts/score_cascade_results.py \\
      question_difficulty/scripts/cascade_results.json

  # Merge a Roihu run (local decoders) with a local-only Haiku run (same
  # --n-items/--seed, so the same items were sampled in both) before scoring:
  python question_difficulty/scripts/score_cascade_results.py \\
      question_difficulty/scripts/cascade_results_roihu.json \\
      --merge question_difficulty/scripts/cascade_results_haiku_only.json \\
      --output question_difficulty/scripts/cascade_results_merged_scored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ONESTOPQA_REPO = REPO_ROOT.parent / "onestop-qa"

sys.path.insert(0, str(REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"))


def main() -> None:
    from data_processing import HumanResponseBank, _texts_match

    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Raw cascade_results.json from run_answerer_cascade.py")
    parser.add_argument("--merge", nargs="*", default=[],
                         help="Additional raw results files to merge in first (per-item answers dicts "
                              "are combined; candidates already present are NOT overwritten)")
    parser.add_argument("--output", default=None,
                         help="Where to write the scored file (default: <results_file> with _scored suffix)")
    args = parser.parse_args()

    results = json.loads(Path(args.results_file).read_text())
    for merge_path in args.merge:
        other = json.loads(Path(merge_path).read_text())
        for item_id, row in other.items():
            if item_id not in results:
                results[item_id] = row
                continue
            for name, answer in row["answers"].items():
                results[item_id]["answers"].setdefault(name, answer)

    assert ONESTOPQA_REPO.exists(), f"expected onestop-qa clone at {ONESTOPQA_REPO}"
    bank = HumanResponseBank(ONESTOPQA_REPO)
    correct_answer_by_item = {item_id: rows[0]["correct_answer_text"] for item_id, rows in bank.rows_by_item().items()}
    options_by_item = {item_id: rows[0]["options"].split("|") for item_id, rows in bank.rows_by_item().items()}

    accuracy: dict[str, list[bool]] = {}
    for item_id, row in results.items():
        if item_id not in correct_answer_by_item:
            continue
        gold = correct_answer_by_item[item_id]
        options = options_by_item[item_id]
        row["gold_answer"] = gold
        for name, answer in row["answers"].items():
            letter = answer.get("letter")
            predicted_text = options["ABCD".index(letter)] if letter else None
            is_correct = _texts_match(predicted_text, gold) if predicted_text else False
            answer["is_correct"] = is_correct
            accuracy.setdefault(name, []).append(is_correct)

    output_path = Path(args.output) if args.output else Path(args.results_file).with_name(
        Path(args.results_file).stem + "_scored.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"scored {len(results)} items, {len(accuracy)} candidates -- wrote {output_path}\n")

    print("Accuracy per candidate:")
    for name, correct_flags in sorted(accuracy.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        n_correct = sum(correct_flags)
        print(f"  {name:16s} {n_correct}/{len(correct_flags)} ({n_correct/len(correct_flags):.1%})")


if __name__ == "__main__":
    main()
