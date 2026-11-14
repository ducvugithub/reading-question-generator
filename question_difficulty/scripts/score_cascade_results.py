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

sys.path.insert(0, str(REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_common import (  # noqa: E402 -- see cascade_common.py, ensures the exact same
    load_items, load_race_training_pool, classify_refusal, is_extractive_candidate,  # per-item option
)                                       # shuffle used when building prompts


def main() -> None:
    from data_processing import _texts_match

    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="Raw cascade_results.json from run_answerer_cascade.py")
    parser.add_argument("--merge", nargs="*", default=[],
                         help="Additional raw results files to merge in first (per-item answers dicts "
                              "are combined; candidates already present are NOT overwritten)")
    parser.add_argument("--output", default=None,
                         help="Where to write the scored file (default: <results_file> with _scored suffix)")
    parser.add_argument("--item-source", default="onestopqa", choices=["onestopqa", "race_pool"],
                         help="Must match whichever --item-source run_answerer_cascade.py used to "
                              "produce results_file -- the two sources use disjoint item_id schemes.")
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

    items = load_race_training_pool() if args.item_source == "race_pool" else load_items()
    correct_answer_by_item = {item_id: it["correct_answer_text"] for item_id, it in items.items()}
    options_by_item = {item_id: it["options"] for item_id, it in items.items()}

    accuracy: dict[str, list[bool]] = {}
    n_excluded = 0  # "needs options" refusals only -- see cascade_common.classify_refusal
    n_content_gap = 0  # "content gap" refusals -- scored as wrong, not excluded (see below)
    for item_id, row in results.items():
        if item_id not in correct_answer_by_item:
            continue
        gold = correct_answer_by_item[item_id]
        options = options_by_item[item_id]
        row["gold_answer"] = gold
        for name, answer in row["answers"].items():
            # Refusal detection only applies to generative candidates --
            # extractive/BERT-style span extractors can't produce free-form
            # refusal language; checking them anyway risks false positives
            # (literal passage dialogue happening to match, e.g. a character
            # saying "I don't see..."), see cascade_common.py's comment.
            refusal_kind = None if is_extractive_candidate(name) else classify_refusal(answer.get("raw_text"))
            if refusal_kind == "needs_options":
                # prompt-format artifact, not a wrong answer -- excluded from
                # both accuracy and (via is_correct=None) the IRT response
                # matrix, same convention data_processing.py uses for broken items
                answer["is_correct"] = None
                n_excluded += 1
                continue
            if refusal_kind == "content_gap":
                # genuine failure to find/infer the answer -- real difficulty
                # signal, counted as wrong rather than excluded
                answer["is_correct"] = False
                accuracy.setdefault(name, []).append(False)
                n_content_gap += 1
                continue
            letter = answer.get("letter")
            predicted_text = options["ABCD".index(letter)] if letter else None
            is_correct = _texts_match(predicted_text, gold) if predicted_text else False
            answer["is_correct"] = is_correct
            accuracy.setdefault(name, []).append(is_correct)

    output_path = Path(args.output) if args.output else Path(args.results_file).with_name(
        Path(args.results_file).stem + "_scored.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"scored {len(results)} items, {len(accuracy)} candidates, "
          f"{n_excluded} needs-options refusal(s) excluded, "
          f"{n_content_gap} content-gap refusal(s) scored as wrong -- wrote {output_path}\n")

    print("Accuracy per candidate:")
    for name, correct_flags in sorted(accuracy.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        n_correct = sum(correct_flags)
        print(f"  {name:16s} {n_correct}/{len(correct_flags)} ({n_correct/len(correct_flags):.1%})")


if __name__ == "__main__":
    main()
