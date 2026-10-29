#!/usr/bin/env python3
"""
Fits a Rasch (1PL) model to a scored race_pool cascade results file (from
score_cascade_results.py --item-source race_pool), treating each cascade
candidate as a "person" and each sampled RACE item as an "item". Mirrors
the fit done for the first (question-level, now-superseded)
cascade_results_race_pool_10k_scored.json -> irt_rasch_race_pool_10k_*.json,
redone here for the passage-level resample so centering has full
per-passage question coverage.

Rasch only (not Bayesian/NUTS) -- a training label only needs the point
estimate, and NUTS at ~10k items x 11 candidates would be impractically
slow (see run_answerer_cascade_race_pool.job's trailing comment).

Usage:
  python question_difficulty/scripts/fit_rasch_race_pool.py \\
      question_difficulty/scripts/cascade_results_race_pool_10k_by_passage_scored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"))

from rasch_model import fit_rasch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scored_file", help="Scored cascade results (has is_correct per answer)")
    parser.add_argument("--output-prefix", default=None,
                         help="Default: <scored_file> stem with _scored suffix stripped")
    args = parser.parse_args()

    scored_path = Path(args.scored_file)
    results = json.loads(scored_path.read_text())

    triples = []
    for item_id, row in results.items():
        for candidate, answer in row["answers"].items():
            if "is_correct" not in answer:
                continue
            triples.append((candidate, item_id, answer["is_correct"]))

    print(f"fitting Rasch on {len(triples)} (candidate, item) responses, "
          f"{len(results)} items, {len(set(t[0] for t in triples))} candidates")
    fit = fit_rasch(triples)
    print(f"converged={fit.converged} loss={fit.loss:.2f}")

    prefix = args.output_prefix or scored_path.stem.replace("_scored", "")
    difficulty_path = scored_path.with_name(f"irt_rasch_{prefix}_difficulty.json")
    ability_path = scored_path.with_name(f"irt_rasch_{prefix}_ability.json")
    difficulty_path.write_text(json.dumps(fit.item_difficulty, indent=2))
    ability_path.write_text(json.dumps(fit.person_ability, indent=2))
    print(f"wrote {difficulty_path}")
    print(f"wrote {ability_path}")

    print("\nability ranking (higher = stronger):")
    for name, theta in sorted(fit.person_ability.items(), key=lambda kv: -kv[1]):
        print(f"  {name:28s} {theta:+.3f}")

    diffs = list(fit.item_difficulty.values())
    print(f"\ndifficulty range: {min(diffs):.2f} to {max(diffs):.2f}")


if __name__ == "__main__":
    main()
