#!/usr/bin/env python3
"""
Builds "data/qde_irt_pilot": a Question Difficulty Estimator (QDE) training
set labeled with REAL per-question cascade-IRT difficulty, instead of the
RACE-subset label prepare_qde_data.py uses (RACE-middle->EASY, RACE-high->
MEDIUM, RACE-C->HARD -- a passage-level label, the same kind of confound
documented for diff-control-race in question_generation/docs/
training_details.md).

Reuses build_records() from question_generation/scripts/
prepare_pilot_irt_difficulty_data.py directly -- same 1,221 items (1,227
minus 6 single-question passages), same passage-mean-centering, same
tertile bucketing -- so this QDE pilot and the diff-control-irt-pilot QG
pilot train on IDENTICAL ground truth, not two independently-recomputed
versions that could silently drift apart.

Output schema matches prepare_qde_data.py's exactly, so the existing
train_encoder.py / evaluate_qde.py run against this data UNCHANGED:
  {"passage": "...", "question": "...", "answer": "...",
   "difficulty": "EASY|MEDIUM|HARD", "source": "..."}

Usage:
  python question_difficulty/scripts/prepare_qde_irt_pilot_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "question_generation/scripts"))

from prepare_pilot_irt_difficulty_data import build_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/qde_irt_pilot")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records()
    by_split = defaultdict(list)
    for r in records:
        split = r.pop("_split")
        r.pop("_item_id", None)
        r.pop("_centered_score", None)
        by_split[split].append(r)

    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        recs = by_split.get(split, [])
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(recs)} records to {path}")

    print("\ndifficulty distribution (train):", Counter(r["difficulty"] for r in by_split["train"]))
    print("difficulty distribution (val):  ", Counter(r["difficulty"] for r in by_split["val"]))
    print("difficulty distribution (test): ", Counter(r["difficulty"] for r in by_split["test"]))


if __name__ == "__main__":
    main()
