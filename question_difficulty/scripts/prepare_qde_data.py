#!/usr/bin/env python3
"""
Prepare Question Difficulty Estimator (QDE) training data.

All three classes come from the RACE family — same distribution, only exam level differs.
This avoids cross-dataset alignment problems and ensures the model learns genuine
difficulty signals rather than dataset-identity artifacts.

  EASY   — RACE-middle  (Chinese middle-school English exams)
  MEDIUM — RACE-high    (Chinese high-school English exams)
  HARD   — RACE-C       (Chinese college entrance / Gaokao, tasksource/race-c)

Sizes: middle ~25K, high ~62K, college ~12.7K.
Use --balanced to cap all sources at ~12.7K (RACE-C is the bottleneck).

Output records:
  {"passage": "...", "question": "...", "answer": "...",
   "difficulty": "EASY|MEDIUM|HARD", "source": "race_middle|race_high|race_c"}

Splits are deterministic at passage level via MD5 hashing (80/10/10).

Usage:
  python question_difficulty/scripts/prepare_qde_data.py
  python question_difficulty/scripts/prepare_qde_data.py --limit 500   # smoke test
  python question_difficulty/scripts/prepare_qde_data.py --balanced    # ~12.7K per class
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator


def _split_bucket(passage: str, train_r: float, val_r: float) -> str:
    h = int(hashlib.md5(passage.encode()).hexdigest(), 16) % 100
    if h < train_r * 100:
        return "train"
    if h < (train_r + val_r) * 100:
        return "val"
    return "test"


def _iter_race(subset: str, difficulty: str, limit: int | None) -> Iterator[dict]:
    """ehovy/race — middle and high subsets. Answer field: letter A/B/C/D."""
    from datasets import load_dataset

    ds = load_dataset("ehovy/race", subset, split="train")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    count = 0
    for rec in ds:
        idx = letter_to_idx.get(rec["answer"])
        if idx is None or idx >= len(rec["options"]):
            continue
        yield {
            "passage":    rec["article"],
            "question":   rec["question"],
            "answer":     rec["options"][idx],
            "difficulty": difficulty,
            "source":     f"race_{subset}",
        }
        count += 1
        if limit and count >= limit:
            break


def _iter_race_c(limit: int | None) -> Iterator[dict]:
    """tasksource/race-c — college level. Answer field: integer label 0-3."""
    from datasets import load_dataset

    ds = load_dataset("tasksource/race-c", split="train")
    count = 0
    for rec in ds:
        idx = rec["label"]
        if idx is None or idx >= len(rec["option"]):
            continue
        yield {
            "passage":    rec["article"],
            "question":   rec["question"],
            "answer":     rec["option"][idx],
            "difficulty": "HARD",
            "source":     "race_c",
        }
        count += 1
        if limit and count >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/qde")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records per source (smoke tests)")
    parser.add_argument("--balanced", action="store_true",
                        help="Cap each source at RACE-C size (~12.7K)")
    args = parser.parse_args()

    if abs(sum(args.split) - 1.0) > 0.01:
        print("Error: split ratios must sum to 1.0")
        sys.exit(1)

    train_r, val_r, _ = args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    # RACE-C train is ~12.7K — use as balance cap so classes are equal.
    cap = args.limit or (12_702 if args.balanced else None)

    sources = [
        ("RACE-middle → EASY",   _iter_race("middle", "EASY",   cap)),
        ("RACE-high   → MEDIUM", _iter_race("high",   "MEDIUM", cap)),
        ("RACE-C      → HARD",   _iter_race_c(cap)),
    ]

    for name, iterator in sources:
        print(f"Loading {name}...", flush=True)
        n = 0
        for rec in iterator:
            bucket = _split_bucket(rec["passage"], train_r, val_r)
            splits[bucket].append(rec)
            n += 1
        print(f"  {n} records", flush=True)

    for split_name, records in splits.items():
        if not records:
            continue
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in splits.values())
    print(
        f"\nDone: train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}  total={total}",
        flush=True,
    )
    print(f"Output: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
