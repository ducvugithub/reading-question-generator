#!/usr/bin/env python3
"""
Prepare Question Difficulty Estimator (QDE) training data.

Combines three curriculum-grounded difficulty sources:
  EASY   — SQuAD 2.0  (span-locatable; BERT-base 88% F1)
  MEDIUM — RACE-middle (Chinese middle-school English exams; BERT-base ~69%)
  HARD   — RACE-high   (Chinese high-school English exams; BERT-base ~63%)

Output records:
  {"passage": "...", "question": "...", "answer": "...",
   "difficulty": "EASY|MEDIUM|HARD", "source": "squad|race_middle|race_high"}

Splits are deterministic at passage level via MD5 hashing (80/10/10).

Usage:
  python question_difficulty/scripts/prepare_qde_data.py
  python question_difficulty/scripts/prepare_qde_data.py --limit 500  # smoke test
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


def _iter_squad(limit: int | None) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="train")
    count = 0
    for rec in ds:
        if not rec["answers"]["text"]:  # skip unanswerable questions
            continue
        yield {
            "passage":    rec["context"],
            "question":   rec["question"],
            "answer":     rec["answers"]["text"][0],
            "difficulty": "EASY",
            "source":     "squad",
        }
        count += 1
        if limit and count >= limit:
            break


def _iter_race(subset: str, difficulty: str, limit: int | None) -> Iterator[dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/qde")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records per source (for smoke tests)")
    parser.add_argument("--balanced", action="store_true",
                        help="Cap each source at the size of the smallest (RACE-middle ~25K)")
    args = parser.parse_args()

    if abs(sum(args.split) - 1.0) > 0.01:
        print("Error: split ratios must sum to 1.0")
        sys.exit(1)

    train_r, val_r, _ = args.split
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    # --balanced caps all sources at RACE-middle size (~25K) for equal class representation
    cap = args.limit or (25_421 if args.balanced else None)

    sources = [
        ("SQuAD → EASY",        _iter_squad(cap)),
        ("RACE-middle → MEDIUM", _iter_race("middle", "MEDIUM", cap)),
        ("RACE-high → HARD",     _iter_race("high",   "HARD",   cap)),
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
