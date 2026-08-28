#!/usr/bin/env python3
"""
Step 1: Load raw RACE++ and HotpotQA, split into train/val/test, save as-is.

- RACE++ (RACE-middle→EASY, RACE-high→MEDIUM, RACE-C→HARD): 80/10/10 split by
  passage hash; the test split is then balanced so each difficulty contributes
  the same number of samples (overflow from majority difficulties is moved to
  train, so no data is dropped).
- HotpotQA (comparison-type only): 80/10/10 split by passage hash.

Downstream (prepare_qg_data.py) reads these raw splits and formats them per
model type — it does no splitting or matching of its own.

Usage:
  python question_generation/scripts/prepare_qg_test_sets.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator
from collections import defaultdict

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# remaining 0.1 is test


def _split_bucket(passage: str) -> str:
    h = int(hashlib.md5(passage.encode()).hexdigest(), 16) % 100
    if h < TRAIN_RATIO * 100:
        return "train"
    if h < (TRAIN_RATIO + VAL_RATIO) * 100:
        return "val"
    return "test"


def _iter_race(subset: str, difficulty: str) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", subset, split="train")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for rec in ds:
        idx = letter_to_idx.get(rec["answer"])
        if idx is None or idx >= len(rec["options"]):
            continue
        yield {
            "passage": rec["article"],
            "question": rec["question"],
            "difficulty": difficulty,
            "source": f"race_{subset}",
        }


def _iter_race_c() -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("tasksource/race-c", split="train")
    for rec in ds:
        if rec["label"] is None or rec["label"] >= len(rec["option"]):
            continue
        yield {
            "passage": rec["article"],
            "question": rec["question"],
            "difficulty": "HARD",
            "source": "race_c",
        }


def _gold_passage(rec: dict) -> str:
    gold_titles = set(rec["supporting_facts"]["title"])
    parts = []
    for title, sents in zip(rec["context"]["title"], rec["context"]["sentences"]):
        if title in gold_titles:
            parts.append(" ".join(sents))
    return " ".join(parts)


def _focus_span(rec: dict) -> str:
    """Extract supporting_facts sentences as the focus span."""
    sf_map: dict[str, set] = {}
    for title, sid in zip(rec["supporting_facts"]["title"], rec["supporting_facts"]["sent_id"]):
        sf_map.setdefault(title, set()).add(sid)
    parts = []
    for title, sents in zip(rec["context"]["title"], rec["context"]["sentences"]):
        if title in sf_map:
            for i, s in enumerate(sents):
                if i in sf_map[title]:
                    parts.append(s.strip())
    return " ".join(parts)


def _iter_hotpotqa_comparison() -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    for rec in ds:
        if rec["type"] != "comparison":
            continue
        passage = _gold_passage(rec)
        if not passage:
            continue
        focus = _focus_span(rec)
        if not focus:
            continue
        yield {"passage": passage, "question": rec["question"], "source": "hotpotqa", "focus": focus}


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def prepare_race(out_dir: Path) -> None:
    print("\n=== Splitting RACE++ (80/10/10, test balanced by difficulty) ===")
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for difficulty, iterator in [
        ("EASY",   _iter_race("middle", "EASY")),
        ("MEDIUM", _iter_race("high", "MEDIUM")),
        ("HARD",   _iter_race_c()),
    ]:
        count = 0
        for rec in iterator:
            splits[_split_bucket(rec["passage"])].append(rec)
            count += 1
        print(f"  Loaded {count} {difficulty} examples")

    # Balance the test split: cap each difficulty to the smallest count,
    # move overflow into train so no data is discarded.
    by_difficulty = defaultdict(list)
    for rec in splits["test"]:
        by_difficulty[rec["difficulty"]].append(rec)

    min_count = min(len(by_difficulty[d]) for d in ("EASY", "MEDIUM", "HARD"))
    balanced_test = []
    overflow = []
    for d in ("EASY", "MEDIUM", "HARD"):
        balanced_test.extend(by_difficulty[d][:min_count])
        overflow.extend(by_difficulty[d][min_count:])

    splits["test"] = balanced_test
    splits["train"].extend(overflow)

    print(f"  Balanced test: {min_count} × 3 = {len(balanced_test)} "
          f"(moved {len(overflow)} overflow records to train)")

    for name, records in splits.items():
        _write(out_dir / "race" / f"{name}.jsonl", records)
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def prepare_hotpotqa(out_dir: Path) -> None:
    print("\n=== Splitting HotpotQA comparison-only (80/10/10) ===")
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    count = 0
    for rec in _iter_hotpotqa_comparison():
        splits[_split_bucket(rec["passage"])].append(rec)
        count += 1
    print(f"  Loaded {count} comparison examples")

    for name, records in splits.items():
        _write(out_dir / "hotpotqa" / f"{name}.jsonl", records)
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def main() -> None:
    out_dir = Path("data/qg/raw")

    print("=" * 80)
    print("Step 1: Splitting raw datasets into train/val/test")
    print("=" * 80)

    prepare_race(out_dir)
    prepare_hotpotqa(out_dir)

    print("\n" + "=" * 80)
    print(f"Done. Raw splits saved under {out_dir}/{{race,hotpotqa}}/{{train,val,test}}.jsonl")
    print("=" * 80)


if __name__ == "__main__":
    main()
