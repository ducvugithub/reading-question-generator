#!/usr/bin/env python3
"""
Convert annotated JSONL → T5 input/output format for M1–M4.

Accepts one or more JSONL input files, combines them, and deterministically
splits by passage into train/val/test using MD5 hashing.

Input formats per model type:
  M1: generate question: passage: {passage} answer: {answer}
  M2: generate question: difficulty: {EASY|MEDIUM|HARD} passage: {passage} answer: {answer}
  M3: generate question: passage: {passage} answer: {answer} triples: {s|r|o ; ...}
  M4: generate question: difficulty: {EASY|MEDIUM|HARD} passage: {passage} answer: {answer} triples: {s|r|o ; ...}

Usage:
  # Both train and eval as input, auto-split 80/10/10
  python scripts/data_prep/prepare_t5_inputs.py \\
    --inputs data/training/en/train.jsonl data/training/en/eval.jsonl \\
    --model-types m1 m2 m3 m4

  # Only annotated eval set, split 80/10/10
  python scripts/data_prep/prepare_t5_inputs.py \\
    --inputs data/training/en/eval.jsonl \\
    --model-types m1 m2 m3 m4

  # Custom split ratios (train/val/test)
  python scripts/data_prep/prepare_t5_inputs.py \\
    --inputs data/training/en/eval.jsonl \\
    --split 0.8 0.1 0.1 \\
    --model-types m1 m2 m3 m4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_DEFAULT_JUDGE = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
_MAX_TRIPLES = 15


def _split_bucket(passage: str, train_ratio: float, val_ratio: float) -> str:
    """Deterministic passage-level split via MD5. Returns 'train', 'val', or 'test'."""
    h = int(hashlib.md5(passage.encode()).hexdigest(), 16) % 100
    if h < train_ratio * 100:
        return "train"
    if h < (train_ratio + val_ratio) * 100:
        return "val"
    return "test"


def _difficulty_label(cog_score: float) -> str:
    if cog_score < 1 / 3:
        return "EASY"
    if cog_score < 2 / 3:
        return "MEDIUM"
    return "HARD"


def _format_triples(kg_coref: list | None, kg_raw: list | None) -> str:
    triples = kg_coref or kg_raw or []
    return " ; ".join(f"{t[0]} | {t[1]} | {t[2]}" for t in triples[:_MAX_TRIPLES])


def _format_input(model_type: str, passage: str, answer: str, difficulty: str, triples_str: str) -> str:
    base = f"passage: {passage} answer: {answer}"
    if model_type == "m1":
        return f"generate question: {base}"
    if model_type == "m2":
        return f"generate question: difficulty: {difficulty} {base}"
    if model_type == "m3":
        return f"generate question: {base} triples: {triples_str}"
    if model_type == "m4":
        return f"generate question: difficulty: {difficulty} {base} triples: {triples_str}"
    raise ValueError(f"Unknown model type: {model_type}")


def prepare(args: argparse.Namespace) -> None:
    train_r, val_r, test_r = args.split

    # Load all records from all input files
    all_records: list[dict] = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"  [warn] {path} not found, skipping")
            continue
        records = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        all_records.extend(records)
        print(f"  loaded {len(records)} records from {path}")

    print(f"Total records loaded: {len(all_records)}")

    for model_type in args.model_types:
        out_dir = Path(args.output_dir) / model_type
        out_dir.mkdir(parents=True, exist_ok=True)

        splits = {"train": [], "val": [], "test": []}
        n_skipped = 0

        for rec in all_records:
            judge_scores = rec.get("llm_diff_judge", {}).get(args.judge)
            if judge_scores is None:
                n_skipped += 1
                continue

            cog = judge_scores["question_cognitive_diff"]
            difficulty = _difficulty_label(cog)
            triples_str = _format_triples(rec.get("kg_coref"), rec.get("kg_raw"))

            out_rec = {
                "input_text":  _format_input(model_type, rec["passage"], rec["answer"], difficulty, triples_str),
                "target_text": rec["question"],
                "difficulty":  difficulty.lower(),
                "lang":        rec.get("lang", "?"),
                "generated":   rec.get("generated", False),
            }

            bucket = _split_bucket(rec["passage"], train_r, val_r)
            splits[bucket].append(out_rec)

        for split_name, records in splits.items():
            if not records:
                continue
            out_path = out_dir / f"{split_name}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total = sum(len(v) for v in splits.values())
        print(
            f"[{model_type}] train={len(splits['train'])} val={len(splits['val'])} "
            f"test={len(splits['test'])} skipped={n_skipped} total={total}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",      nargs="+", required=True,
                        help="One or more annotated JSONL files")
    parser.add_argument("--model-types", nargs="+", default=["m1", "m2", "m3", "m4"],
                        choices=["m1", "m2", "m3", "m4"])
    parser.add_argument("--split",       nargs=3,   type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test ratios (default: 0.8 0.1 0.1)")
    parser.add_argument("--output-dir",  default="data/t5")
    parser.add_argument("--judge",       default=_DEFAULT_JUDGE)
    args = parser.parse_args()

    total = sum(args.split)
    if abs(total - 1.0) > 0.01:
        print(f"Error: split ratios must sum to 1.0 (got {total})")
        sys.exit(1)

    prepare(args)


if __name__ == "__main__":
    main()
