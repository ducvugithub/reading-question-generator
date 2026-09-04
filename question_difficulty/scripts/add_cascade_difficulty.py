#!/usr/bin/env python3
"""
Estimate question difficulty using a cascade of QA models (weak → strong).

For each QA pair: run 3 QA models on (passage, question) and check if each
model predicts an answer with F1 > 0.5 against the ground truth.

  cascade score = fraction of models that answered correctly
  1.0 → all 3 correct → EASY
  0.67 → 2/3 correct → MEDIUM
  0.33 → 1/3 correct → HARD
  0.0  → none correct → VERY HARD

Result stored as `qa_cascade_diff` (float 0.0–1.0) in each record.

Usage:
  python scripts/data_prep/add_cascade_difficulty.py \
    --input data/raw_data/en/eval.jsonl \
    --limit 20 --verbose

  # Full run
  python scripts/data_prep/add_cascade_difficulty.py \
    --input data/raw_data/en/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from question_answering.qa_evaluator import QAEvaluator

_MODELS = [
    ("weak",   "distilbert-base-cased-distilled-squad"),
    ("medium", "deepset/roberta-base-squad2"),
    ("strong", "deepset/deberta-v3-base-squad2"),
]

_CORRECT_THRESHOLD = 0.5   # F1 above this = model answered correctly
_CHECKPOINT_EVERY  = 100

_evaluator = QAEvaluator()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Annotated JSONL file")
    parser.add_argument("--limit",    type=int,   default=None)
    parser.add_argument("--min-cog",  type=float, default=None, help="Only score records with haiku_cog >= this value")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--dry-run",  action="store_true", help="Don't write output")
    args = parser.parse_args()

    path = Path(args.input)
    records = _load_jsonl(path)

    # Optionally restrict to high-cog records only
    def _get_cog(r: dict) -> float:
        judge = r.get("llm_diff_judge") or {}
        scores = next(iter(judge.values()), {}) if judge else {}
        return scores.get("question_cognitive_diff", 0.0)

    already_done = sum(1 for r in records if "qa_cascade_diff" in r)
    todo = [
        i for i, r in enumerate(records)
        if "qa_cascade_diff" not in r
        and (args.min_cog is None or _get_cog(r) >= args.min_cog)
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Total: {len(records)} | already scored: {already_done} | to score: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    print("Loading QA pipelines...")
    from transformers import pipeline
    pipelines = []
    for label, model_id in _MODELS:
        print(f"  loading {label}: {model_id}")
        pipelines.append((label, pipeline("question-answering", model=model_id)))
    print("All models loaded.\n")

    for batch_start in range(0, len(todo), _CHECKPOINT_EVERY):
        batch = todo[batch_start : batch_start + _CHECKPOINT_EVERY]

        for i in tqdm(batch, desc=f"batch {batch_start//100 + 1}"):
            rec = records[i]
            passage  = rec.get("passage", "")
            question = rec.get("question", "")
            gold     = rec.get("answer", "")

            if not passage or not question or not gold:
                rec["qa_cascade_diff"] = None
                continue

            f1_scores: list[float] = []
            for label, qa in pipelines:
                try:
                    result = qa(question=question, context=passage, max_answer_len=50)
                    f1 = _evaluator.token_f1(result["answer"], gold)
                except Exception:
                    f1 = 0.0
                f1_scores.append(f1)

            n_correct = sum(f >= _CORRECT_THRESHOLD for f in f1_scores)
            score = n_correct / len(f1_scores)
            rec["qa_cascade_diff"] = round(score, 4)

            if args.verbose:
                labels = [_MODELS[j][0] for j in range(len(f1_scores))]
                f1_str = "  ".join(f"{labels[j]}={f1_scores[j]:.2f}" for j in range(len(f1_scores)))
                haiku_cog = (rec.get("llm_diff_judge") or {})
                haiku_cog = next(iter(haiku_cog.values()), {}).get("question_cognitive_diff", "?") if haiku_cog else "?"
                print(f"\nQ:  {question[:80]}")
                print(f"A:  {gold}")
                print(f"F1: {f1_str}  →  cascade={score:.2f}  haiku_cog={haiku_cog}")

        if not args.dry_run:
            _save_jsonl(path, records)
            done_so_far = already_done + batch_start + len(batch)
            print(f"  checkpoint saved ({done_so_far}/{len(records)})")

    # Summary
    scored = [r for r in records if isinstance(r.get("qa_cascade_diff"), float)]
    if scored:
        scores = [r["qa_cascade_diff"] for r in scored]
        easy   = sum(s > 2/3 for s in scores)
        medium = sum(1/3 < s <= 2/3 for s in scores)
        hard   = sum(s <= 1/3 for s in scores)
        print(f"\nDone. Scored {len(scored)} records.")
        print(f"  easy (>0.67):   {easy:5d}  ({easy/len(scores)*100:.1f}%)")
        print(f"  medium (0.33-0.67): {medium:5d}  ({medium/len(scores)*100:.1f}%)")
        print(f"  hard (≤0.33):   {hard:5d}  ({hard/len(scores)*100:.1f}%)")


if __name__ == "__main__":
    main()
