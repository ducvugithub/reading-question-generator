#!/usr/bin/env python3
"""
Zero-shot LLM-as-judge Question Difficulty Estimator.

No training required — prompts a Claude model to rate each (passage, question, answer)
triple as EASY / MEDIUM / HARD, then reports macro F1 against RACE curriculum labels.

This is the zero-shot baseline; all other QDE methods (feature-based, encoder,
contrastive) are trained on the same RACE labels. Comparing macro F1 tells you how
much fine-tuning adds over a capable LLM's zero-shot judgment.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python question_difficulty/scripts/evaluate_llm_verdict.py
  python question_difficulty/scripts/evaluate_llm_verdict.py --split val --limit 200
  python question_difficulty/scripts/evaluate_llm_verdict.py --model claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_LABEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
_LABEL_NAMES = ["EASY", "MEDIUM", "HARD"]

_SYSTEM_PROMPT = """\
You are an expert in reading comprehension assessment.
Given a passage, a question, and the correct answer, classify the question difficulty.

Respond with EXACTLY one word on the first line: EASY, MEDIUM, or HARD.
Then on the next lines give a one-sentence justification.

Difficulty guide:
  EASY   — Factual recall; answer is directly stated in the passage; simple vocabulary.
  MEDIUM — Requires inference or connecting two pieces of information; moderate vocabulary.
  HARD   — Requires multi-step reasoning, synthesis, or abstract/critical thinking; complex vocabulary.
"""

_USER_TEMPLATE = """\
PASSAGE:
{passage}

QUESTION: {question}

ANSWER: {answer}

Difficulty (EASY / MEDIUM / HARD):"""


def _parse_label(text: str) -> int | None:
    """Extract the first occurrence of EASY/MEDIUM/HARD from the response."""
    m = re.search(r'\b(EASY|MEDIUM|HARD)\b', text.upper())
    if m:
        return _LABEL_MAP[m.group(1).lower()]
    return None


def evaluate_split(
    records: list[dict],
    client,
    model: str,
    max_passage_words: int,
    delay: float,
) -> tuple[list[int], list[int]]:
    """Call the LLM for each record; return (true_labels, pred_labels)."""
    true_labels, pred_labels = [], []
    n = len(records)

    for i, rec in enumerate(records):
        true_label = _LABEL_MAP.get(rec["difficulty"].lower())
        if true_label is None:
            continue

        # Truncate very long passages to save tokens
        words = rec["passage"].split()
        passage = " ".join(words[:max_passage_words]) if len(words) > max_passage_words else rec["passage"]

        user_msg = _USER_TEMPLATE.format(
            passage=passage,
            question=rec["question"],
            answer=rec["answer"],
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=64,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
            pred = _parse_label(text)
        except Exception as e:
            print(f"  [!] API error on record {i}: {e}", flush=True)
            pred = None

        if pred is None:
            print(f"  [!] Unparseable response on record {i}: {text!r}", flush=True)
            pred = 1  # fallback to MEDIUM

        true_labels.append(true_label)
        pred_labels.append(pred)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} done", flush=True)

        if delay > 0:
            time.sleep(delay)

    return true_labels, pred_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",    default="data/qde")
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--model",       default="claude-haiku-4-5",
                        help="Claude model ID (default: claude-haiku-4-5)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max records to evaluate (cost control)")
    parser.add_argument("--max-passage-words", type=int, default=400,
                        help="Truncate passages to this many words")
    parser.add_argument("--delay",       type=float, default=0.1,
                        help="Seconds between API calls (rate limiting)")
    parser.add_argument("--output",      default=None,
                        help="Write per-record predictions to this JSONL file")
    args = parser.parse_args()

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    if not data_path.exists():
        print(f"Error: {data_path} not found — run prepare_qde_data.py first")
        sys.exit(1)

    records = [json.loads(l) for l in data_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        import random
        random.Random(42).shuffle(records)
        records = records[:args.limit]

    print(f"Model : {args.model}")
    print(f"Split : {args.split}  ({len(records)} records)")
    print(f"Evaluating...", flush=True)

    true_labels, pred_labels = evaluate_split(
        records, client, args.model, args.max_passage_words, args.delay
    )

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    print(f"\nMacro F1: {macro_f1:.4f}")
    print(classification_report(true_labels, pred_labels, target_names=_LABEL_NAMES))

    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':8}  {'EASY':>6}  {'MEDIUM':>6}  {'HARD':>6}")
    for i, name in enumerate(_LABEL_NAMES):
        print(f"  {name:8}  {cm[i][0]:6}  {cm[i][1]:6}  {cm[i][2]:6}")

    # Label distribution breakdown (shows if LLM is biased toward one class)
    from collections import Counter
    pred_dist = Counter(pred_labels)
    true_dist = Counter(true_labels)
    print("\nDistribution  true / predicted:")
    for j, name in enumerate(_LABEL_NAMES):
        print(f"  {name:8}  true={true_dist[j]:5}  pred={pred_dist[j]:5}")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            for rec, true, pred in zip(records, true_labels, pred_labels):
                f.write(json.dumps({
                    "question":  rec["question"],
                    "difficulty": rec["difficulty"],
                    "true_label": _LABEL_NAMES[true],
                    "pred_label": _LABEL_NAMES[pred],
                    "correct": true == pred,
                }, ensure_ascii=False) + "\n")
        print(f"\nPredictions written to {out_path}")


if __name__ == "__main__":
    main()
