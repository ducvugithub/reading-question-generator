#!/usr/bin/env python3
"""
Score (passage, generated question, answer) triples with the pretrained
regression EncoderQDE (question_difficulty/models/encoder_irt_pilot_regression)
to sanity-check whether the QG model's EASY/MEDIUM/HARD tags actually order
by predicted difficulty.

NOTE: this encoder was trained on irt_bayesian_model_only_difficulty.json
(the options-aware pilot cascade), NOT the blind-cascade signal the current
QG model is conditioned on -- a proxy check, not an exact match to the QG
model's own training signal. Treat the ordering as a directional sanity
check, not a precise difficulty estimate.

Input: jsonl with {"passage", "difficulty", "question", "answer"} per line
(see sample_rerank_diffcond.py --output and answer_rerank_outputs.py).

Usage:
  python question_difficulty/scripts/score_generated_difficulty.py \
    --input question_generation/results/sample_rerank_with_answers.jsonl \
    --model-dir question_difficulty/models/encoder_irt_pilot_regression/microsoft_deberta-v3-base \
    --base-model microsoft/deberta-v3-base
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from question_difficulty.methods.encoder.model import EncoderQDE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = EncoderQDE(args.base_model, n_classes=1)
    state = torch.load(Path(args.model_dir) / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    records = [json.loads(l) for l in (REPO_ROOT / args.input).open(encoding="utf-8") if l.strip()]

    by_difficulty = defaultdict(list)
    for rec in records:
        question_text = f"{rec['question']} [answer: {rec['answer']}]"
        enc = tokenizer(question_text, rec["passage"], max_length=args.max_len,
                         truncation=True, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            score = model(enc["input_ids"], enc["attention_mask"],
                           enc.get("token_type_ids")).squeeze().item()
        by_difficulty[rec["difficulty"]].append(score)
        print(f"[{rec['difficulty']:6s}] score={score:+.3f}  Q: {rec['question'][:70]}")

    print("\n=== Mean predicted difficulty_score by tag (higher = harder) ===")
    for d in ("EASY", "MEDIUM", "HARD"):
        scores = by_difficulty.get(d, [])
        if scores:
            print(f"  {d:6s}: mean={sum(scores)/len(scores):+.3f}  n={len(scores)}")


if __name__ == "__main__":
    main()
