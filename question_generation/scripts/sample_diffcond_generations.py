#!/usr/bin/env python3
"""
Qualitative check for the STRUCTURAL difficulty-conditioned model
(T5WithDifficultyConditioning, see diff_cond_model.py / train_seq2seq_diffcond.py):
same held-out-passage comparison as sample_irt_10k_generations.py, but passing
difficulty_ids directly to generate() instead of relying on the input tag.

Usage:
  python question_generation/scripts/sample_diffcond_generations.py --n 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoTokenizer
from diff_cond_model import T5WithDifficultyConditioning, DIFF2ID

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="question_generation/models/qg_10k_blind_diffcond/final")
    parser.add_argument("--test-file", default="data/qg/raw/race_irt_10k_blind/test.jsonl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    records = [json.loads(l) for l in (REPO_ROOT / args.test_file).open(encoding="utf-8") if l.strip()]
    by_passage: dict[str, list[dict]] = {}
    for r in records:
        by_passage.setdefault(r["passage"], []).append(r)

    passages = list(by_passage.keys())
    random.Random(args.seed).shuffle(passages)
    picked = passages[:args.n]

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = T5WithDifficultyConditioning.from_pretrained(args.model_path)
    model.eval()

    for passage in picked:
        recs = by_passage[passage]
        print("\n" + "=" * 100)
        print(f"PASSAGE (first 300 chars): {passage[:300]}...")
        print("ORIGINAL QUESTIONS IN THIS PASSAGE (true bucket -> question):")
        for r in recs:
            print(f"  [{r['difficulty']:6s}] {r['question']}")
        print("-" * 100)
        for difficulty in DIFFICULTIES:
            # Training data still has the <EASY>/<MEDIUM>/<HARD> input tag too
            # (kept for redundant signal) -- match that at inference.
            input_ids = tokenizer.encode(f"<{difficulty}> {passage}", return_tensors="pt",
                                          truncation=True, max_length=1024)
            difficulty_ids = torch.tensor([DIFF2ID[difficulty]])
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, difficulty_ids=difficulty_ids,
                                          max_length=64, num_beams=args.num_beams)
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  GENERATED [{difficulty:6s}]: {question}")


if __name__ == "__main__":
    main()
