#!/usr/bin/env python3
"""
Sample+rerank deployment strategy for the difficulty-conditioned model.

Earlier checks showed beam search reliably collapses to the same top-1
sequence regardless of the EASY/MEDIUM/HARD tag, even though --do-sample
proved the tag DOES shift the full output distribution. So: sample several
candidates per tag, then pick whichever candidate the model itself scores as
most SPECIFIC to that tag -- i.e. highest log-likelihood under its own tag
relative to the other two tags (a margin/contrastive rerank), rather than
relying on which single sequence has the highest raw probability (that's
exactly what beam search already does, and it collapses).

Uses the decoder-side additive-bias model (diff_cond_model.py), the
architecture that showed the best raw differentiation (4/8 test passages)
of everything tried so far.

Usage:
  python question_generation/scripts/sample_rerank_diffcond.py --n 8 --num-samples 8
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


def score_candidate(model, input_ids, candidate_ids, pad_token_id, tag_id) -> float:
    """Total log-likelihood of candidate_ids under the model conditioned on tag_id."""
    labels = candidate_ids.clone()
    labels[labels == pad_token_id] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels, difficulty_ids=torch.tensor([tag_id]))
    n_tokens = (labels != -100).sum().item()
    return -out.loss.item() * n_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="question_generation/models/qg_10k_blind_diffcond_v1_lowlr/final")
    parser.add_argument("--test-file", default="data/qg/raw/race_irt_10k_blind/test.jsonl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", default=None, help="optional path to write selected (passage, difficulty, question, margin) as jsonl")
    args = parser.parse_args()

    out_f = (REPO_ROOT / args.output).open("w", encoding="utf-8") if args.output else None

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

        for tag_idx, difficulty in enumerate(DIFFICULTIES):
            input_ids = tokenizer.encode(f"<{difficulty}> {passage}", return_tensors="pt",
                                          truncation=True, max_length=1024)
            with torch.no_grad():
                samples = model.generate(
                    input_ids=input_ids,
                    difficulty_ids=torch.tensor([tag_idx]),
                    max_length=64, do_sample=True, num_return_sequences=args.num_samples,
                    temperature=args.temperature, top_p=args.top_p,
                )

            best_text, best_margin = None, -1e9
            for cand in samples:
                cand_ids = cand.unsqueeze(0)
                own_score = score_candidate(model, input_ids, cand_ids, tokenizer.pad_token_id, tag_idx)
                other_scores = [score_candidate(model, input_ids, cand_ids, tokenizer.pad_token_id, j)
                                for j in range(3) if j != tag_idx]
                margin = own_score - max(other_scores)
                if margin > best_margin:
                    best_margin = margin
                    best_text = tokenizer.decode(cand, skip_special_tokens=True)

            print(f"  GENERATED [{difficulty:6s}] (margin={best_margin:.2f}): {best_text}")
            if out_f is not None:
                out_f.write(json.dumps({
                    "passage": passage, "difficulty": difficulty,
                    "question": best_text, "margin": best_margin,
                }, ensure_ascii=False) + "\n")

    if out_f is not None:
        out_f.close()


if __name__ == "__main__":
    main()
