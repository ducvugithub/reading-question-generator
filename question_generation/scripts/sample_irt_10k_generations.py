#!/usr/bin/env python3
"""
Quick qualitative check for diff-control-irt-10k: for a handful of held-out
test passages, generate a question at EACH of EASY/MEDIUM/HARD and print them
side by side with the original question and its true (cascade-IRT) bucket.

Not the rigorous test (that's running generated questions back through the
answerer cascade to check for a real accuracy gradient) -- this is just "does
forcing a different tag visibly change what gets generated at all."

Usage:
  python question_generation/scripts/sample_irt_10k_generations.py --n 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]


def guided_generate(model, tokenizer, passage: str, difficulty: str, guidance_scale: float,
                     max_length: int = 64) -> str:
    """Classifier-free-guidance-style decoding: amplify the logit delta the
    difficulty tag causes (tagged vs untagged input) instead of trusting the
    model's own (small) weighting of the tag against the much longer passage."""
    tagged_ids = tokenizer(f"<{difficulty}> {passage}", return_tensors="pt",
                            truncation=True, max_length=1024).input_ids
    untagged_ids = tokenizer(passage, return_tensors="pt",
                              truncation=True, max_length=1024).input_ids
    with torch.no_grad():
        enc_cond = model.get_encoder()(tagged_ids)
        enc_uncond = model.get_encoder()(untagged_ids)

    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
    for _ in range(max_length):
        with torch.no_grad():
            out_cond = model(encoder_outputs=enc_cond, decoder_input_ids=decoder_input_ids)
            out_uncond = model(encoder_outputs=enc_uncond, decoder_input_ids=decoder_input_ids)
        logits_cond = out_cond.logits[:, -1, :]
        logits_uncond = out_uncond.logits[:, -1, :]
        boosted = logits_cond + (guidance_scale - 1.0) * (logits_cond - logits_uncond)
        next_token = boosted.argmax(dim=-1, keepdim=True)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="question_generation/models/qg_10k/final/diff-control-irt-10k/final")
    parser.add_argument("--test-file", default="data/qg/raw/race_irt_10k/test.jsonl")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true",
                         help="Sample instead of beam search -- reveals whether the difficulty "
                              "token shifts the model's output DISTRIBUTION, not just its single "
                              "most-likely (beam search) answer")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                         help="CFG-style amplification of the tag's effect on logits "
                              "(tagged vs untagged delta). 1.0 = off (normal decoding).")
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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
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
            if args.guidance_scale != 1.0:
                question = guided_generate(model, tokenizer, passage, difficulty, args.guidance_scale)
                print(f"  GENERATED [{difficulty:6s}] (guidance={args.guidance_scale}): {question}")
                continue
            input_text = f"<{difficulty}> {passage}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                if args.do_sample:
                    outputs = model.generate(
                        input_ids, max_length=64, do_sample=True,
                        num_return_sequences=args.num_samples,
                        temperature=args.temperature, top_p=args.top_p,
                    )
                else:
                    outputs = model.generate(input_ids, max_length=64, num_beams=4)
            questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if args.do_sample:
                print(f"  GENERATED [{difficulty:6s}]:")
                for q in questions:
                    print(f"      - {q}")
            else:
                print(f"  GENERATED [{difficulty:6s}]: {questions[0]}")


if __name__ == "__main__":
    main()
