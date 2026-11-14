#!/usr/bin/env python3
"""One-off diagnostic for T5WithDifficultyFiLM against the real checkpoint:
confirm difficulty_ids=None matches the plain model exactly (identity init),
and that a freshly-initialized (zero) FiLM doesn't blow up loss."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diff_film_model import T5WithDifficultyFiLM, DIFF2ID

BASE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "question_generation/models/qg/flan-t5-base/baseline-race/final"
DATA_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/qg/diff-control-irt-10k-blind/train.jsonl"

print(f"Loading tokenizer + models from {BASE_MODEL}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
plain_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
film_model = T5WithDifficultyFiLM.from_pretrained(BASE_MODEL)
plain_model.eval()
film_model.eval()

records = [json.loads(l) for l in Path(DATA_FILE).open(encoding="utf-8")][:4]
inputs = tokenizer([r["input_text"] for r in records], max_length=1024, truncation=True,
                    padding=True, return_tensors="pt")
targets = tokenizer([r["target_text"] for r in records], max_length=64, truncation=True,
                     padding=True, return_tensors="pt")
labels = targets["input_ids"].clone()
labels[targets["attention_mask"] == 0] = -100
difficulty_ids = torch.tensor([DIFF2ID[r["difficulty"]] for r in records])

with torch.no_grad():
    out_plain = plain_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
    print(f"1. plain model loss:                    {out_plain.loss.item():.4f}")

    out_film_none = film_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                labels=labels, difficulty_ids=None)
    print(f"2. FiLM model loss (difficulty_ids=None): {out_film_none.loss.item():.4f}")

    out_film_zero = film_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                labels=labels, difficulty_ids=difficulty_ids)
    print(f"3. FiLM model loss (zero-init FiLM, ids set): {out_film_zero.loss.item():.4f}")

    print(f"\nlogits abs-max -- plain: {out_plain.logits.abs().max().item():.2f}, "
          f"FiLM(none): {out_film_none.logits.abs().max().item():.2f}, "
          f"FiLM(zero-init): {out_film_zero.logits.abs().max().item():.2f}")
    print(f"difficulty_film weight abs-sum: {film_model.difficulty_film.weight.abs().sum().item()}")

assert abs(out_plain.loss.item() - out_film_none.loss.item()) < 1e-3, "FiLM(None) should match plain exactly!"
assert abs(out_plain.loss.item() - out_film_zero.loss.item()) < 1e-3, "zero-init FiLM should be identity!"
print("\nALL CHECKS PASSED")
