#!/usr/bin/env python3
"""One-off diagnostic: isolate why T5WithDifficultyConditioning's loss exploded
(~8.5e4 vs ~1.4 for the plain tag-only model) on the same checkpoint/data.

Compares, on the same small batch:
  1. plain AutoModelForSeq2SeqLM.forward(..., labels=...)            [reference]
  2. T5WithDifficultyConditioning.forward(..., difficulty_ids=None)   [bias off]
  3. T5WithDifficultyConditioning.forward(..., difficulty_ids=<ids>)  [bias on]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diff_cond_model import T5WithDifficultyConditioning, DIFF2ID

BASE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "question_generation/models/qg/flan-t5-base/baseline-race/final"
DATA_FILE = sys.argv[2] if len(sys.argv) > 2 else "data/qg/diff-control-irt-10k-blind/train.jsonl"

print(f"Loading tokenizer + models from {BASE_MODEL}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
plain_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
diffcond_model = T5WithDifficultyConditioning.from_pretrained(BASE_MODEL)
plain_model.eval()
diffcond_model.eval()

records = [json.loads(l) for l in Path(DATA_FILE).open(encoding="utf-8")][:4]
inputs = tokenizer([r["input_text"] for r in records], max_length=1024, truncation=True,
                    padding=True, return_tensors="pt")
targets = tokenizer([r["target_text"] for r in records], max_length=64, truncation=True,
                     padding=True, return_tensors="pt")
labels = targets["input_ids"].clone()
labels[targets["attention_mask"] == 0] = -100
difficulty_ids = torch.tensor([DIFF2ID[r["difficulty"]] for r in records])

print(f"plain  config.tie_word_embeddings = {plain_model.config.tie_word_embeddings}, "
      f"lm_head is shared tensor: {plain_model.lm_head.weight.data_ptr() == plain_model.shared.weight.data_ptr()}")
print(f"diffcond config.tie_word_embeddings = {diffcond_model.config.tie_word_embeddings}, "
      f"lm_head is shared tensor: {diffcond_model.lm_head.weight.data_ptr() == diffcond_model.shared.weight.data_ptr()}")

with torch.no_grad():
    out_plain = plain_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
    print(f"1. plain model loss:                      {out_plain.loss.item():.4f}")

    out_diffcond_nobias = diffcond_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                          labels=labels, difficulty_ids=None)
    print(f"2. diffcond model loss (difficulty_ids=None): {out_diffcond_nobias.loss.item():.4f}")

    out_diffcond_bias = diffcond_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                        labels=labels, difficulty_ids=difficulty_ids)
    print(f"3. diffcond model loss (difficulty_ids set):  {out_diffcond_bias.loss.item():.4f}")

    # Extra: logits magnitude comparison
    print(f"\nlogits abs-max -- plain: {out_plain.logits.abs().max().item():.2f}, "
          f"diffcond(no bias): {out_diffcond_nobias.logits.abs().max().item():.2f}, "
          f"diffcond(bias): {out_diffcond_bias.logits.abs().max().item():.2f}")
    print(f"difficulty_embedding weight norm per class: "
          f"{diffcond_model.difficulty_embedding.weight.norm(dim=1).tolist()}")
