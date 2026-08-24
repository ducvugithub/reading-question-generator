#!/usr/bin/env python3
"""
Fine-tune T5 for question generation.

Reads data/qg/{step}/{split}.jsonl produced by prepare_qg_data.py
and fine-tunes t5-base using HuggingFace Seq2SeqTrainer.

Steps:
  baseline             — baseline QG: context → question (no conditioning)
  diff-control         — difficulty-controlled QG: difficulty + context → question
  focus-span-control   — focus-span QG: focus + context → question
  step4                — M6 full: difficulty + focus + context → question (QDE-enriched data)

Usage:
  python question_generation/scripts/train_seq2seq.py --model-type baseline
  python question_generation/scripts/train_seq2seq.py --model-type diff-control --epochs 5
  python question_generation/scripts/train_seq2seq.py --model-type baseline --limit 200 --epochs 1  # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_hf_dataset(train_records: list[dict], eval_records: list[dict], tokenizer, max_input_len: int, max_target_len: int):
    from datasets import Dataset

    def tokenize(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            batch["target_text"],
            max_length=max_target_len,
            truncation=True,
            padding="max_length",
        )
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in targets["input_ids"]
        ]
        inputs["labels"] = labels
        return inputs

    cols_to_remove = [c for c in ["input_text", "target_text", "difficulty", "source", "step"]
                      if c in train_records[0]]
    train_ds = Dataset.from_list(train_records).map(tokenize, batched=True, remove_columns=cols_to_remove)
    eval_ds  = Dataset.from_list(eval_records).map(tokenize, batched=True, remove_columns=cols_to_remove)
    return train_ds, eval_ds


def compute_metrics_fn(tokenizer):
    import numpy as np
    try:
        import evaluate
        rouge = evaluate.load("rouge")
    except Exception:
        rouge = None

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        if rouge:
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            return {k: round(v, 4) for k, v in result.items()}
        return {}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type",  required=True,
                        choices=["baseline", "diff-control", "focus-span-control", "step4"])
    parser.add_argument("--data-dir",    default="data/qg")
    parser.add_argument("--output-dir",  default="question_generation/models/qg")
    parser.add_argument("--base-model",  default="t5-base")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--max-input",   type=int,   default=512)
    parser.add_argument("--max-target",  type=int,   default=64)
    parser.add_argument("--save-steps",  type=int,   default=500, help="Eval/log/save every N steps (use small value for smoke tests)")
    parser.add_argument("--fp16",        action="store_true", help="Enable mixed precision (GPU only)")
    parser.add_argument("--limit",            type=int, default=None, help="Use only first N examples (for testing)")
    parser.add_argument("--early-stopping",   type=int, default=3,    help="Stop after N evals with no improvement (0 to disable)")
    args = parser.parse_args()

    print(f"Model type: {args.model_type.upper()} | limit: {args.limit}", flush=True)
    print("Importing transformers...", flush=True)
    from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(args.base_model, "config.json")
    if cached is None:
        print(f"Downloading {args.base_model} from HuggingFace (~900MB)...", flush=True)
    else:
        print(f"Loading {args.base_model} from cache", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    model     = T5ForConditionalGeneration.from_pretrained(args.base_model)

    data_dir = Path(args.data_dir) / args.model_type
    train_records, eval_records = [], []

    train_path = data_dir / "train.jsonl"
    eval_path  = data_dir / "val.jsonl"
    print(f"Data: {data_dir}", flush=True)
    if train_path.exists():
        records = load_jsonl(train_path)
        train_records = records[:args.limit] if args.limit else records
    if eval_path.exists():
        records = load_jsonl(eval_path)
        eval_records = records[:args.limit] if args.limit else records

    print(f"Train: {len(train_records)} examples, Eval: {len(eval_records)} examples", flush=True)

    train_ds, eval_ds = build_hf_dataset(train_records, eval_records, tokenizer, args.max_input, args.max_target)

    out_dir = Path(args.output_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=200,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target,
        fp16=args.fp16,
        logging_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=callbacks,
    )

    print(f"Training {args.model_type.upper()} → {out_dir}", flush=True)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved to {out_dir}/final", flush=True)


if __name__ == "__main__":
    main()
