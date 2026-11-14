#!/usr/bin/env python3
"""
Fine-tune T5 with FiLM-style structural difficulty conditioning (see
diff_film_model.py): per-decoder-layer scale/shift derived from the
difficulty tag, applied at every layer -- a stronger structural lever than
diff_cond_model.py's single additive bias before the LM head, which only
weakly broke beam search's tag-invariant collapse even with a 20x LR boost.

Usage:
  python question_generation/scripts/train_seq2seq_film.py \\
      --data-dir data/qg/diff-control-irt-10k-blind \\
      --base-model question_generation/models/qg/flan-t5-base/baseline-race/final \\
      --output-dir question_generation/models/qg_10k_blind_film \\
      --epochs 20 --batch-size 16 --lr 1e-4 --bf16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--early-stopping", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--film-lr-multiplier", type=float, default=50.0,
                         help="difficulty_film gets lr * this -- same lesson as diff_cond "
                              "training: a freshly initialized param needs a much higher LR "
                              "to move meaningfully against a fully pretrained model")
    args = parser.parse_args()

    print("Importing transformers...", flush=True)
    import torch
    from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
    from diff_film_model import T5WithDifficultyFiLM, DIFF2ID

    print(f"Loading {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = T5WithDifficultyFiLM.from_pretrained(args.base_model)

    data_dir = Path(args.data_dir)
    train_records = load_jsonl(data_dir / "train.jsonl")
    eval_records = load_jsonl(data_dir / "val.jsonl")
    if args.limit:
        train_records = train_records[:args.limit]
        eval_records = eval_records[:args.limit]
    print(f"Train: {len(train_records)} examples, Eval: {len(eval_records)} examples", flush=True)

    from datasets import Dataset

    def tokenize(batch):
        inputs = tokenizer(batch["input_text"], max_length=args.max_input, truncation=True, padding="max_length")
        targets = tokenizer(batch["target_text"], max_length=args.max_target, truncation=True, padding="max_length")
        labels = [[(t if t != tokenizer.pad_token_id else -100) for t in ids] for ids in targets["input_ids"]]
        inputs["labels"] = labels
        inputs["difficulty_ids"] = [DIFF2ID[d] for d in batch["difficulty"]]
        return inputs

    cols_to_remove = [c for c in ["input_text", "target_text", "source", "model_type", "difficulty"]
                      if c in train_records[0]]
    train_ds = Dataset.from_list(train_records).map(tokenize, batched=True, remove_columns=cols_to_remove)
    eval_ds = Dataset.from_list(eval_records).map(tokenize, batched=True, remove_columns=cols_to_remove)

    class DiffCondCollator:
        def __call__(self, features):
            return {
                "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
                "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
                "difficulty_ids": torch.tensor([f["difficulty_ids"] for f in features], dtype=torch.long),
            }

    out_dir = Path(args.output_dir)
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
        predict_with_generate=False,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        label_names=["labels"],
    )

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    film_params = [model.difficulty_film.weight]
    film_param_ids = {id(p) for p in film_params}
    base_params = [p for p in model.parameters() if id(p) not in film_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr, "weight_decay": 0.01},
            # No weight decay on difficulty_film: decay actively pulls gamma/beta
            # back toward the identity transform (gamma=1, beta=0) every step,
            # fighting the exact signal this param exists to grow.
            {"params": film_params, "lr": args.lr * args.film_lr_multiplier, "weight_decay": 0.0},
        ],
    )
    print(f"difficulty_film LR: {args.lr * args.film_lr_multiplier} (base LR: {args.lr})", flush=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DiffCondCollator(),
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}", flush=True)

    print(f"Training FiLM model -> {out_dir}", flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved to {out_dir}/final", flush=True)


if __name__ == "__main__":
    main()
