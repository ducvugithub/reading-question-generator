#!/usr/bin/env python3
"""
Fine-tune T5 for difficulty-controlled question generation using adapter modules.

One Pfeiffer bottleneck adapter per difficulty level (EASY/MEDIUM/HARD), applied
in the T5 encoder/decoder stack, instead of the `<EASY|MEDIUM|HARD>` text-prefix
approach used by diff-control-race. Reuses data/qg/baseline-race/{split}.jsonl —
it already has plain-passage input_text and a difficulty field, which is exactly
what adapter routing needs.

Model type: adapter-control-race

Usage:
  python question_generation/scripts/train_adapter_qg.py
  python question_generation/scripts/train_adapter_qg.py --freeze-base=false --epochs 5
  python question_generation/scripts/train_adapter_qg.py --limit 200 --epochs 1  # smoke test
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_seq2seq import compute_metrics_fn, load_jsonl  # noqa: E402

MODEL_TYPE = "adapter-control-race"
DIFFICULTIES = ["easy", "medium", "hard"]


def build_hf_dataset_with_difficulty(records: list[dict], tokenizer, max_input_len: int, max_target_len: int):
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
        inputs["difficulty"] = [d.lower() for d in batch["difficulty"]]
        return inputs

    cols_to_remove = [c for c in ["input_text", "target_text", "source", "model_type"] if c in records[0]]
    return Dataset.from_list(records).map(tokenize, batched=True, remove_columns=cols_to_remove)


class DifficultyGroupedBatchSampler:
    """Groups indices by difficulty so every batch is homogeneous.

    Needed because the `adapters` library activates one adapter per forward
    pass (model.set_active_adapters), not per-example — mixed-difficulty
    batches can't be routed to different adapters within a single batch.
    """

    def __init__(self, difficulties: list[str], batch_size: int, shuffle: bool, seed: int = 0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.groups: dict[str, list[int]] = {}
        for i, d in enumerate(difficulties):
            self.groups.setdefault(d, []).append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for idxs in self.groups.values():
            idxs = idxs[:]
            if self.shuffle:
                rng.shuffle(idxs)
            batches.extend(idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum((len(idxs) + self.batch_size - 1) // self.batch_size for idxs in self.groups.values())


@dataclass
class DifficultyGroupedCollator:
    """Wraps a base collator; strips the raw `difficulty` string field before
    tokenizer padding (it isn't a tensor field) and reattaches it as
    `difficulty_group` on the padded batch for the trainer to read."""

    base_collator: object

    def __call__(self, features: list[dict]):
        difficulties = [f.pop("difficulty") for f in features]
        assert len(set(difficulties)) == 1, f"non-homogeneous batch: {set(difficulties)}"
        batch = self.base_collator(features)
        batch["difficulty_group"] = difficulties[0]
        return batch


def build_trainer_class():
    from torch.utils.data import DataLoader
    from adapters import Seq2SeqAdapterTrainer

    class DifficultyAdapterTrainer(Seq2SeqAdapterTrainer):
        def __init__(self, *args, train_difficulties=None, eval_difficulties=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_difficulties = train_difficulties
            self.eval_difficulties = eval_difficulties

        def get_train_dataloader(self):
            sampler = DifficultyGroupedBatchSampler(
                self.train_difficulties, self.args.per_device_train_batch_size, shuffle=True
            )
            return DataLoader(self.train_dataset, batch_sampler=sampler, collate_fn=self.data_collator)

        def get_eval_dataloader(self, eval_dataset=None):
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            sampler = DifficultyGroupedBatchSampler(
                self.eval_difficulties, self.args.per_device_eval_batch_size, shuffle=False
            )
            return DataLoader(eval_dataset, batch_sampler=sampler, collate_fn=self.data_collator)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            difficulty = inputs.pop("difficulty_group")
            model.set_active_adapters(difficulty)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
            difficulty = inputs.pop("difficulty_group")
            model.set_active_adapters(difficulty)
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **kwargs
            )

    return DifficultyAdapterTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="data/qg")
    parser.add_argument("--data-model-type", default="baseline-race",
                        help="Which data/qg/<type> dir to read from — plain-passage input "
                             "with a difficulty field, e.g. baseline-race")
    parser.add_argument("--output-dir",  default="question_generation/models/qg")
    parser.add_argument("--base-model",  default="t5-base")
    parser.add_argument("--reduction-factor", type=int, default=16)
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze base T5 weights and train only the adapters (parameter-efficient). "
                             "--no-freeze-base fine-tunes base + adapters together.")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--max-input",   type=int,   default=512)
    parser.add_argument("--max-target",  type=int,   default=64)
    parser.add_argument("--save-steps",  type=int,   default=500)
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--bf16",        action="store_true")
    parser.add_argument("--limit",            type=int, default=None)
    parser.add_argument("--early-stopping",   type=int, default=3)
    args = parser.parse_args()

    print(f"Model type: {MODEL_TYPE.upper()} | freeze_base={args.freeze_base} | limit: {args.limit}", flush=True)
    print("Importing transformers/adapters...", flush=True)
    from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, \
        DataCollatorForSeq2Seq, EarlyStoppingCallback
    import adapters
    from adapters import AdapterConfig

    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(args.base_model, "config.json")
    print(f"{'Loading from cache' if cached else 'Downloading'}: {args.base_model}", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    adapters.init(model)
    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=args.reduction_factor)
    for level in DIFFICULTIES:
        model.add_adapter(level, config=adapter_config)
    if args.freeze_base:
        model.train_adapter(DIFFICULTIES)
    else:
        model.set_active_adapters(None)  # avoid stacking all three; per-batch routing sets this later

    data_dir = Path(args.data_dir) / args.data_model_type
    train_path, eval_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    print(f"Data: {data_dir}", flush=True)

    train_records = load_jsonl(train_path)
    eval_records = load_jsonl(eval_path)
    if args.limit:
        train_records, eval_records = train_records[:args.limit], eval_records[:args.limit]
    for rec in train_records + eval_records:
        if not rec.get("difficulty"):
            raise ValueError(
                f"{data_dir} has no `difficulty` field per record — adapter routing needs it. "
                "Pick a --data-model-type sourced from RACE (e.g. baseline-race)."
            )

    print(f"Train: {len(train_records)} examples, Eval: {len(eval_records)} examples", flush=True)

    train_ds = build_hf_dataset_with_difficulty(train_records, tokenizer, args.max_input, args.max_target)
    eval_ds = build_hf_dataset_with_difficulty(eval_records, tokenizer, args.max_input, args.max_target)
    train_difficulties = list(train_ds["difficulty"])
    eval_difficulties = list(eval_ds["difficulty"])

    base_model_slug = args.base_model.split("/")[-1]
    out_dir = Path(args.output_dir) / base_model_slug / MODEL_TYPE
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
        bf16=args.bf16,
        logging_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # keep the `difficulty` column alive through the collator
        report_to="none",
    )

    base_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    collator = DifficultyGroupedCollator(base_collator)

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    DifficultyAdapterTrainer = build_trainer_class()
    trainer = DifficultyAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=callbacks,
        train_difficulties=train_difficulties,
        eval_difficulties=eval_difficulties,
    )

    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}", flush=True)

    print(f"Training {MODEL_TYPE.upper()} → {out_dir}", flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    model.save_all_adapters(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved base model + adapters ({', '.join(DIFFICULTIES)}) to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
