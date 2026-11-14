#!/usr/bin/env python3
"""
Fine-tune T5 with FiLM-style difficulty conditioning (see diff_film_model.py)
PLUS a direct anti-collapse term: for every training example, decode the SAME
teacher-forced target sequence three times (once per difficulty class,
reusing one encoder pass), and penalize how similar the three resulting
token-probability distributions are -- directly optimizing the thing
`eval_film_diff_quality.py` measures (collapse rate), instead of relying on
difficulty-specific gradient routing alone to produce divergence as a side
effect (see question_generation/docs/difficulty_steering_mechanisms.md's
results table: v3-v6 pushed LR/patience with diminishing returns and a
spreading quality cost -- this is the next lever, not more of the same one).

The "own" class still gets normal cross-entropy supervision against the real
target. The two "alternate" classes are only used for the diversity penalty
(their own .loss field is NOT meaningful supervision -- it would be training
the model to produce an EASY target under a HARD tag -- so only their
logits are used, never their loss).

Usage:
  python question_generation/scripts/train_seq2seq_film_anticollapse.py \\
      --data-dir data/qg_v3/diff-control-irt-10k-blind \\
      --base-model question_generation/models/qg/flan-t5-base/baseline-race/final \\
      --output-dir question_generation/models/qg_10k_blind_film_v7 \\
      --epochs 20 --batch-size 16 --lr 1e-4 --diversity-weight 0.3 --bf16
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
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--film-lr-multiplier", type=float, default=100.0)
    parser.add_argument("--diversity-weight", type=float, default=0.3,
                         help="weight on the anti-collapse penalty (mean pairwise cosine "
                              "similarity between the 3 difficulty classes' output "
                              "distributions for the same teacher-forced sequence, added "
                              "to the normal cross-entropy loss)")
    args = parser.parse_args()

    print("Importing transformers...", flush=True)
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
    from transformers.modeling_outputs import BaseModelOutput
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
            {"params": film_params, "lr": args.lr * args.film_lr_multiplier, "weight_decay": 0.0},
        ],
    )
    print(f"difficulty_film LR: {args.lr * args.film_lr_multiplier} (base LR: {args.lr}), "
          f"diversity_weight: {args.diversity_weight}", flush=True)

    class AntiCollapseTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = inputs["labels"]
            own_ids = inputs["difficulty_ids"]

            # Encode once, reuse for all 3 difficulty-conditioned decodes.
            encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            own_out = model(
                encoder_outputs=encoder_outputs, attention_mask=attention_mask,
                labels=labels, difficulty_ids=own_ids, return_dict=True,
            )
            ce_loss = own_out.loss

            alt1_ids = (own_ids + 1) % 3
            alt2_ids = (own_ids + 2) % 3
            # Reuse the same teacher-forced target sequence under the two OTHER
            # difficulty tags -- only the logits are used (never .loss: training
            # toward this example's real target under the wrong tag would be
            # incorrect supervision), purely to measure/penalize how similar the
            # three tags' output distributions are for this passage.
            alt1_out = model(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state),
                attention_mask=attention_mask, labels=labels, difficulty_ids=alt1_ids, return_dict=True,
            )
            alt2_out = model(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state),
                attention_mask=attention_mask, labels=labels, difficulty_ids=alt2_ids, return_dict=True,
            )

            mask = (labels != -100).float()  # (B, T)
            p_own = F.softmax(own_out.logits, dim=-1)
            p_alt1 = F.softmax(alt1_out.logits, dim=-1)
            p_alt2 = F.softmax(alt2_out.logits, dim=-1)

            def masked_cos_sim(a, b):
                sim = F.cosine_similarity(a, b, dim=-1)  # (B, T)
                return (sim * mask).sum() / mask.sum().clamp(min=1)

            diversity_penalty = (
                masked_cos_sim(p_own, p_alt1) + masked_cos_sim(p_own, p_alt2) + masked_cos_sim(p_alt1, p_alt2)
            ) / 3.0

            loss = ce_loss + self.diversity_weight * diversity_penalty
            if return_outputs:
                return loss, own_out
            return loss

    trainer = AntiCollapseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DiffCondCollator(),
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    trainer.diversity_weight = args.diversity_weight

    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}", flush=True)

    print(f"Training FiLM model (anti-collapse) -> {out_dir}", flush=True)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved to {out_dir}/final", flush=True)


if __name__ == "__main__":
    main()
