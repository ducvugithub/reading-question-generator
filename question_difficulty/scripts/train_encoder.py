#!/usr/bin/env python3
"""
Fine-tune an encoder (RoBERTa, DeBERTa, BERT, …) for Question Difficulty Estimation.

Input format:  [CLS] question [SEP] passage [SEP]
Labels:        EASY=0  MEDIUM=1  HARD=2

Usage:
  # RoBERTa (default)
  python question_difficulty/scripts/train_encoder.py

  # DeBERTa — often stronger for classification
  python question_difficulty/scripts/train_encoder.py --model-name microsoft/deberta-v3-base

  # Quick smoke test
  python question_difficulty/scripts/train_encoder.py --limit 200 --epochs 2

  # Regression on the real continuous difficulty_score instead of the EASY/MEDIUM/HARD
  # bucket (requires a difficulty_score field -- see prepare_qde_irt_pilot_data.py)
  python question_difficulty/scripts/train_encoder.py --data-dir data/qde_irt_pilot --regression
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_LABEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
_LABEL_NAMES = ["EASY", "MEDIUM", "HARD"]


class QDEDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_len: int, regression: bool = False):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.regression = regression

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        # Format: "question [answer: ...]" paired with passage
        question_text = f"{rec['question']} [answer: {rec['answer']}]"
        enc = self.tokenizer(
            question_text,
            rec["passage"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        if self.regression:
            label = torch.tensor(rec["difficulty_score"], dtype=torch.float32)
        else:
            label = torch.tensor(_LABEL_MAP[rec["difficulty"].lower()], dtype=torch.long)
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(1, dtype=torch.long)).squeeze(0),
            "label":          label,
        }


def load_records(path: Path, limit: int | None) -> list[dict]:
    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if limit:
        random.Random(42).shuffle(records)
        records = records[:limit]
    return records


def evaluate(model, loader, device) -> tuple[float, list[int], list[int]]:
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask, ttype if ttype.any() else None)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1, all_preds, all_labels


def evaluate_regression(model, loader, device) -> tuple[float, list[float], list[float]]:
    """Returns (MAE, predictions, true scores) -- MAE in the same units as
    difficulty_score (a Rasch/Bayesian-IRT logit-scale difficulty), not
    percent; main() converts to %-of-range for interpretability."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            preds = model(ids, mask, ttype if ttype.any() else None).squeeze(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    mae = sum(abs(p - l) for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return mae, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",  default="roberta-base",
                        help="HuggingFace encoder (roberta-base, microsoft/deberta-v3-base, bert-base-uncased, ...)")
    parser.add_argument("--data-dir",    default="data/qde")
    parser.add_argument("--output-dir",  default="question_difficulty/models/encoder")
    parser.add_argument("--max-len",     type=int,   default=512)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze the encoder (train only the classifier head, a linear probe) -- "
                             "intended for small datasets where full fine-tuning overfits fast")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0,
                        help="With --freeze-encoder, also re-enable gradients for the top N transformer "
                             "blocks -- a middle ground between a pure linear probe and full fine-tuning")
    parser.add_argument("--regression", action="store_true",
                        help="Regress the continuous difficulty_score directly (MSE loss, 1 output) instead "
                             "of classifying the EASY/MEDIUM/HARD bucket -- requires a difficulty_score field "
                             "in the data (see prepare_qde_irt_pilot_data.py); a 3-class confusion matrix over "
                             "an arbitrarily tertile-bucketed continuous signal is hard to interpret, and this "
                             "uses the full signal instead of discarding it at the bucket boundary")
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Max records per split (for smoke tests)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model_name}", flush=True)

    from transformers import AutoTokenizer
    from question_difficulty.methods.encoder.model import EncoderQDE

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_dir = Path(args.data_dir)
    train_records = load_records(data_dir / "train.jsonl", args.limit)
    val_records   = load_records(data_dir / "val.jsonl",   args.limit)
    test_records  = load_records(data_dir / "test.jsonl",  args.limit)
    print(f"train={len(train_records)}  val={len(val_records)}  test={len(test_records)}", flush=True)

    train_loader = DataLoader(QDEDataset(train_records, tokenizer, args.max_len, args.regression),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(QDEDataset(val_records,   tokenizer, args.max_len, args.regression),
                              batch_size=args.batch_size)
    test_loader  = DataLoader(QDEDataset(test_records,  tokenizer, args.max_len, args.regression),
                              batch_size=args.batch_size)

    model = EncoderQDE(args.model_name, n_classes=(1 if args.regression else 3),
                        freeze_encoder=args.freeze_encoder,
                        unfreeze_last_n_layers=args.unfreeze_last_n_layers).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({n_trainable/n_total:.1%})", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=0.01)
    criterion = torch.nn.MSELoss() if args.regression else torch.nn.CrossEntropyLoss()

    # For regression, "best" = lowest val MAE; for classification, highest val macro F1.
    best_val_metric = float("inf") if args.regression else 0.0
    best_epoch = 0
    out_dir = Path(args.output_dir) / args.model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            ttype  = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask, ttype if ttype.any() else None)
            loss = criterion(logits.squeeze(-1), labels) if args.regression else criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        if args.regression:
            val_metric, _, _ = evaluate_regression(model, val_loader, device)
            print(f"Epoch {epoch}/{args.epochs}  train_mse={train_loss:.4f}  val_mae={val_metric:.4f}", flush=True)
            is_best = val_metric < best_val_metric
        else:
            val_metric, _, _ = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_macro_f1={val_metric:.4f}", flush=True)
            is_best = val_metric > best_val_metric

        if is_best:
            best_val_metric = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(f"\nBest val_metric={best_val_metric:.4f} at epoch {best_epoch}", flush=True)

    # Evaluate best model on test set
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    if args.regression:
        all_scores = [r["difficulty_score"] for r in train_records + val_records + test_records]
        score_range = max(all_scores) - min(all_scores)
        test_mae, preds, labels = evaluate_regression(model, test_loader, device)
        print(f"Test MAE: {test_mae:.4f}  (score range={score_range:.4f}, "
              f"MAE as %-of-range={test_mae/score_range:.1%})", flush=True)
        # A naive "always predict the mean" baseline -- MAE should beat this by a real margin,
        # otherwise the model isn't adding anything over just guessing the average difficulty.
        mean_pred = sum(r["difficulty_score"] for r in train_records) / len(train_records)
        baseline_mae = sum(abs(mean_pred - l) for l in labels) / len(labels)
        print(f"Baseline MAE (always predict train mean={mean_pred:.4f}): {baseline_mae:.4f}", flush=True)
    else:
        test_acc, preds, labels = evaluate(model, test_loader, device)
        print(f"Test accuracy: {test_acc:.4f}", flush=True)
        from sklearn.metrics import classification_report
        print(classification_report(labels, preds, target_names=_LABEL_NAMES))

    # Save tokenizer config alongside model
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "config.json").write_text(
        json.dumps({"model_name": args.model_name, "max_len": args.max_len}, indent=2)
    )
    print(f"Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
