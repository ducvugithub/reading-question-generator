#!/usr/bin/env python3
"""
Train a contrastive Question Difficulty Estimator using triplet margin loss.

Phase 1 — Contrastive training:
  Triplets (anchor, positive, negative) are mined online per batch.
  The model learns an embedding space where same-difficulty questions cluster.

Phase 2 — Classifier:
  A logistic regression is fit on frozen embeddings extracted from the
  best contrastive checkpoint. This is the final EASY/MEDIUM/HARD classifier.

Usage:
  python question_difficulty/scripts/train_contrastive.py
  python question_difficulty/scripts/train_contrastive.py --model-name microsoft/deberta-v3-base
  python question_difficulty/scripts/train_contrastive.py --limit 200 --epochs 2  # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_LABEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
_LABEL_NAMES = ["EASY", "MEDIUM", "HARD"]


class TripletQDEDataset(Dataset):
    """
    Returns (anchor, positive, negative) triplets sampled online.

    mode='same_class'  — positive=same label, negative=any different label (standard)
    mode='ordinal'     — positive=adjacent level, negative=furthest level
                         enforces d(EASY,MEDIUM) < d(EASY,HARD); MEDIUM anchors fall
                         back to same-class (no clear "closer" neighbour)
    mode='mixed'       — 50/50 coin flip between same_class and ordinal each step
    """

    def __init__(self, records: list[dict], tokenizer, max_len: int, mode: str = "same_class"):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        # Index records by label for fast positive/negative sampling
        self.by_label: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for i, rec in enumerate(records):
            label = _LABEL_MAP[rec["difficulty"].lower()]
            self.by_label[label].append(i)

    # Ordinal neighbours: label → (adjacent_label, far_label)
    # MEDIUM (1) has no clear "closer" end — falls back to same-class
    _ORDINAL = {0: (1, 2), 2: (1, 0)}  # EASY→(MEDIUM,HARD), HARD→(MEDIUM,EASY)

    def _encode(self, rec: dict) -> dict:
        question_text = f"{rec['question']} [answer: {rec['answer']}]"
        enc = self.tokenizer(
            question_text,
            rec["passage"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(1, dtype=torch.long)).squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        anchor_rec = self.records[idx]
        anchor_label = _LABEL_MAP[anchor_rec["difficulty"].lower()]

        use_ordinal = (
            self.mode == "ordinal"
            or (self.mode == "mixed" and random.random() < 0.5)
        )

        if use_ordinal and anchor_label in self._ORDINAL:
            # Ordinal triplet: positive=adjacent level, negative=far level
            adj_label, far_label = self._ORDINAL[anchor_label]
            pos_idx = random.choice(self.by_label[adj_label])
            neg_idx = random.choice(self.by_label[far_label])
        else:
            # Same-class triplet: positive=same label, negative=any other label
            pos_candidates = [i for i in self.by_label[anchor_label] if i != idx]
            pos_idx = random.choice(pos_candidates) if pos_candidates else idx
            neg_labels = [l for l in self.by_label if l != anchor_label and self.by_label[l]]
            neg_idx = random.choice(self.by_label[random.choice(neg_labels)])

        pos_rec = self.records[pos_idx]
        neg_rec = self.records[neg_idx]

        return {
            "anchor":   self._encode(anchor_rec),
            "positive": self._encode(pos_rec),
            "negative": self._encode(neg_rec),
            "label":    torch.tensor(anchor_label, dtype=torch.long),
        }


class EmbeddingDataset(Dataset):
    """Plain dataset for extracting embeddings (no triplets)."""

    def __init__(self, records: list[dict], tokenizer, max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        question_text = f"{rec['question']} [answer: {rec['answer']}]"
        enc = self.tokenizer(
            question_text,
            rec["passage"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label = _LABEL_MAP[rec["difficulty"].lower()]
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(1, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def load_records(path: Path, limit: int | None) -> list[dict]:
    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if limit:
        random.Random(42).shuffle(records)
        records = records[:limit]
    return records


def collate_triplets(batch):
    def stack(key, sub):
        return torch.stack([b[key][sub] for b in batch])
    return {
        "anchor":   {k: stack("anchor",   k) for k in batch[0]["anchor"]},
        "positive": {k: stack("positive", k) for k in batch[0]["positive"]},
        "negative": {k: stack("negative", k) for k in batch[0]["negative"]},
        "label":    torch.stack([b["label"] for b in batch]),
    }


def extract_embeddings(model, records, tokenizer, max_len, batch_size, device):
    model.eval()
    ds = EmbeddingDataset(records, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size)
    all_embs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            embs  = model(ids, mask, ttype if ttype.any() else None)
            all_embs.append(embs.cpu())
            all_labels.extend(batch["label"].tolist())
    return torch.cat(all_embs).numpy(), all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",   default="roberta-base")
    parser.add_argument("--data-dir",     default="data/qde")
    parser.add_argument("--output-dir",   default="question_difficulty/models/contrastive")
    parser.add_argument("--triplet-mode", default="mixed",
                        choices=["same_class", "ordinal", "mixed"],
                        help="same_class: cluster by label; ordinal: enforce EASY<MEDIUM<HARD geometry; mixed: 50/50")
    parser.add_argument("--embed-dim",   type=int,   default=256)
    parser.add_argument("--margin",      type=float, default=0.5,
                        help="Triplet margin (cosine distance)")
    parser.add_argument("--max-len",     type=int,   default=512)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--limit",       type=int,   default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model_name}", flush=True)

    from transformers import AutoTokenizer
    from question_difficulty.methods.contrastive.model import ContrastiveQDE
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_dir = Path(args.data_dir)
    train_records = load_records(data_dir / "train.jsonl", args.limit)
    val_records   = load_records(data_dir / "val.jsonl",   args.limit)
    test_records  = load_records(data_dir / "test.jsonl",  args.limit)
    print(f"train={len(train_records)}  val={len(val_records)}  test={len(test_records)}", flush=True)

    print(f"Triplet mode: {args.triplet_mode}", flush=True)
    train_loader = DataLoader(
        TripletQDEDataset(train_records, tokenizer, args.max_len, mode=args.triplet_mode),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_triplets,
    )

    model = ContrastiveQDE(args.model_name, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a, b: 1 - F.cosine_similarity(a, b),
        margin=args.margin,
    )

    out_dir = Path(args.output_dir) / args.model_name.replace("/", "_") / args.triplet_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: contrastive training
    best_val_acc, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            def _fwd(part):
                b = batch[part]
                ids   = b["input_ids"].to(device)
                mask  = b["attention_mask"].to(device)
                ttype = b["token_type_ids"].to(device)
                return model(ids, mask, ttype if ttype.any() else None)

            anchor_emb   = _fwd("anchor")
            positive_emb = _fwd("positive")
            negative_emb = _fwd("negative")

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Quick val check via logistic regression on embeddings
        train_embs, train_labels = extract_embeddings(
            model, train_records, tokenizer, args.max_len, args.batch_size, device
        )
        val_embs, val_labels = extract_embeddings(
            model, val_records, tokenizer, args.max_len, args.batch_size, device
        )
        probe = LogisticRegression(max_iter=300, random_state=42)
        probe.fit(train_embs, train_labels)
        val_acc = probe.score(val_embs, val_labels)

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"triplet_loss={total_loss/n_batches:.4f}  val_acc(probe)={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}", flush=True)

    # Phase 2: fit final classifier on best checkpoint embeddings
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    train_embs, train_labels = extract_embeddings(
        model, train_records, tokenizer, args.max_len, args.batch_size, device
    )
    val_embs,  val_labels  = extract_embeddings(
        model, val_records,   tokenizer, args.max_len, args.batch_size, device
    )
    test_embs, test_labels = extract_embeddings(
        model, test_records,  tokenizer, args.max_len, args.batch_size, device
    )

    if np.isnan(train_embs).any():
        print("WARNING: embeddings contain NaN — embedding collapse detected. "
              "Try a lower learning rate. Skipping probe fit.", flush=True)
        return

    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(train_embs, train_labels)

    test_acc  = clf.score(test_embs, test_labels)
    test_preds = clf.predict(test_embs)
    print(f"Test accuracy: {test_acc:.4f}", flush=True)
    print(classification_report(test_labels, test_preds, target_names=_LABEL_NAMES))

    # Save
    import pickle
    with (out_dir / "probe_clf.pkl").open("wb") as f:
        pickle.dump(clf, f)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "config.json").write_text(
        json.dumps({
            "model_name": args.model_name,
            "embed_dim":  args.embed_dim,
            "max_len":    args.max_len,
        }, indent=2)
    )
    print(f"Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
