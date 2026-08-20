#!/usr/bin/env python3
"""
Evaluate all trained QDE methods on the held-out test set and print a comparison table.

Methods evaluated (skipped if model files are missing):
  feature_based           — GBT on linguistic features
  encoder/roberta-base    — Fine-tuned RoBERTa
  encoder/deberta-v3-base — Fine-tuned DeBERTa
  contrastive/*/mixed     — Contrastive encoder + LR probe

LLM-verdict is run separately via evaluate_llm_verdict.py (requires ANTHROPIC_API_KEY).

Usage:
  python question_difficulty/scripts/evaluate_qde.py
  python question_difficulty/scripts/evaluate_qde.py --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_LABEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
_LABEL_NAMES = ["EASY", "MEDIUM", "HARD"]


def load_test(data_dir: Path) -> tuple[list[dict], list[int]]:
    records = [json.loads(l) for l in (data_dir / "test.jsonl").open() if l.strip()]
    labels = [_LABEL_MAP[r["difficulty"].lower()] for r in records]
    return records, labels


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return {
        "macro_f1":  f1_score(y_true, y_pred, average="macro"),
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_easy":   float(per_class[0]),
        "f1_medium": float(per_class[1]),
        "f1_hard":   float(per_class[2]),
    }


# ── Feature-based ──────────────────────────────────────────────────────────────

def predict_feature_based(records: list[dict], model_path: Path) -> list[int]:
    from question_difficulty.methods.feature_based.features import FEATURE_NAMES, extract
    with model_path.open("rb") as f:
        clf = pickle.load(f)
    X = [[extract(r["passage"], r["question"], r["answer"])[k] for k in FEATURE_NAMES]
         for r in records]
    return clf.predict(np.array(X, dtype=np.float32)).tolist()


# ── Shared tokenisation loader ─────────────────────────────────────────────────

def _make_loader(records: list[dict], tokenizer, max_len: int, batch_size: int):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class _DS(Dataset):
        def __getitem__(self, i: int) -> dict:
            r = records[i]
            q = f"{r['question']} [answer: {r['answer']}]"
            enc = tokenizer(q, r["passage"], max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc.get("token_type_ids",
                                          torch.zeros(1, dtype=torch.long)).squeeze(0),
            }
        def __len__(self) -> int: return len(records)

    return DataLoader(_DS(), batch_size=batch_size)


# ── Encoder ────────────────────────────────────────────────────────────────────

def predict_encoder(records: list[dict], model_dir: Path, batch_size: int) -> list[int]:
    import torch
    from transformers import AutoTokenizer
    from question_difficulty.methods.encoder.model import EncoderQDE

    cfg = json.loads((model_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = EncoderQDE(cfg["model_name"]).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in _make_loader(records, tokenizer, cfg["max_len"], batch_size):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            logits = model(ids, mask, ttype if ttype.any() else None)
            preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


# ── Contrastive ────────────────────────────────────────────────────────────────

def predict_contrastive(records: list[dict], model_dir: Path, batch_size: int) -> list[int]:
    import torch
    from transformers import AutoTokenizer
    from question_difficulty.methods.contrastive.model import ContrastiveQDE

    cfg = json.loads((model_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ContrastiveQDE(cfg["model_name"], embed_dim=cfg["embed_dim"]).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    with open(model_dir / "probe_clf.pkl", "rb") as f:
        probe = pickle.load(f)

    embs = []
    with torch.no_grad():
        for batch in _make_loader(records, tokenizer, cfg["max_len"], batch_size):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            emb = model(ids, mask, ttype if ttype.any() else None)
            embs.append(emb.cpu().numpy())
    return probe.predict(np.vstack(embs)).tolist()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/qde")
    parser.add_argument("--model-dir",  default="question_difficulty/models")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    print("Loading test set...", flush=True)
    records, y_true = load_test(data_dir)
    print(f"  {len(records)} examples", flush=True)

    results: dict[str, dict] = {}

    # Feature-based GBT
    gbt_path = model_dir / "feature_based" / "gbt_model.pkl"
    if gbt_path.exists():
        print("\nEvaluating: feature_based GBT", flush=True)
        results["feature_based"] = metrics(y_true, predict_feature_based(records, gbt_path))

    # Encoder
    for backbone in ["roberta-base", "microsoft_deberta-v3-base"]:
        d = model_dir / "encoder" / backbone
        if (d / "best_model.pt").exists():
            name = f"encoder/{backbone}"
            print(f"\nEvaluating: {name}", flush=True)
            results[name] = metrics(y_true, predict_encoder(records, d, args.batch_size))

    # Contrastive
    for backbone in ["roberta-base", "microsoft_deberta-v3-base"]:
        for mode in ["same_class", "ordinal", "mixed"]:
            d = model_dir / "contrastive" / backbone / mode
            if (d / "best_model.pt").exists():
                name = f"contrastive/{backbone}/{mode}"
                print(f"\nEvaluating: {name}", flush=True)
                results[name] = metrics(y_true, predict_contrastive(records, d, args.batch_size))

    # Comparison table
    print("\n" + "=" * 82)
    print(f"{'Method':<47} {'MacroF1':>8} {'Acc':>7} {'F1-E':>6} {'F1-M':>6} {'F1-H':>6}")
    print("-" * 82)
    for name, m in sorted(results.items(), key=lambda x: -x[1]["macro_f1"]):
        print(f"{name:<47} {m['macro_f1']:>8.4f} {m['accuracy']:>7.4f}"
              f" {m['f1_easy']:>6.4f} {m['f1_medium']:>6.4f} {m['f1_hard']:>6.4f}")
    print("=" * 82)
    print("LLM-verdict: run evaluate_llm_verdict.py separately (needs ANTHROPIC_API_KEY)")


if __name__ == "__main__":
    main()
