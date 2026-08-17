#!/usr/bin/env python3
"""
Train a feature-based Question Difficulty Estimator (gradient-boosted trees).

Reads data/qde/{train,val,test}.jsonl produced by prepare_qde_data.py,
extracts linguistic features, and trains a GradientBoostingClassifier.

Usage:
  python question_difficulty/scripts/train_feature_based.py
  python question_difficulty/scripts/train_feature_based.py --data-dir data/qde --limit 5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import random

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from question_difficulty.methods.feature_based.features import FEATURE_NAMES, extract

_LABEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
_LABEL_NAMES = ["EASY", "MEDIUM", "HARD"]


def load_split(path: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if limit:
        random.Random(42).shuffle(records)
        records = records[:limit]

    X, y = [], []
    for rec in records:
        label = _LABEL_MAP.get(rec["difficulty"].lower())
        if label is None:
            continue
        feats = extract(rec["passage"], rec["question"], rec["answer"])
        X.append([feats[k] for k in FEATURE_NAMES])
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/qde")
    parser.add_argument("--output-dir", default="question_difficulty/models/feature_based")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max records per split (for smoke tests)")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth",    type=int, default=5)
    parser.add_argument("--lr",           type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading features...", flush=True)
    X_train, y_train = load_split(data_dir / "train.jsonl", args.limit)
    X_val,   y_val   = load_split(data_dir / "val.jsonl",   args.limit)
    X_test,  y_test  = load_split(data_dir / "test.jsonl",  args.limit)
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}", flush=True)

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_sample_weight

    # Class weights to compensate for EASY >> MEDIUM imbalance
    sample_weights = compute_sample_weight("balanced", y_train)

    print(f"Training GBT (n_estimators={args.n_estimators}, max_depth={args.max_depth}, class_weight=balanced)...", flush=True)
    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    def _report(split_name, y_true, y_pred):
        print(f"\n{split_name}:")
        print(classification_report(y_true, y_pred, target_names=_LABEL_NAMES))
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix (rows=true, cols=predicted):")
        print(f"  {'':8}  {'EASY':>6}  {'MEDIUM':>6}  {'HARD':>6}")
        for i, name in enumerate(_LABEL_NAMES):
            print(f"  {name:8}  {cm[i][0]:6}  {cm[i][1]:6}  {cm[i][2]:6}")

    _report("Val",  y_val,  clf.predict(X_val))
    _report("Test", y_test, clf.predict(X_test))

    # Feature importance
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:<16} {imp:.4f}")

    # Save model
    import pickle
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "gbt_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved to {model_path}", flush=True)


if __name__ == "__main__":
    main()
