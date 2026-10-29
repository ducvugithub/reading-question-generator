#!/usr/bin/env python3
"""
Builds the "diff-control-irt-10k" raw split: same construction as
diff-control-irt-pilot (prepare_pilot_irt_difficulty_data.py) -- passage-
mean-centered, tertile-bucketed cascade-IRT difficulty -- but at ~10K items
sampled from the real ~87,866-question RACE training pool instead of the
1,227 OneStopQA/RACE items with human responses (Option 3: skip the QDE
estimator, which plateaued at a weak ceiling regardless of architecture at
~991 examples -- see question_difficulty/slurms/train_qde_irt_pilot.job's
history comments -- and run the real answerer cascade directly at a larger
scale instead, since no human data is needed for a model-only IRT fit).

Source data: question_difficulty/scripts/cascade_results_race_pool_10k_by_passage_scored.json
(11-model local-only cascade, fp16, run on 10,003 items sampled as WHOLE
PASSAGES -- see run_answerer_cascade_race_pool.job's comments on why
passage-level sampling, not the first run's question-level sampling, is
required for centering to work: question-level sampling left 84.6% of
sampled passages with only 1 of their ~3.65 real questions, which makes
per-passage centering degenerate for most of the data). Rasch-fit (not
Bayesian -- a training label only needs the point estimate, and NUTS at
this item count would be impractically slow) via fit_rasch_race_pool.py ->
irt_rasch_race_pool_10k_by_passage_difficulty.json.

Label construction identical to the pilot's build_records() (see that
script's docstring for the centering rationale and the single-question-
passage degenerate case): drop single-question passages, center each
question's difficulty on its own passage's mean, tertile-bucket globally,
split 80/10/10 by passage (never by question).

Usage:
  python question_generation/scripts/prepare_irt_10k_difficulty_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "question_difficulty/scripts"))

from cascade_common import load_race_training_pool  # noqa: E402

DIFFICULTY_FILE = REPO_ROOT / "question_difficulty/scripts/irt_rasch_race_pool_10k_by_passage_difficulty.json"


def _split_bucket(passage: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    h = int(hashlib.md5(passage.encode()).hexdigest(), 16) % 100
    if h < train_ratio * 100:
        return "train"
    if h < (train_ratio + val_ratio) * 100:
        return "val"
    return "test"


def build_records() -> list[dict]:
    items = load_race_training_pool()
    item_difficulty = json.loads(DIFFICULTY_FILE.read_text())

    by_passage: dict[str, list[str]] = defaultdict(list)
    for item_id, it in items.items():
        if item_id in item_difficulty:
            by_passage[it["passage"]].append(item_id)

    n_single = sum(1 for ids in by_passage.values() if len(ids) == 1)
    multi_passage_items = {p: ids for p, ids in by_passage.items() if len(ids) >= 2}
    print(f"{len(by_passage)} passages total, dropping {n_single} single-question passages, "
          f"{len(multi_passage_items)} remain")

    centered: dict[str, float] = {}
    for passage, ids in multi_passage_items.items():
        passage_mean = sum(item_difficulty[i] for i in ids) / len(ids)
        for item_id in ids:
            centered[item_id] = item_difficulty[item_id] - passage_mean

    sorted_scores = sorted(centered.values())
    n = len(sorted_scores)
    low_cut = sorted_scores[n // 3]
    high_cut = sorted_scores[(2 * n) // 3]
    print(f"{n} items after dropping singles, tertile cuts: low={low_cut:+.3f} high={high_cut:+.3f}")

    def bucket(score: float) -> str:
        if score <= low_cut:
            return "EASY"
        if score <= high_cut:
            return "MEDIUM"
        return "HARD"

    records = []
    for passage, ids in multi_passage_items.items():
        split = _split_bucket(passage)
        for item_id in ids:
            it = items[item_id]
            source = "race_middle" if item_id.startswith("middle_") else "race_high"
            records.append({
                "passage": it["passage"],
                "question": it["question"],
                "answer": it["correct_answer_text"],
                "difficulty": bucket(centered[item_id]),
                "source": source,
                "_split": split,
                "_item_id": item_id,
                "_centered_score": centered[item_id],
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/qg/raw/race_irt_10k")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records()
    by_split = defaultdict(list)
    for r in records:
        by_split[r.pop("_split")].append(r)

    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        recs = by_split.get(split, [])
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(recs)} records to {path}")

    print("\ndifficulty distribution (train):", Counter(r["difficulty"] for r in by_split["train"]))
    print("difficulty distribution (val):  ", Counter(r["difficulty"] for r in by_split["val"]))
    print("difficulty distribution (test): ", Counter(r["difficulty"] for r in by_split["test"]))


if __name__ == "__main__":
    main()
