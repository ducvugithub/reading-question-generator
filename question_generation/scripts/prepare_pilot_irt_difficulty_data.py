#!/usr/bin/env python3
"""
Builds the "diff-control-irt-pilot" raw split: a small (~1.2K item) QG
training set labeled with REAL per-question difficulty (from this project's
answerer-cascade + IRT fit), instead of diff-control-race's passage-inherited
RACE-subset label (RACE-middle->EASY, RACE-high->MEDIUM, RACE-C->HARD, which
`training_details.md` documents as confounded with passage style since every
passage only ever has ONE label).

Source data: the 1,227 OneStopQA/RACE items with real human responses AND a
model-only Bayesian IRT difficulty (see question_difficulty/notebooks/
irt_human_vs_model.ipynb) -- the same set used throughout this project's IRT
validation work, reused here as QG training data for the first time.

build_records() is also imported directly by
question_difficulty/scripts/prepare_qde_irt_pilot_data.py (a parallel
experiment: training a Question Difficulty Estimator on the SAME labels,
to eventually predict difficulty for the ~87K RACE questions this cascade
was never run on) -- kept as one shared function so both pilots train on
literally identical ground truth, not two independently-computed versions.

Label construction (see irt_human_vs_model.ipynb's discussion for why):
1. Drop the 6/432 passages with only 1 question -- centering a single-item
   group always yields exactly 0 regardless of its real difficulty, which
   would silently mislabel it as "medium" (see irt_human_vs_model.ipynb).
2. For each remaining passage, compute the mean cascade-IRT difficulty of
   its own questions, and subtract it from each question's own score
   ("centering"). This removes the passage-level component of difficulty
   (confirmed empirically: 36.9% of raw variance is between-passage, and the
   RACE/OneStopQA source tag explains only 7.7% of THAT -- so centering on
   the passage's own mean, not the tag, is what actually removes it) so the
   label reflects "harder than this passage's OTHER questions", not "which
   passage is this".
3. Bucket the centered scores into EASY/MEDIUM/HARD via global tertiles (so
   all three buckets are roughly balanced, unlike raw RACE-subset sizes).
4. Split 80/10/10 by passage (never by question -- a passage's questions must
   never straddle splits, same rule prepare_qg_test_sets.py uses for the
   original 87K set). val.jsonl must be non-empty: train_seq2seq.py uses it
   for eval-based early stopping and best-checkpoint selection regardless of
   model type -- an empty val split isn't just thin, it breaks training.

Usage:
  python question_generation/scripts/prepare_pilot_irt_difficulty_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "question_difficulty/scripts"))

from cascade_common import load_items  # noqa: E402


def _split_bucket(passage: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    h = int(hashlib.md5(passage.encode()).hexdigest(), 16) % 100
    if h < train_ratio * 100:
        return "train"
    if h < (train_ratio + val_ratio) * 100:
        return "val"
    return "test"


def build_records() -> list[dict]:
    items = load_items()
    model_diff = json.loads(
        (REPO_ROOT / "question_difficulty/scripts/irt_bayesian_model_only_difficulty.json").read_text()
    )
    item_meta = json.loads(
        (REPO_ROOT / "question_difficulty/notebooks/irt_item_meta_cache.json").read_text()
    )

    by_passage: dict[str, list[str]] = defaultdict(list)
    for item_id, it in items.items():
        if item_id in model_diff:
            by_passage[it["passage"]].append(item_id)

    n_single = sum(1 for ids in by_passage.values() if len(ids) == 1)
    multi_passage_items = {p: ids for p, ids in by_passage.items() if len(ids) >= 2}
    print(f"{len(by_passage)} passages total, dropping {n_single} single-question passages, "
          f"{len(multi_passage_items)} remain")

    centered: dict[str, float] = {}
    for passage, ids in multi_passage_items.items():
        passage_mean = sum(model_diff[i] for i in ids) / len(ids)
        for item_id in ids:
            centered[item_id] = model_diff[item_id] - passage_mean

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
            source = "_".join(item_meta.get(item_id, ["unknown", "unknown"])).lower()
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
    parser.add_argument("--output-dir", default="data/qg/raw/race_irt_pilot")
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

    from collections import Counter
    print("\ndifficulty distribution (train):", Counter(r["difficulty"] for r in by_split["train"]))
    print("difficulty distribution (val):  ", Counter(r["difficulty"] for r in by_split["val"]))
    print("difficulty distribution (test): ", Counter(r["difficulty"] for r in by_split["test"]))


if __name__ == "__main__":
    main()
