#!/usr/bin/env python3
"""
Compute sent_total_entropy_layer11 (from AttentionDispersionSignal) for every
real RACE question, at full training-set scale -- not just the ~100-passage
validation sample. See question_generation/docs/difficulty_steering_mechanisms.md
for why this signal/layer was chosen, and the plan this feeds into
(diff-entropy-token-control-race).

Reuses AttentionDispersionSignal.compute() as-is (it computes all layers in
one forward pass regardless), then extracts just the one field we need.

Input: data/qg/raw/race/{train,val,test}.jsonl (from prepare_qg_test_sets.py)
Output: same records, with an added "entropy_layer11" field.

Usage:
  python question_generation/scripts/compute_difficulty_signal.py --split train
  python question_generation/scripts/compute_difficulty_signal.py --split train --limit 2000  # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from question_difficulty.methods.feature_based.difficulty_signals import AttentionDispersionSignal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/qg/raw/race")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--qa-model", default="deepset/roberta-base-squad2")
    parser.add_argument("--limit", type=int, default=None, help="For smoke testing")
    parser.add_argument("--output", default=None,
                        help="Default: <raw-dir>/<split>_entropy_layer11.jsonl")
    args = parser.parse_args()

    in_path = Path(args.raw_dir) / f"{args.split}.jsonl"
    out_path = Path(args.output) if args.output else Path(args.raw_dir) / f"{args.split}_entropy_layer11.jsonl"

    records = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records from {in_path}", flush=True)

    print(f"Loading QA model for attention extraction: {args.qa_model}", flush=True)
    signal = AttentionDispersionSignal(qa_model_name=args.qa_model)

    # Resume support -- skip records already written
    done = 0
    if out_path.exists():
        done = sum(1 for _ in out_path.open())
        print(f"Resuming -- {done} records already done", flush=True)

    with out_path.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            if i < done:
                continue
            try:
                features = signal.compute(rec["passage"], rec["question"], answer="")
                entropy = features.get("sent_total_entropy_layer11")
            except Exception as e:
                print(f"  [!] Failed on record {i}: {e}", file=sys.stderr)
                entropy = None

            out_rec = {**rec, "entropy_layer11": entropy}
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(records)} done", flush=True)

    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
