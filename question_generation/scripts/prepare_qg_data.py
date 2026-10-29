#!/usr/bin/env python3
"""
Step 2: Format raw train/val/test splits (from prepare_qg_test_sets.py) into
T5 QG input/target pairs for each model type.

Model types (4 primary + 1 reference):
  baseline-race:            <passage> → question
                            Source: data/qg/raw/race — RACE-only baseline.

  diff-control-race:        <EASY|MEDIUM|HARD> <passage> → question
                            Source: data/qg/raw/race — difficulty-controlled QG.

  baseline-hotpot:          <passage> → question
                            Source: data/qg/raw/hotpotqa — multi-hop baseline.

  focus-control-hotpot:     <passage_with_focus_spans> → question
                            Source: data/qg/raw/hotpotqa — focus-span-controlled QG.

  baseline-all:             <passage> → question
                            Source: data/qg/raw/race + data/qg/raw/hotpotqa combined.

  diff-control-irt-pilot:   <EASY|MEDIUM|HARD> <passage> → question
                            Source: data/qg/raw/race_irt_pilot (built by
                            prepare_pilot_irt_difficulty_data.py, NOT this
                            step's usual prepare_qg_test_sets.py) -- difficulty
                            is real per-question cascade-IRT signal (passage-
                            mean-centered, tertile-bucketed), not the passage-
                            inherited RACE-subset label diff-control-race uses.
                            Only ~1.2K items (vs. diff-control-race's 87K) and
                            has no val split (too few passages to split 3 ways)
                            -- a pilot to test whether a REAL per-question
                            signal steers generation where the confounded one
                            didn't, before investing in scaling labels to the
                            full 87K set.

  diff-control-irt-10k:     <EASY|MEDIUM|HARD> <passage> → question
                            Source: data/qg/raw/race_irt_10k (built by
                            prepare_irt_10k_difficulty_data.py) -- same
                            construction as diff-control-irt-pilot (real
                            cascade-IRT difficulty, passage-mean-centered,
                            tertile-bucketed), but ~9.9K items sampled
                            directly from the real 87K-question RACE training
                            pool (Option 3: skip the QDE estimator, which
                            plateaued at ~991 examples regardless of
                            architecture, and run the cascade itself at
                            scale instead).

Reads pre-split train/val/test from
data/qg/raw/{race,hotpotqa,race_irt_pilot,race_irt_10k}/{split}.jsonl — does
no splitting itself, so all model types trained on the same source share
identical test passages.

Usage:
  python question_generation/scripts/prepare_qg_data.py --steps baseline-race diff-control-race
  python question_generation/scripts/prepare_qg_data.py --steps baseline-hotpot focus-control-hotpot
  python question_generation/scripts/prepare_qg_data.py --steps baseline-all
  python question_generation/scripts/prepare_qg_data.py --steps diff-control-irt-pilot
  python question_generation/scripts/prepare_qg_data.py --steps diff-control-irt-10k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_raw(raw_dir: Path, dataset: str, split: str) -> list[dict]:
    path = raw_dir / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run prepare_qg_test_sets.py first"
        )
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _wrap_focus_spans(passage: str, focus: str) -> str:
    return passage.replace(focus, f"<FOCUS_SPAN>{focus}</FOCUS_SPAN>", 1) if focus else passage


def _format_input(model_type: str, rec: dict) -> str:
    passage = rec["passage"]

    if model_type in ("baseline-all", "baseline-race", "baseline-hotpot"):
        return passage

    if model_type in ("diff-control-race", "diff-control-irt-pilot", "diff-control-irt-10k"):
        return f"<{rec['difficulty']}> {passage}"

    if model_type == "focus-control-hotpot":
        return _wrap_focus_spans(passage, rec.get("focus", ""))

    raise ValueError(f"Unknown model_type: {model_type}")


def _sources_for_step(model_type: str, raw_dir: Path, split: str) -> list[dict]:
    if model_type in ("baseline-race", "diff-control-race"):
        return _load_raw(raw_dir, "race", split)
    if model_type in ("baseline-hotpot", "focus-control-hotpot"):
        return _load_raw(raw_dir, "hotpotqa", split)
    if model_type == "baseline-all":
        return _load_raw(raw_dir, "race", split) + _load_raw(raw_dir, "hotpotqa", split)
    if model_type == "diff-control-irt-pilot":
        return _load_raw(raw_dir, "race_irt_pilot", split)
    if model_type == "diff-control-irt-10k":
        return _load_raw(raw_dir, "race_irt_10k", split)
    raise ValueError(f"Unknown model_type: {model_type}")


def prepare_step(model_type: str, raw_dir: Path, out_dir: Path) -> None:
    step_dir = out_dir / model_type
    step_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        records = _sources_for_step(model_type, raw_dir, split)
        out_records = []
        for rec in records:
            out = {
                "input_text": _format_input(model_type, rec),
                "target_text": rec["question"],
                "source": rec.get("source", ""),
                "model_type": model_type,
            }
            if rec.get("difficulty"):
                out["difficulty"] = rec["difficulty"]
            out_records.append(out)

        path = step_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sizes = {s: sum(1 for _ in (step_dir / f"{s}.jsonl").open()) for s in ("train", "val", "test")}
    print(f"  [{model_type}] train={sizes['train']} val={sizes['val']} test={sizes['test']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=["baseline-all", "baseline-race", "baseline-hotpot",
                 "diff-control-race", "focus-control-hotpot", "diff-control-irt-pilot",
                 "diff-control-irt-10k"],
        default=["baseline-all", "baseline-race", "diff-control-race",
                 "baseline-hotpot", "focus-control-hotpot"],
    )
    parser.add_argument("--raw-dir", default="data/qg/raw")
    parser.add_argument("--output-dir", default="data/qg")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    for step in args.steps:
        print(f"\n=== {step.upper()} ===", flush=True)
        prepare_step(step, raw_dir, out_dir)


if __name__ == "__main__":
    main()
