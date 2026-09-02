#!/usr/bin/env python3
"""
Validate whether the per-question difficulty signals (question_difficulty/docs/
cognitive_difficulty_estimation.md, "Method 4") actually capture real signal,
before building any classifier/embedding/adapter on top of them.

Two checks, neither needs ground truth labels:

1. Non-degeneracy: does each signal produce genuinely different numbers
   across questions, or is it flat/constant? (std/range of raw values)
2. THE CRITICAL TEST: does each signal vary WITHIN a single passage's
   multiple real questions, not just across different passages? A signal
   that only varies across passages is just re-deriving passage identity
   (the same confound we're trying to escape) -- only within-passage
   variance is real per-question signal.

For attention_dispersion specifically, this script also picks the best
layer: whichever layer's entropy shows the largest within-passage spread
(no ground truth needed for this either -- see docs for why).

Usage:
  python question_difficulty/scripts/validate_difficulty_signals.py --limit-passages 30
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from question_difficulty.methods.feature_based.difficulty_signals import (
    AnswerExtractivenessSignal,
    AttentionDispersionSignal,
    QuestionAnswerSimilaritySignal,
)

_MC_PATTERNS = ("which of the following", "which one of the following", "which of these")


def _is_mc_question(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _MC_PATTERNS)


def _iter_race_with_answers(subset: str, difficulty: str, limit_passages: int | None):
    from datasets import load_dataset

    ds = load_dataset("ehovy/race", subset, split="train")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    by_passage = defaultdict(list)
    for rec in ds:
        idx = letter_to_idx.get(rec["answer"])
        if idx is None or idx >= len(rec["options"]):
            continue
        if _is_mc_question(rec["question"]):
            continue
        by_passage[rec["article"]].append({
            "question": rec["question"],
            "answer": rec["options"][idx],
            "difficulty": difficulty,
        })
        if limit_passages and len(by_passage) >= limit_passages * 3:
            break  # over-fetch since many will have too few non-MC questions

    for passage, items in by_passage.items():
        if len(items) >= 2:  # need multiple questions for the within-passage test
            yield passage, items


def _iter_race_c_with_answers(limit_passages: int | None):
    """RACE-C (HARD) -- separate HF dataset, different field names than
    ehovy/race (option/label instead of options/answer)."""
    from datasets import load_dataset

    ds = load_dataset("tasksource/race-c", split="train")
    by_passage = defaultdict(list)
    for rec in ds:
        if rec["label"] is None or rec["label"] >= len(rec["option"]):
            continue
        if _is_mc_question(rec["question"]):
            continue
        by_passage[rec["article"]].append({
            "question": rec["question"],
            "answer": rec["option"][rec["label"]],
            "difficulty": "HARD",
        })
        if limit_passages and len(by_passage) >= limit_passages * 3:
            break

    for passage, items in by_passage.items():
        if len(items) >= 2:
            yield passage, items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-passages", type=int, default=30,
                        help="Number of passages (each with >=2 real questions) to validate on")
    parser.add_argument("--qa-model", default="deepset/roberta-base-squad2")
    parser.add_argument("--output", default="question_difficulty/results/signal_validation.jsonl")
    args = parser.parse_args()

    print("Loading RACE passages with multiple real (non-MC) questions, "
          "balanced across EASY/MEDIUM/HARD...", flush=True)
    per_subset_limit = -(-args.limit_passages // 3)  # ceil division, ~1/3 from each level
    passages = []
    counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}

    for passage, items in _iter_race_with_answers("middle", "EASY", per_subset_limit):
        passages.append((passage, items))
        counts["EASY"] += 1
        if counts["EASY"] >= per_subset_limit:
            break
    for passage, items in _iter_race_with_answers("high", "MEDIUM", per_subset_limit):
        passages.append((passage, items))
        counts["MEDIUM"] += 1
        if counts["MEDIUM"] >= per_subset_limit:
            break
    for passage, items in _iter_race_c_with_answers(per_subset_limit):
        passages.append((passage, items))
        counts["HARD"] += 1
        if counts["HARD"] >= per_subset_limit:
            break

    print(f"Loaded {len(passages)} passages ({counts}), "
          f"{sum(len(items) for _, items in passages)} total questions", flush=True)

    print("Loading signal extractors (this downloads models on first run)...", flush=True)
    print(f"  QA model for attention extraction: {args.qa_model}", flush=True)
    print("  (single model — token-level and sentence-level tables are two different", flush=True)
    print("   summaries of this same model's attention output, not from separate models)", flush=True)
    attn_signal = AttentionDispersionSignal(qa_model_name=args.qa_model)
    extractiveness_signal = AnswerExtractivenessSignal()
    qa_similarity_signal = QuestionAnswerSimilaritySignal()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    with output_path.open("w", encoding="utf-8") as fout:
        for p_idx, (passage, items) in enumerate(passages):
            for item in items:
                features = {}
                features.update(attn_signal.compute(passage, item["question"], item["answer"]))
                features.update(extractiveness_signal.compute(passage, item["question"], item["answer"]))
                features.update(qa_similarity_signal.compute(passage, item["question"], item["answer"]))

                record = {
                    "passage_id": p_idx,
                    "question": item["question"],
                    "answer": item["answer"],
                    "subset_difficulty": item["difficulty"],
                    **features,
                }
                all_records.append(record)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {p_idx + 1}/{len(passages)} passages done", flush=True)

    print(f"\nSaved raw signal values to {output_path}\n")
    analyze(all_records, args.qa_model)


_LAYER_RE = re.compile(r"_(layer\d+|alllayers)$")


def _within_passage_stats(records: list[dict], keys: list[str]) -> dict[str, dict]:
    """For each key: mean/std of per-passage spread, global range, n_passages.
    No ratio, no "best" — just the raw numbers, for you to decide with."""
    by_passage = defaultdict(list)
    for r in records:
        by_passage[r["passage_id"]].append(r)

    stats = {}
    for key in keys:
        within = []
        for items in by_passage.values():
            if len(items) < 2:
                continue
            values = [item[key] for item in items]
            within.append(max(values) - min(values))
        all_vals = [r[key] for r in records]
        stats[key] = {
            "mean_spread": statistics.mean(within) if within else 0.0,
            "std_spread": statistics.pstdev(within) if len(within) > 1 else 0.0,
            "global_range": (max(all_vals) - min(all_vals)) if all_vals else 0.0,
            "n_passages": len(within),
        }
    return stats


def _layer_order(layer_id: str) -> tuple[int, int]:
    if layer_id == "alllayers":
        return (1, 0)
    return (0, int(layer_id.replace("layer", "")))


def _print_pivot_table(title: str, records: list[dict], keys: list[str],
                       metric_names: list[str], col_extractor, csv_rows: list[dict],
                       table_name: str, md_sections: list[str]) -> None:
    """col_extractor(key) -> (layer_id, metric_name) or None to skip."""
    stats = _within_passage_stats(records, keys)
    by_layer: dict[str, dict[str, dict]] = defaultdict(dict)
    for key in keys:
        parsed = col_extractor(key)
        if parsed is None:
            continue
        layer_id, metric = parsed
        by_layer[layer_id][metric] = stats[key]
        csv_rows.append({"table": table_name, "metric": metric, "layer": layer_id, **stats[key]})

    display_names = [f"{m}_spread" for m in metric_names]

    col_width = max(16, max(len(m) for m in display_names) + 2)
    print(f"\n{title} — within-passage spread (mean ± std across passages)")
    header = f"  {'layer':10s}" + "".join(f"{m:>{col_width}s}" for m in display_names)
    print(header)

    md_lines = [f"## {title}", "", "within-passage spread (mean ± std across passages)", "",
                "| layer | " + " | ".join(display_names) + " |",
                "|---" * (len(metric_names) + 1) + "|"]

    for layer_id in sorted(by_layer, key=_layer_order):
        row = by_layer[layer_id]
        cells = "".join(
            f"{row.get(m, {}).get('mean_spread', float('nan')):.3f}"
            f"±{row.get(m, {}).get('std_spread', float('nan')):.3f}".rjust(col_width)
            for m in metric_names
        )
        print(f"  {layer_id:10s}{cells}")

        md_cells = " | ".join(
            f"{row.get(m, {}).get('mean_spread', float('nan')):.3f}"
            f"±{row.get(m, {}).get('std_spread', float('nan')):.3f}"
            for m in metric_names
        )
        md_lines.append(f"| {layer_id} | {md_cells} |")

    md_sections.append("\n".join(md_lines))


def analyze(records: list[dict], qa_model_name: str) -> None:
    all_keys = [k for k in records[0].keys()
                if k not in ("passage_id", "question", "answer", "subset_difficulty")
                and isinstance(records[0][k], (int, float))]

    print("=" * 80)
    print("CHECK 1: Non-degeneracy (global spread, no labels needed)")
    print("=" * 80)
    for key in all_keys:
        values = [r[key] for r in records]
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        flag = "  [DEGENERATE]" if std < 1e-4 else ""
        print(f"  {key:30s} std={std:.4f}  range=[{min(values):.3f}, {max(values):.3f}]{flag}")

    print()
    print("=" * 80)
    print("CHECK 2: Within-passage variance (THE CRITICAL TEST)")
    print("=" * 80)

    token_keys = [k for k in all_keys if k.startswith("tok_")]
    sent_keys = [k for k in all_keys if k.startswith("sent_")]
    other_keys = [k for k in all_keys if k not in token_keys and k not in sent_keys]

    def _token_extractor(key: str):
        m = _LAYER_RE.search(key)
        if not m:
            return None
        layer_id = m.group(1)
        metric = key[len("tok_"):-len(m.group(0))]
        return layer_id, metric

    def _sent_extractor(key: str):
        m = _LAYER_RE.search(key)
        if not m:
            return None
        layer_id = m.group(1)
        metric = key[len("sent_"):-len(m.group(0))]
        return layer_id, metric

    csv_rows: list[dict] = []
    md_sections: list[str] = []

    _print_pivot_table(
        "TOKEN-LEVEL", records, token_keys,
        ["entropy", "max", "min", "top5", "top10", "top15"], _token_extractor,
        csv_rows, "token", md_sections,
    )
    _print_pivot_table(
        "SENTENCE-LEVEL", records, sent_keys,
        ["total_entropy", "avg_entropy", "total_max", "avg_max", "total_min", "avg_min"],
        _sent_extractor, csv_rows, "sentence", md_sections,
    )

    print("\nOTHER SIGNALS — within-passage spread (mean ± std across passages)")
    other_stats = _within_passage_stats(records, other_keys)
    other_md = ["## OTHER SIGNALS", "", "| signal | mean±std | global_range | n_passages |", "|---|---|---|---|"]
    for key, s in sorted(other_stats.items(), key=lambda x: -x[1]["mean_spread"]):
        print(f"  {key:35s} {s['mean_spread']:.3f}±{s['std_spread']:.3f}"
              f"  (global_range={s['global_range']:.3f}, n={s['n_passages']})")
        csv_rows.append({"table": "other", "metric": key, "layer": "-", **s})
        other_md.append(f"| {key} | {s['mean_spread']:.3f}±{s['std_spread']:.3f} | "
                        f"{s['global_range']:.3f} | {s['n_passages']} |")
    md_sections.append("\n".join(other_md))

    csv_path = Path("question_difficulty/results/signal_validation_stats.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("table,metric,layer,mean_spread,std_spread,global_range,n_passages\n")
        for row in csv_rows:
            f.write(f"{row['table']},{row['metric']},{row['layer']},"
                    f"{row['mean_spread']:.4f},{row['std_spread']:.4f},"
                    f"{row['global_range']:.4f},{row['n_passages']}\n")

    md_path = Path("question_difficulty/results/signal_validation_stats.md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Difficulty Signal Validation — Within-Passage Spread\n\n")
        f.write(f"QA model for attention extraction: `{qa_model_name}`\n\n")
        f.write("\n\n".join(md_sections))
        f.write("\n")

    print(f"\nFull stats written to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
