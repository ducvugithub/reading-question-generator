#!/usr/bin/env python3
"""
Quick EDA for the seq2seq QG training dataset.

Writes data/training/eda_<date>.json and prints a compact table to stdout.

Usage:
  python scripts/eda_dataset.py
  python scripts/eda_dataset.py --data-dir data/training
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from datetime import date
from pathlib import Path


def _count_sentences(text: str) -> int:
    return len(re.split(r"(?<=[.!?])\s+", text.strip()))


def _percentile(vals: list[float], p: int) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "count":  len(vals),
        "min":    round(min(vals), 1),
        "median": round(statistics.median(vals), 1),
        "mean":   round(statistics.mean(vals), 1),
        "p75":    round(_percentile(vals, 75), 1),
        "p90":    round(_percentile(vals, 90), 1),
        "max":    round(max(vals), 1),
    }


def analyze_split(path: Path) -> dict:
    records = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not records:
        return {}

    passage_counter = collections.Counter(r["passage"] for r in records)
    qpp = list(passage_counter.values())

    kg_raw_n   = [len(r["kg_raw"])   if r.get("kg_raw")   else 0 for r in records]
    kg_coref_n = [len(r["kg_coref"]) if r.get("kg_coref") else 0 for r in records]

    return {
        "records":          len(records),
        "unique_passages":  len(passage_counter),
        "single_q_passages": sum(1 for v in qpp if v == 1),
        "q_per_passage":    _stats(qpp),
        "passage_chars":     _stats([len(r["passage"])                  for r in records]),
        "passage_words":     _stats([len(r["passage"].split())          for r in records]),
        "passage_sentences": _stats([_count_sentences(r["passage"])     for r in records]),
        "question_chars":   _stats([len(r["question"])         for r in records]),
        "answer_chars":     _stats([len(r["answer"])           for r in records]),
        "kg_raw_triples":   _stats(kg_raw_n),
        "kg_coref_triples": _stats(kg_coref_n) if any(kg_coref_n) else None,
        "cefr":             dict(sorted(collections.Counter(
                                r.get("cefr", "?") for r in records).items())),
    }


def print_table(results: dict[str, dict]) -> None:
    cols = ["lang/split", "records", "uniq_pass", "q/pass(med)", "pass_sents(med)",
            "pass_words(med)", "kg_raw(med)", "kg_raw(p90)", "kg_raw(max)", "coref"]
    widths = [max(len(c), 12) for c in cols]

    def row(cells):
        return "  ".join(str(c).rjust(w) for c, w in zip(cells, widths))

    print("\n" + row(cols))
    print("  ".join("-" * w for w in widths))

    for key in sorted(results):
        s = results[key]
        if not s:
            continue
        coref_pct = (
            f"{100 * sum(s['kg_coref_triples'].values()) / s['records']:.0f}%"
            if s.get("kg_coref_triples") else "—"
        )
        # coref% = fraction of records that have any coref triples
        has_coref = s["kg_coref_triples"] is not None
        coref_str = "yes" if has_coref else "—"
        print(row([
            key,
            f"{s['records']:,}",
            f"{s['unique_passages']:,}",
            s["q_per_passage"].get("median", "—"),
            s["passage_sentences"].get("median", "—"),
            s["passage_words"].get("median", "—"),
            s["kg_raw_triples"].get("median", "—"),
            s["kg_raw_triples"].get("p90", "—"),
            s["kg_raw_triples"].get("max", "—"),
            coref_str,
        ]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/training")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = data_dir / f"eda_{date.today()}.json"

    LANGS = {"en", "fi", "ru"}

    results: dict[str, dict] = {}
    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name not in LANGS:
            continue
        for split in ("train", "eval"):
            p = lang_dir / f"{split}.jsonl"
            if p.exists():
                print(f"  {lang_dir.name}/{split} ...", flush=True)
                results[f"{lang_dir.name}/{split}"] = analyze_split(p)

    if not results:
        print(f"No JSONL files found under {data_dir}")
        return

    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {output}")

    print_table(results)


if __name__ == "__main__":
    main()
