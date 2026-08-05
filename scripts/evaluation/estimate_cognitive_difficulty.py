#!/usr/bin/env python3
"""
Estimate cognitive difficulty on records from a JSONL dataset file.

Usage:
  # Run graph-based on first 10 records
  python scripts/estimate_cognitive_difficulty.py --input data/training/en/train.jsonl --mode graph --limit 10

  # Run LLM-based (requires ANTHROPIC_API_KEY)
  python scripts/estimate_cognitive_difficulty.py --input data/training/en/train.jsonl --mode llm --limit 5

  # Annotate a full file and write output
  python scripts/estimate_cognitive_difficulty.py --input data/training/en/train.jsonl --mode graph --output data/training/en/train_cog.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from question_generation.difficulty import (
    GraphCognitiveDifficultyEstimator,
    LLMCognitiveDifficultyEstimator,
)


def _build_estimator(mode: str):
    if mode == "graph":
        return GraphCognitiveDifficultyEstimator()
    if mode == "llm":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set")
        return LLMCognitiveDifficultyEstimator(client=anthropic.Anthropic(api_key=api_key))
    sys.exit(f"Unknown mode: {mode!r}")


_DEMO_PASSAGE = (
    "Marie Curie was born in Warsaw, Poland, in 1867. She moved to Paris to study at the Sorbonne, "
    "where she became the first woman to earn a doctorate in physics in France. Together with her husband "
    "Pierre Curie, she discovered the radioactive elements polonium and radium. She won the Nobel Prize in "
    "Physics in 1903, sharing it with Pierre and Henri Becquerel, and later won the Nobel Prize in Chemistry "
    "in 1911, making her the first person to win Nobel Prizes in two different sciences. Her research on "
    "radioactivity laid the groundwork for modern nuclear physics and cancer radiotherapy. She died in 1934 "
    "from aplastic anaemia, caused by prolonged exposure to radiation during her research."
)

# Questions at varying cognitive levels for the same passage
_DEMO_QA_PAIRS = [
    ("Where was Marie Curie born?", "Warsaw"),
    ("When did Marie Curie win her first Nobel Prize?", "1903"),
    ("Who did Marie Curie share the Nobel Prize in Physics with?", "Pierre Curie and Henri Becquerel"),
    ("Why did Marie Curie die?", "prolonged exposure to radiation"),
    ("How did Marie Curie's work influence modern medicine?", "cancer radiotherapy"),
]


def _build_demo_records() -> list[dict]:
    print("Extracting KG from demo passage (this may take ~30s)...", flush=True)
    import torch
    # stanza 1.14.0 model files contain pickle classes blocked by weights_only=True
    # in torch >= 2.0; patch for local dev only (safe — stanza models are trusted)
    _orig_load = torch.load
    torch.load = lambda f, *a, **kw: _orig_load(f, *a, **{**kw, "weights_only": False})
    from knowledge_graph.extractor import KnowledgeGraphExtractor

    extractor = KnowledgeGraphExtractor(lang="en", coref=True)
    kg_raw_triples, kg_coref_triples = extractor.extract_both(_DEMO_PASSAGE)
    kg_raw = [[t.subject, t.relation, t.object] for t in kg_raw_triples]
    kg_coref = [[t.subject, t.relation, t.object] for t in kg_coref_triples]
    print(f"Extracted {len(kg_raw)} raw triples, {len(kg_coref)} coref triples.\n", flush=True)

    return [
        {"passage": _DEMO_PASSAGE, "question": q, "answer": a, "kg_raw": kg_raw, "kg_coref": kg_coref}
        for q, a in _DEMO_QA_PAIRS
    ]


def _load_records(path: Path | None, limit: int | None, demo: bool) -> list[dict]:
    if demo:
        return _build_demo_records()
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate cognitive difficulty on a QG JSONL dataset.")
    parser.add_argument("--input", type=Path, default=None, metavar="FILE")
    parser.add_argument("--demo", action="store_true", help="Run on built-in sample records (no --input needed)")
    parser.add_argument("--mode", choices=["graph", "llm"], default="graph")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE",
                        help="Write annotated records to this file (default: print to stdout)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N records")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full record details alongside the estimate")
    args = parser.parse_args()

    if not args.demo and not args.input:
        sys.exit("Provide --input FILE or use --demo to run on sample records.")

    estimator = _build_estimator(args.mode)
    records = _load_records(args.input, args.limit, args.demo)

    out_file = open(args.output, "w", encoding="utf-8") if args.output else None
    label_counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}

    try:
        for i, rec in enumerate(records):
            result = estimator.estimate(
                question=rec["question"],
                answer=rec["answer"],
                kg_raw=rec.get("kg_raw") or [],
                kg_coref=rec.get("kg_coref"),
            )

            label_counts[result["label"]] += 1
            rec["cognitive_difficulty"] = result

            if out_file:
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                if args.verbose:
                    print(f"\n{'─' * 60}")
                    if rec.get("passage"):
                        print(f"PASSAGE: {rec['passage']}")
                    print(f"Q : {rec['question']}")
                    print(f"A : {rec['answer']}")
                    print(f"KG raw  : {rec.get('kg_raw', [])}")
                    if rec.get("kg_coref") and rec.get("kg_coref") != rec.get("kg_raw"):
                        print(f"KG coref: {rec['kg_coref']}")
                    if "reasoning" in result:
                        print(f"Reasoning: {result['reasoning']}")
                    print(f"→ score={result['score']:.3f}  label={result['label']}")
                else:
                    reasoning = f"  {result['reasoning']}" if "reasoning" in result else ""
                    print(f"[{i+1:>5}] {result['label']:6s} ({result['score']:.3f})  {rec['question'][:80]}{reasoning}")

    finally:
        if out_file:
            out_file.close()

    print(f"\nSummary ({args.mode}): easy={label_counts['easy']} medium={label_counts['medium']} hard={label_counts['hard']}")
    if args.output:
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
