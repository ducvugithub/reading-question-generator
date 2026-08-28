#!/usr/bin/env python3
"""
Assess question quality using multiple open-source LLMs (Qwen, Llama 2, Mistral).

Reads generated questions and outputs quality assessments from each LLM.

Usage:
  python question_answering/scripts/assess_llm_quality.py \
    --input question_generation/results/qg_generated.jsonl \
    --output question_answering/results/llm_quality_assessments.jsonl \
    --models Qwen/Qwen2-7B-Instruct meta-llama/Llama-2-7b-chat-hf

Outputs to: question_answering/results/llm_quality_assessments.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

import torch

# Add this directory to path to import llm_assessor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_assessor import LLMAssessorPool


_DEFAULT_MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="question_generation/results/qg_generated.jsonl",
                        help="Input JSONL with generated questions")
    parser.add_argument("--output", default="question_answering/results/llm_quality_assessments.jsonl",
                        help="Output JSONL with quality assessments")
    parser.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                        help="HF model IDs to load")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for testing)")
    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load records
    records = [json.loads(l) for l in input_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records")

    # Detect device
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device >= 0 else 'CPU'}")

    # Load LLM assessors
    print(f"\nLoading {len(args.models)} LLM assessors...")
    try:
        pool = LLMAssessorPool(args.models, device=device)
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)

    # Run assessments
    print(f"\nAssessing {len(records)} questions...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="LLM assessment"):
            passage = rec.get("passage", "")
            question = rec.get("generated_question", "")

            if not passage or not question:
                continue

            # Get assessments from all models
            assessments = pool.assess_all(passage, question)

            # Get consensus
            consensus = pool.get_consensus(assessments)

            # Write output
            output_rec = {
                **rec,
                "llm_assessments": assessments,
                "llm_consensus": consensus,
            }
            fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

    print(f"\n✓ Quality assessments saved to {output_path}")
    print(f"\nNext: Combine with QA scores in Stage 3 evaluation")


if __name__ == "__main__":
    main()
