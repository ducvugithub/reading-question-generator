#!/usr/bin/env python3
"""
Run multiple QA models on generated questions to validate answerability.

Loads all 4 pre-trained QA models (RoBERTa, BERT, DistilRoBERTa, DeBERTa),
runs inference on (passage, question) pairs, and outputs confidence scores.

Usage:
  python question_answering/scripts/run_qa_models.py

Input JSONL format:
  {"passage": "...", "question": "...", "answer": "..."}

Output JSONL format:
  {
    "passage": "...", "generated_question": "...", "target_difficulty": "EASY",

    # Full results per model
    "qa_model_results": {
      "deepset/roberta-base-squad2": {
        "score": 0.95,
        "answer": "Paris",
        "start": 10,
        "end": 15
      },
      ...
    },

    # Convenience fields for analysis
    "qa_scores": {"model1": 0.95, "model2": 0.92, ...},
    "qa_answer_spans": {"model1": "Paris", "model2": "Paris", ...},
    "qa_agreement": true,                    # All models agree on answer?
    "qa_consensus_answer": "Paris",          # If agreement, what is it?
    "qa_avg_score": 0.93,                    # Average confidence (metric TBD)
    "qa_num_models": 4                       # How many models succeeded
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

import torch


_QA_MODELS = [
    "deepset/roberta-base-squad2",
    "google-bert/bert-base-uncased-finetuned-squad",
    "mrm8488/distilroberta-base-finetuned-squad",
    "microsoft/deberta-v3-base",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="question_generation/results/qg_generated.jsonl",
                        help="Input JSONL file with (passage, question, answer)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file with QA scores")
    parser.add_argument("--models", nargs="+", default=_QA_MODELS,
                        help="QA models to use")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load records
    records = [json.loads(l) for l in input_path.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} records from {input_path}")

    # Detect device
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device >= 0 else 'CPU'}")

    # Load all QA pipelines
    print(f"\nLoading {len(args.models)} QA models...")
    try:
        from transformers import pipeline
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers torch", file=sys.stderr)
        sys.exit(1)

    qa_pipelines = {}
    for model_name in args.models:
        print(f"  Loading {model_name}...", flush=True)
        qa_pipelines[model_name] = pipeline(
            "question-answering",
            model=model_name,
            device=device,
        )

    # Run inference
    print(f"\nRunning QA inference on {len(records)} records...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="QA inference"):
            passage = rec.get("passage", "")
            question = rec.get("question", "")

            if not passage or not question:
                print(f"  [!] Skipping record with missing passage or question", file=sys.stderr)
                continue

            # Run all QA models
            qa_results = {}
            for model_name, qa_pipe in qa_pipelines.items():
                try:
                    result = qa_pipe(question=question, context=passage)
                    qa_results[model_name] = {
                        "score": float(result["score"]),
                        "answer": result.get("answer", ""),
                        "start": result.get("start"),
                        "end": result.get("end"),
                    }
                except Exception as e:
                    print(f"  [!] Error on {model_name}: {e}", file=sys.stderr)
                    qa_results[model_name] = {
                        "score": None,
                        "answer": None,
                        "start": None,
                        "end": None,
                        "error": str(e),
                    }

            # Extract scores and answer spans
            qa_scores = {k: v["score"] for k, v in qa_results.items()}
            answer_spans = {k: v["answer"] for k, v in qa_results.items() if v["answer"]}

            # Check agreement (multiple models extract same answer)
            unique_answers = set(answer_spans.values()) if answer_spans else set()
            qa_agreement = len(unique_answers) == 1 if unique_answers else False
            consensus_answer = list(unique_answers)[0] if qa_agreement else None

            # Compute average score (skip None values)
            valid_scores = [s for s in qa_scores.values() if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

            # Write output — record everything, decide metrics later
            output_rec = {
                **rec,
                "qa_model_results": qa_results,       # Full details per model
                "qa_scores": qa_scores,               # Just confidence scores
                "qa_answer_spans": answer_spans,      # Extracted answers by model
                "qa_agreement": qa_agreement,         # Do models agree?
                "qa_consensus_answer": consensus_answer,  # Agreed answer (if any)
                "qa_avg_score": avg_score,            # Average confidence (metric TBD)
                "qa_num_models": len(valid_scores),   # How many succeeded
            }
            fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

    print(f"\n✓ QA scores saved to {output_path}")


if __name__ == "__main__":
    main()
