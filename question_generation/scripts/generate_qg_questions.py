#!/usr/bin/env python3
"""
Generate questions using trained QG models with different difficulty levels.

Loads trained T5 models (baseline, diff-control) and generates questions
for test passages at each difficulty level (EASY, MEDIUM, HARD).

Usage:
  python question_generation/scripts/generate_qg_questions.py \
    --models baseline diff-control \
    --num-per-difficulty 1

Input: data/qg/{step}/test.jsonl
Output: question_generation/results/qg_generated.jsonl
  {"passage": "...", "original_question": "...", "model": "baseline",
   "target_difficulty": "EASY", "generated_question": "...", ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


_DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]


def _format_input(model_type: str, passage: str, difficulty: str = "") -> str:
    """Format input text based on model type.

    Format: <DIFFICULTY> {passage}
    - baseline: {passage}
    - diff-control: <EASY|MEDIUM|HARD> {passage}
    """
    if model_type == "baseline":
        return passage
    elif model_type == "diff-control":
        return f"<{difficulty}> {passage}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_questions(
    model,
    tokenizer,
    passage: str,
    difficulty: str,
    model_type: str,
    num_beams: int = 4,
    max_length: int = 64,
    device: str = "cpu",
) -> list[str]:
    """Generate num_beams unique questions for a passage at target difficulty."""
    input_text = _format_input(model_type, passage, difficulty)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            temperature=1.0,
            do_sample=False,
        )

    questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=["baseline-race", "diff-control-race"],
                        choices=["baseline-all", "baseline-race", "baseline-hotpot",
                                "diff-control-race", "focus-control-hotpot"],
                        help="Model types to use for generation")
    parser.add_argument("--num-per-difficulty", type=int, default=1,
                        help="Number of questions to generate per difficulty level")
    parser.add_argument("--model-dir", default="question_generation/models",
                        help="Directory containing trained models")
    parser.add_argument("--data-dir", default="data/qg",
                        help="Directory containing prepared QG data")
    parser.add_argument("--output", default="question_generation/results/qg_generated.jsonl",
                        help="Output JSONL file with generated questions")
    parser.add_argument("--num-beams", type=int, default=4,
                        help="Number of beams for beam search")
    parser.add_argument("--max-length", type=int, default=64,
                        help="Maximum length of generated questions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for testing)")
    args = parser.parse_args()

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test set (use baseline as reference for passages)
    test_path = Path(args.data_dir) / "baseline" / "test.jsonl"
    if not test_path.exists():
        print(f"Error: {test_path} not found — run prepare_qg_data.py first", file=sys.stderr)
        sys.exit(1)

    test_records = [json.loads(l) for l in test_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        test_records = test_records[:args.limit]

    print(f"Loaded {len(test_records)} test examples")

    # Load models
    print(f"\nLoading models: {args.models}")
    models_dict = {}
    for model_type in args.models:
        model_path = Path(args.model_dir) / model_type
        if not (model_path / "pytorch_model.bin").exists():
            print(f"Warning: Model not found at {model_path}, skipping", file=sys.stderr)
            continue

        print(f"  Loading {model_type}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)
        model.eval()
        models_dict[model_type] = (model, tokenizer)

    if not models_dict:
        print("Error: No models loaded", file=sys.stderr)
        sys.exit(1)

    # Generate questions
    print(f"\nGenerating questions ({args.num_per_difficulty} per difficulty level)...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in tqdm(test_records, desc="Generating"):
            passage = rec.get("passage", "")
            original_question = rec.get("target_text", "")

            if not passage:
                continue

            # Generate with each model at each difficulty
            for model_type, (model, tokenizer) in models_dict.items():
                for difficulty in _DIFFICULTIES:
                    try:
                        questions = generate_questions(
                            model,
                            tokenizer,
                            passage,
                            difficulty,
                            model_type,
                            num_beams=args.num_beams,
                            max_length=args.max_length,
                            device=device,
                        )

                        # Take top num_per_difficulty questions
                        for i, question in enumerate(questions[:args.num_per_difficulty]):
                            output_rec = {
                                "passage": passage,
                                "original_question": original_question,
                                "model": model_type,
                                "target_difficulty": difficulty,
                                "generated_question": question,
                                "beam_rank": i,
                            }
                            fout.write(json.dumps(output_rec, ensure_ascii=False) + "\n")

                    except Exception as e:
                        print(f"  [!] Error generating {model_type}/{difficulty}: {e}", file=sys.stderr)
                        continue

    print(f"\n✓ Generated questions saved to {output_path}")
    print(f"\nNext: python question_answering/scripts/run_qa_models.py \\")
    print(f"        --input {output_path} \\")
    print(f"        --output question_answering/results/qa_scores.jsonl")


if __name__ == "__main__":
    main()
