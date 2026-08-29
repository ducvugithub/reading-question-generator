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

_DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


_DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]


def _format_input(model_type: str, passage: str, difficulty: str = "") -> str:
    """Format input text based on model type.

    - baseline-race: {passage} (difficulty ignored — control group, output
      should not change across forced tokens since the model never sees them)
    - diff-control-race: <EASY|MEDIUM|HARD> {passage}
    - adapter-control-race: {passage} (no text token — difficulty is routed
      via the active adapter instead, see generate_questions())
    """
    if model_type == "baseline-race":
        return passage
    elif model_type == "diff-control-race":
        return f"<{difficulty}> {passage}"
    elif model_type == "adapter-control-race":
        return passage
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
    do_sample: bool = False,
    num_samples: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[str]:
    """Generate questions for a passage at target difficulty.

    Beam search (do_sample=False) always returns the single most probable
    sequence — if that sequence is the same regardless of the difficulty
    token, beam search can't tell you whether the token shifted the
    underlying distribution at all, only whether it shifted the single top
    answer. Sampling (do_sample=True, num_samples>1) draws from the model's
    actual distribution so a shift in probability mass shows up as a shift in
    the *spread* of outputs, even when the single most-likely answer doesn't
    change.
    """
    input_text = _format_input(model_type, passage, difficulty)
    if model_type == "adapter-control-race":
        model.set_active_adapters(difficulty.lower())

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        if do_sample:
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                num_return_sequences=num_samples,
                temperature=temperature,
                top_p=top_p,
            )
        else:
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
                        choices=["baseline-race", "diff-control-race", "adapter-control-race"],
                        help="Model types to use for generation (difficulty-conditioned models only — "
                             "focus-control-hotpot uses a different conditioning mechanism, not "
                             "difficulty tokens, so it doesn't fit this script's forced-token loop)")
    parser.add_argument("--base-model", default="flan-t5-base",
                        help="Base model slug to load checkpoints from, e.g. t5-base, "
                             "flan-t5-base, flan-t5-large (matches the folder under --model-dir)")
    parser.add_argument("--num-per-difficulty", type=int, default=1,
                        help="Number of questions to generate per difficulty level")
    parser.add_argument("--model-dir", default="question_generation/models/qg",
                        help="Directory containing trained models")
    parser.add_argument("--data-dir", default="data/qg",
                        help="Directory containing prepared QG data")
    parser.add_argument("--output", default="question_generation/results/qg_generated.jsonl",
                        help="Output JSONL file with generated questions")
    parser.add_argument("--num-beams", type=int, default=4,
                        help="Number of beams for beam search (ignored if --do-sample)")
    parser.add_argument("--do-sample", action="store_true",
                        help="Use sampling instead of beam search — reveals whether the "
                             "difficulty token shifts the model's output *distribution*, not "
                             "just its single most-likely answer (see generate_questions() docstring)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of sampled sequences per (passage, difficulty) when --do-sample")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more variety), only used with --do-sample")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling top-p, only used with --do-sample")
    parser.add_argument("--max-length", type=int, default=64,
                        help="Maximum length of generated questions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for testing)")
    args = parser.parse_args()

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test set (baseline-race and diff-control-race share identical
    # passages/questions — either works as the passage source)
    test_path = Path(args.data_dir) / "baseline-race" / "test.jsonl"
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
        model_path = Path(args.model_dir) / args.base_model / model_type / "final"
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}, skipping", file=sys.stderr)
            continue

        print(f"  Loading {model_type} from {model_path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if model_type == "adapter-control-race":
            import adapters

            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            adapters.init(model)
            for level in _DIFFICULTY_LEVELS:
                model.load_adapter(str(model_path / level))
            model = model.to(device)
        else:
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
            # baseline-race/test.jsonl has no "passage" key — its input_text
            # IS the raw passage unmodified (baseline adds no token/prefix)
            passage = rec.get("input_text", "")
            original_question = rec.get("target_text", "")
            true_difficulty = rec.get("difficulty", "")

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
                            do_sample=args.do_sample,
                            num_samples=args.num_samples,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )

                        # Take top num_per_difficulty questions (or all sampled ones)
                        n_keep = args.num_samples if args.do_sample else args.num_per_difficulty
                        for i, question in enumerate(questions[:n_keep]):
                            output_rec = {
                                "passage": passage,
                                "original_question": original_question,
                                "true_difficulty": true_difficulty,
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
