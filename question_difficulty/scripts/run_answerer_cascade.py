#!/usr/bin/env python3
"""
Cascade experiment: run a lineup of Answerers (weak -> strong, local ->
remote) over sampled OneStopQA/RACE items, and correlate each one's
accuracy against the IRT (Rasch) item difficulty from
question_difficulty_with_irt.ipynb (real human response data).

Candidates, roughly weakest -> strongest (see question_answering/answerer.py):
  extractive          LocalExtractiveAnswerer(deepset/roberta-base-squad2) -- BERT-family span extraction
  qwen05_<precision>   LocalDecoderAnswerer(Qwen2.5-0.5B-Instruct)         -- small local decoder
  falcon1b_<precision> LocalDecoderAnswerer(tiiuae/Falcon3-1B-Instruct)    -- small local decoder, different family
  qwen15_<precision>   LocalDecoderAnswerer(Qwen2.5-1.5B-Instruct)         -- bigger local decoder
  haiku                ClaudeBedrockAnswerer(Haiku, via Bedrock)          -- remote, real cost per item
  opus                 ClaudeBedrockAnswerer(Opus, via Bedrock) -- --include-opus -- remote, real cost per item, opt-in only

Each local decoder model is loaded once per entry in --precisions
(default "fp16" -- cheapest/fastest, for validating the script itself
before a heavier run), so e.g. --precisions fp16,fp32 doubles the number
of decoder candidates (qwen05_fp16, qwen05_fp32, ...) to let you compare
precision vs. accuracy directly. "int8"/"int4" need bitsandbytes + a CUDA
GPU (see question_answering/qa_model.py's DecoderOnlyQAModel) -- they
raise a clear error on this project's Apple Silicon dev machines; run
those via question_difficulty/slurms/run_answerer_cascade.job on CSC
Roihu instead. Only float32 is confirmed NaN-free end-to-end (see
DecoderOnlyQAModel's docstring for the fp16+eager NaN bug this avoids) --
fp16 is confirmed safe for the generation path this script uses
(predict_multiple_choice_answer, which never touches eager attention).

Mistral-7B / Falcon-7B are NOT included here -- too large for this
machine's 18GB RAM; run those on Roihu with their own Answerer instances
if needed later.

Bedrock tiers use profile "fsecure-golden-retriever-ci" / region
"eu-north-1" by default (--aws-profile/--aws-region to override) -- every
call to them is REAL, BILLED Bedrock usage. Check the sample size before a
large run.

Usage:
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10 --precisions fp16,fp32
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10 --include-opus
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ONESTOPQA_REPO = REPO_ROOT.parent / "onestop-qa"
RASCH_DIFFICULTY_CACHE = REPO_ROOT / "question_difficulty/notebooks/irt_rasch_difficulty_cache.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"))


def load_items() -> dict:
    from data_processing import HumanResponseBank

    assert ONESTOPQA_REPO.exists(), f"expected onestop-qa clone at {ONESTOPQA_REPO}"
    assert RASCH_DIFFICULTY_CACHE.exists(), (
        f"{RASCH_DIFFICULTY_CACHE} not found -- run question_difficulty_with_irt.ipynb's "
        "IRT section first, it writes this cache"
    )
    bank = HumanResponseBank(ONESTOPQA_REPO)
    item_difficulty = json.loads(RASCH_DIFFICULTY_CACHE.read_text())

    items = {}
    for item_id, rows in bank.rows_by_item().items():
        if item_id not in item_difficulty:
            continue
        r = rows[0]
        items[item_id] = {
            "passage": r["paragraph"],
            "question": r["question"],
            "options": r["options"].split("|"),
            "correct_answer_text": r["correct_answer_text"],
            "difficulty": item_difficulty[item_id],
        }
    return items


DECODER_MODELS = [
    ("qwen05", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("falcon1b", "tiiuae/Falcon3-1B-Instruct"),
    ("qwen15", "Qwen/Qwen2.5-1.5B-Instruct"),
]


def build_candidates(precisions: list[str], include_opus: bool, skip_remote: bool,
                      aws_profile: str, aws_region: str) -> list[tuple[str, object]]:
    """Returns [(name, Answerer), ...] -- loaded lazily, one at a time, so
    peak memory stays at whichever single local model is currently loaded
    plus whatever's already been loaded before it (all fit comfortably in
    18GB, but this keeps things simple and matches how the notebooks do it).
    Each entry in DECODER_MODELS is loaded once per precision in
    `precisions` (see module docstring).

    skip_remote: drop the Bedrock tiers entirely -- SLURM compute nodes
    (e.g. Roihu) typically have no internet egress, so a Bedrock call
    would just hang/fail there; the point of running there is the local
    GPU precision sweep (int8/int4), not the remote tiers anyway."""
    from question_answering.qa_model import DecoderOnlyQAModel, ExtractiveQAModel
    from question_answering.answerer import ClaudeBedrockAnswerer, LocalDecoderAnswerer, LocalExtractiveAnswerer

    candidates = []

    print("loading extractive (deepset/roberta-base-squad2)...")
    candidates.append(("extractive", LocalExtractiveAnswerer(ExtractiveQAModel("deepset/roberta-base-squad2"))))

    for name, model_id in DECODER_MODELS:
        for precision in precisions:
            print(f"loading {name}_{precision} ({model_id}, precision={precision})...")
            candidates.append((f"{name}_{precision}",
                                LocalDecoderAnswerer(DecoderOnlyQAModel(model_id, precision=precision))))

    if skip_remote:
        return candidates

    print("loading haiku (Bedrock)...")
    candidates.append(("haiku", ClaudeBedrockAnswerer(
        model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        aws_region=aws_region, aws_profile=aws_profile,
    )))

    if include_opus:
        print("loading opus (Bedrock)...")
        candidates.append(("opus", ClaudeBedrockAnswerer(
            model="eu.anthropic.claude-opus-4-8",
            aws_region=aws_region, aws_profile=aws_profile,
        )))

    return candidates


def main() -> None:
    from data_processing import _texts_match

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precisions", default="fp16",
                         help="Comma-separated precisions to run each local decoder model at: "
                              "fp32,fp16,int8,int4 (int8/int4 need bitsandbytes + CUDA, e.g. Roihu)")
    parser.add_argument("--include-opus", action="store_true", help="Add the Opus Bedrock tier (real, larger cost per item)")
    parser.add_argument("--skip-remote", action="store_true",
                         help="Drop the Bedrock tiers entirely (use on a SLURM compute node with no internet egress)")
    parser.add_argument("--aws-profile", default="fsecure-golden-retriever-ci")
    parser.add_argument("--aws-region", default="eu-north-1")
    parser.add_argument("--output", default=str(REPO_ROOT / "question_difficulty/scripts/cascade_results.json"))
    args = parser.parse_args()

    items = load_items()
    print(f"{len(items)} items available with IRT difficulty\n")

    random.seed(args.seed)
    sample_ids = random.sample(list(items.keys()), min(args.n_items, len(items)))

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    candidates = build_candidates(precisions, args.include_opus, args.skip_remote, args.aws_profile, args.aws_region)
    print(f"\ncascade: {[name for name, _ in candidates]}\n")

    results = {}
    output_path = Path(args.output)
    for i, item_id in enumerate(sample_ids):
        it = items[item_id]
        row = {"difficulty": it["difficulty"], "answers": {}}
        for name, answerer in candidates:
            letter, confidence = answerer.answer(it["passage"], it["question"], it["options"])
            predicted_text = it["options"]["ABCD".index(letter)] if letter else None
            is_correct = _texts_match(predicted_text, it["correct_answer_text"]) if predicted_text else False
            row["answers"][name] = {"letter": letter, "confidence": confidence, "is_correct": is_correct}
        results[item_id] = row

        print(f"[{i+1}/{len(sample_ids)}] {item_id}  (difficulty={it['difficulty']:+.3f})  "
              + "  ".join(f"{name}={'Y' if row['answers'][name]['is_correct'] else 'N'}" for name, _ in candidates))

        output_path.write_text(json.dumps(results, indent=2))

    print(f"\nsaved to {output_path}")
    print("\nAccuracy per candidate:")
    for name, _ in candidates:
        n_correct = sum(r["answers"][name]["is_correct"] for r in results.values())
        print(f"  {name:12s} {n_correct}/{len(results)} ({n_correct/len(results):.1%})")


if __name__ == "__main__":
    main()
