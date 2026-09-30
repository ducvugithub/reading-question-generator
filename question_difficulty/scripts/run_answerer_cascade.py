#!/usr/bin/env python3
"""
Cascade experiment: run a lineup of Answerers (weak -> strong, local ->
remote) over sampled OneStopQA/RACE items. Records each candidate's raw
(letter, confidence) per item ONLY -- no correctness scoring here, see
score_cascade_results.py for that (cheap, local, pure string matching --
doesn't need this script's models loaded at all, so it's kept separate).

Candidates, roughly weakest -> strongest (see question_answering/answerer.py):
  extractive_<model>       LocalExtractiveAnswerer(one of EXTRACTIVE_MODELS)        -- BERT-family span extraction, no precision axis
  smollm2_135m_<precision> LocalDecoderAnswerer(SmolLM2-135M-Instruct)              -- tiny local decoder
  smollm2_360m_<precision> LocalDecoderAnswerer(SmolLM2-360M-Instruct)              -- tiny local decoder
  qwen2_05_<precision>     LocalDecoderAnswerer(Qwen2-0.5B-Instruct)                -- small local decoder, older/weaker generation than qwen05
  qwen05_<precision>       LocalDecoderAnswerer(Qwen2.5-0.5B-Instruct)              -- small local decoder
  tinyllama11_<precision>  LocalDecoderAnswerer(TinyLlama-1.1B-Chat-v1.0)           -- small local decoder, older/less-optimized recipe
  falcon1b_<precision>     LocalDecoderAnswerer(tiiuae/Falcon3-1B-Instruct)         -- small local decoder, different family
  qwen15_<precision>       LocalDecoderAnswerer(Qwen2.5-1.5B-Instruct)              -- bigger local decoder
  haiku                    ClaudeBedrockAnswerer(Haiku, via Bedrock)                -- remote, real cost per item
  opus                     ClaudeBedrockAnswerer(Opus, via Bedrock) -- --include-opus -- remote, real cost per item, opt-in only
  nova_micro/nova_lite/    BedrockConverseAnswerer(Nova, via Bedrock)               -- remote, real cost per item
    nova_pro
  glm47flash               BedrockConverseAnswerer(GLM-4.7-flash, via Bedrock)      -- remote, real cost per item

Each local decoder model runs once per entry in --precisions (default
"fp16" -- cheapest/fastest, for validating the script itself before a
heavier run), so e.g. --precisions fp16,fp32 doubles the number of
decoder candidates (qwen05_fp16, qwen05_fp32, ...) to let you compare
precision vs. accuracy directly. "int8"/"int4" need bitsandbytes + a CUDA
GPU (see question_answering/qa_model.py's DecoderOnlyQAModel) -- they
raise a clear error on this project's Apple Silicon dev machines; run
those via question_difficulty/slurms/run_answerer_cascade.job on CSC
Roihu instead. Only float32 is confirmed NaN-free end-to-end (see
DecoderOnlyQAModel's docstring for the fp16+eager NaN bug this avoids) --
fp16 is confirmed safe for the generation path this script uses
(predict_multiple_choice_answer, which never touches eager attention).

Candidates are loaded and run ONE AT A TIME -- fully processes every
sampled item for one candidate, unloads it (frees GPU/CPU memory), THEN
loads the next. This bounds peak memory to whichever single model is
currently loaded, regardless of how many precisions/models you sweep --
important on a real GPU with fixed VRAM (a V100 does not have room for
all 12 decoder-precision combos loaded simultaneously; this project's
Mac happened to tolerate it via unified memory, which is what let this
go unnoticed locally).

Mistral-7B / Falcon-7B are NOT included here -- too large for this
machine's 18GB RAM; run those on Roihu with their own Answerer instances
if needed later.

Bedrock tiers use profile "fsecure-golden-retriever-ci" / region
"eu-west-1" by default (--aws-profile/--aws-region to override) -- every
call to them is REAL, BILLED Bedrock usage. Check the sample size before a
large run.

Usage:
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10 --precisions fp16,fp32
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10 --include-opus
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_common import load_items  # noqa: E402 -- see cascade_common.py


DECODER_MODELS = [
    ("smollm2_135m", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smollm2_360m", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("qwen2_05", "Qwen/Qwen2-0.5B-Instruct"),
    ("qwen05", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("tinyllama11", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("falcon1b", "tiiuae/Falcon3-1B-Instruct"),
    ("qwen15", "Qwen/Qwen2.5-1.5B-Instruct"),
]

# Vetted in entropy_manual_review.ipynb (all verified to actually load --
# see that notebook's §2 for why each one was picked/excluded).
EXTRACTIVE_MODELS = [
    ("extractive_roberta_base", "deepset/roberta-base-squad2"),
    ("extractive_deberta_v3", "deepset/deberta-v3-base-squad2"),
    ("extractive_distilbert", "distilbert-base-cased-distilled-squad"),
    ("extractive_roberta_nonsquad", "consciousAI/question-answering-roberta-base-s-v2"),
]


def candidate_specs(precisions: list[str], include_opus: bool, skip_remote: bool,
                     aws_profile: str, aws_region: str) -> list[tuple[str, callable]]:
    """Returns [(name, load_fn), ...] -- load_fn() constructs (and loads
    the weights for) that one Answerer when called, nothing is loaded yet.
    The caller is responsible for calling load_fn(), using the result, then
    releasing it before moving to the next entry (see run_cascade)."""
    from question_answering.qa_model import DecoderOnlyQAModel, ExtractiveQAModel
    from question_answering.answerer import (BedrockConverseAnswerer, ClaudeBedrockAnswerer,
                                              LocalDecoderAnswerer, LocalExtractiveAnswerer)

    specs = [(name, lambda model_id=model_id: LocalExtractiveAnswerer(ExtractiveQAModel(model_id)))
             for name, model_id in EXTRACTIVE_MODELS]

    for name, model_id in DECODER_MODELS:
        for precision in precisions:
            specs.append((f"{name}_{precision}",
                          lambda model_id=model_id, precision=precision:
                              LocalDecoderAnswerer(DecoderOnlyQAModel(model_id, precision=precision))))

    if skip_remote:
        return specs

    specs.append(("haiku", lambda: ClaudeBedrockAnswerer(
        model="eu.anthropic.claude-haiku-4-5-20251001-v1:0", aws_region=aws_region, aws_profile=aws_profile,
    )))
    if include_opus:
        specs.append(("opus", lambda: ClaudeBedrockAnswerer(
            model="eu.anthropic.claude-opus-4-8", aws_region=aws_region, aws_profile=aws_profile,
        )))

    # Non-Anthropic Bedrock models -- verified working via BedrockConverseAnswerer
    # (converse API) on 2026-09-30, same account/region.
    for name, model_id in [
        ("nova_micro", "eu.amazon.nova-micro-v1:0"),
        ("nova_lite", "eu.amazon.nova-lite-v1:0"),
        ("nova_pro", "eu.amazon.nova-pro-v1:0"),
        ("glm47flash", "zai.glm-4.7-flash"),
    ]:
        specs.append((name, lambda model_id=model_id: BedrockConverseAnswerer(
            model=model_id, aws_region=aws_region, aws_profile=aws_profile,
        )))

    return specs


def _release(answerer) -> None:
    """Frees a local model's weights (no-op for remote answerers, which
    hold no local memory)."""
    model = getattr(answerer, "model", None)
    if model is not None:
        del model
    del answerer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def run_cascade(items: dict, sample_ids: list[str], specs: list[tuple[str, callable]],
                 output_path: Path) -> dict:
    """One candidate fully loaded and run over every sampled item at a
    time, then released, before the next candidate loads -- see module
    docstring for why (peak memory = one model, not sum of all of them).
    Results are item-keyed (not candidate-keyed) so run_cascade can be
    called multiple times against the same output file (e.g. once on
    Roihu for the local decoders, once locally for Haiku) and merged --
    see score_cascade_results.py's merge step."""
    results = json.loads(output_path.read_text()) if output_path.exists() else {}
    for item_id in sample_ids:
        results.setdefault(item_id, {"difficulty": items[item_id]["difficulty"], "answers": {}})

    for name, load_fn in specs:
        print(f"loading {name}...")
        answerer = load_fn()

        for i, item_id in enumerate(sample_ids):
            it = items[item_id]
            letter, confidence = answerer.answer(it["passage"], it["question"], it["options"])
            results[item_id]["answers"][name] = {"letter": letter, "confidence": confidence}
            if (i + 1) % 10 == 0 or (i + 1) == len(sample_ids):
                print(f"  [{name}] {i+1}/{len(sample_ids)} done")

        output_path.write_text(json.dumps(results, indent=2))
        print(f"  {name} done, checkpoint saved to {output_path}")
        _release(answerer)

    return results


def main() -> None:
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
    parser.add_argument("--aws-region", default="eu-west-1")
    parser.add_argument("--output", default=str(REPO_ROOT / "question_difficulty/scripts/cascade_results.json"))
    args = parser.parse_args()

    items = load_items()
    print(f"{len(items)} items available with IRT difficulty\n")

    random.seed(args.seed)
    sample_ids = random.sample(list(items.keys()), min(args.n_items, len(items)))

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    specs = candidate_specs(precisions, args.include_opus, args.skip_remote, args.aws_profile, args.aws_region)
    print(f"cascade: {[name for name, _ in specs]}\n")

    results = run_cascade(items, sample_ids, specs, Path(args.output))
    print(f"\nsaved to {args.output}")
    print("Run score_cascade_results.py on this file to add is_correct and print accuracy.")


if __name__ == "__main__":
    main()
