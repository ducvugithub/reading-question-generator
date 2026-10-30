#!/usr/bin/env python3
"""
Cascade experiment: run a lineup of Answerers (weak -> strong, local ->
remote) over sampled OneStopQA/RACE items. Records each candidate's raw
(letter, confidence) per item ONLY -- no correctness scoring here, see
score_cascade_results.py for that (cheap, local, pure string matching --
doesn't need this script's models loaded at all, so it's kept separate).

Every candidate is one row in CANDIDATES: (name, type, model_id,
default_on). `type` picks which Answerer wrapper class to use (see
_make_loader) -- "extractive" (question_answering.qa_model.ExtractiveQAModel,
local, no precision axis), "decoder" (DecoderOnlyQAModel, local, expanded
once per --precisions entry), "claude_bedrock" (ClaudeBedrockAnswerer,
Anthropic SDK's Bedrock client -- Claude only, converse() gives Claude a
verbose non-letter answer on this prompt so it stays on this class), or
"bedrock_converse" (BedrockConverseAnswerer, boto3's provider-agnostic
converse API -- everything else on Bedrock: Nova, GLM, future providers),
or "laya" (LayaAnswerer, local non-autoregressive decision model -- see
its commented-out CANDIDATES entry for why it's not enabled).
Adding a new candidate is one line in CANDIDATES, never a code change.

Which candidates actually run is controlled by --types/--include/--exclude,
not a growing pile of one-off boolean flags:
  included = (name in --include) or (type in --types and default_on)
  then dropped if name in --exclude
So e.g. "skip everything remote" is `--types extractive,decoder`; "just
try the one new model I added" is `--include my_new_model --types ""`;
"run everything except the slow one" is `--exclude qwen15`. `opus` has
default_on=False (real cost, bigger than Haiku) -- add `--include opus`
to run it.

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
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10 --types extractive,decoder
  python question_difficulty/scripts/run_answerer_cascade.py --n-items 10 --include opus
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

from cascade_common import load_items, load_race_training_pool  # noqa: E402 -- see cascade_common.py


# (name, type, model_id, default_on) -- see module docstring for `type` meanings
# and how default_on interacts with --types/--include/--exclude.
CANDIDATES = [
    # Vetted in entropy_manual_review.ipynb (all verified to actually load --
    # see that notebook's §2 for why each one was picked/excluded).
    ("extractive_roberta_base", "extractive", "deepset/roberta-base-squad2", True),
    ("extractive_deberta_v3", "extractive", "deepset/deberta-v3-base-squad2", True),
    ("extractive_distilbert", "extractive", "distilbert-base-cased-distilled-squad", True),
    ("extractive_roberta_nonsquad", "extractive", "consciousAI/question-answering-roberta-base-s-v2", True),

    ("smollm2_135m", "decoder", "HuggingFaceTB/SmolLM2-135M-Instruct", True),
    ("smollm2_360m", "decoder", "HuggingFaceTB/SmolLM2-360M-Instruct", True),
    ("qwen2_05", "decoder", "Qwen/Qwen2-0.5B-Instruct", True),          # older/weaker generation than qwen05
    ("qwen05", "decoder", "Qwen/Qwen2.5-0.5B-Instruct", True),
    ("tinyllama11", "decoder", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", True),  # older/less-optimized recipe
    ("falcon1b", "decoder", "tiiuae/Falcon3-1B-Instruct", True),
    ("qwen15", "decoder", "Qwen/Qwen2.5-1.5B-Instruct", True),

    # Added for architecture diversity: the local decoder pool above is
    # Qwen-heavy (3 of 7 entries are Qwen variants), so a shared training-data/
    # recipe quirk across them (e.g. the main-idea-questions-look-easy pattern
    # found in diff-control-irt-10k's manual review) can't be told apart from
    # a genuine cross-architecture LLM property with only this lineup. Llama
    # and Gemma are both GATED on HF -- license must be accepted on the model
    # page with the same HF account/token used for HF_HUB_OFFLINE caching, or
    # these fail to download (not an offline-mode bug, an auth one).
    ("llama32_1b", "decoder", "meta-llama/Llama-3.2-1B-Instruct", True),
    ("gemma3_1b", "decoder", "google/gemma-3-1b-it", True),

    ("haiku", "claude_bedrock", "eu.anthropic.claude-haiku-4-5-20251001-v1:0", True),
    ("opus", "claude_bedrock", "eu.anthropic.claude-opus-4-8", False),  # off by default: real, larger cost per item

    # Non-Anthropic Bedrock models -- verified working via BedrockConverseAnswerer
    # (converse API) on 2026-09-30, same account/region.
    ("nova_micro", "bedrock_converse", "eu.amazon.nova-micro-v1:0", True),
    ("nova_lite", "bedrock_converse", "eu.amazon.nova-lite-v1:0", True),
    ("nova_pro", "bedrock_converse", "eu.amazon.nova-pro-v1:0", True),
    ("glm47flash", "bedrock_converse", "zai.glm-4.7-flash", True),
    ("gemma3_4b", "bedrock_converse", "google.gemma-3-4b-it", True),

    # Tried manually (not via this script) on 5-7 sample items on 2026-10-02:
    # ~2/7 correct, near the 25% chance floor -- out of the box, Laya doesn't
    # read-comprehend at a useful level for this task (see LayaAnswerer's
    # docstring in question_answering/answerer.py for the full writeup,
    # including why 512-token context turned out not to be the blocker).
    # Left commented out rather than removed: a candidate worth revisiting
    # IF it gets fine-tuned on RACE/OneStopQA-style MC data first.
    # ("laya", "laya", "convaiinnovations/laya", False),
]

ALL_TYPES = ["extractive", "decoder", "claude_bedrock", "bedrock_converse"]


def _make_loader(type_: str, model_id: str, precision: str | None,
                  aws_profile: str, aws_region: str) -> callable:
    """One zero-arg factory that constructs (and loads the weights for)
    the Answerer for this (type, model_id[, precision]) -- nothing is
    loaded until it's called. See run_cascade for the load/run/release
    cycle this feeds into."""
    if type_ == "extractive":
        from question_answering.qa_model import ExtractiveQAModel
        from question_answering.answerer import LocalExtractiveAnswerer
        return lambda: LocalExtractiveAnswerer(ExtractiveQAModel(model_id))
    if type_ == "decoder":
        from question_answering.qa_model import DecoderOnlyQAModel
        from question_answering.answerer import LocalDecoderAnswerer
        return lambda: LocalDecoderAnswerer(DecoderOnlyQAModel(model_id, precision=precision))
    if type_ == "claude_bedrock":
        from question_answering.answerer import ClaudeBedrockAnswerer
        return lambda: ClaudeBedrockAnswerer(model=model_id, aws_region=aws_region, aws_profile=aws_profile)
    if type_ == "bedrock_converse":
        from question_answering.answerer import BedrockConverseAnswerer
        return lambda: BedrockConverseAnswerer(model=model_id, aws_region=aws_region, aws_profile=aws_profile)
    if type_ == "laya":
        from question_answering.answerer import LayaAnswerer
        return lambda: LayaAnswerer(repo=model_id)
    raise ValueError(f"unknown candidate type {type_!r}")


def candidate_specs(precisions: list[str], types: list[str], include: list[str], exclude: list[str],
                     aws_profile: str, aws_region: str) -> list[tuple[str, callable]]:
    """Returns [(name, load_fn), ...] in CANDIDATES order, filtered per
    the module docstring's include/exclude/types rule. `precisions`
    only applies to type="decoder" entries (each expands into one
    candidate per precision, e.g. qwen15 -> qwen15_fp32, qwen15_fp16, ...)."""
    specs = []
    for name, type_, model_id, default_on in CANDIDATES:
        included = name in include or (type_ in types and default_on)
        if not included or name in exclude:
            continue
        if type_ == "decoder":
            for precision in precisions:
                specs.append((f"{name}_{precision}", _make_loader(type_, model_id, precision, aws_profile, aws_region)))
        else:
            specs.append((name, _make_loader(type_, model_id, None, aws_profile, aws_region)))
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
    see score_cascade_results.py's merge step.

    A candidate already complete for every sample_id in the existing
    output file is skipped entirely (not reloaded/rerun) -- makes
    re-running after a timeout/interruption (e.g. a long SLURM job hitting
    its walltime) resume rather than redo already-finished candidates.
    Checkpointing is still per-candidate, not per-item: a candidate
    interrupted PARTWAY through has no items recorded for it yet (nothing
    is written until the candidate's full item loop finishes), so it
    reruns from scratch, not from wherever it left off."""
    results = json.loads(output_path.read_text()) if output_path.exists() else {}
    for item_id in sample_ids:
        results.setdefault(item_id, {"difficulty": items[item_id]["difficulty"], "answers": {}})

    for name, load_fn in specs:
        if all(name in results[item_id]["answers"] for item_id in sample_ids):
            print(f"skipping {name}, already complete for all {len(sample_ids)} items")
            continue
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
                         help="Comma-separated precisions for type=decoder candidates: "
                              "fp32,fp16,int8,int4 (int8/int4 need bitsandbytes + CUDA, e.g. Roihu)")
    parser.add_argument("--types", default=",".join(ALL_TYPES),
                         help=f"Comma-separated candidate types to include (default all: {','.join(ALL_TYPES)}). "
                              "E.g. --types extractive,decoder to drop everything remote.")
    parser.add_argument("--include", default="",
                         help="Comma-separated candidate names to force-include even if their type "
                              "is excluded or default_on=False (e.g. --include opus)")
    parser.add_argument("--exclude", default="",
                         help="Comma-separated candidate names to force-exclude even if their type "
                              "is included and default_on=True")
    parser.add_argument("--aws-profile", default="fsecure-golden-retriever-ci")
    parser.add_argument("--aws-region", default="eu-west-1")
    parser.add_argument("--output", default=str(REPO_ROOT / "question_difficulty/scripts/cascade_results.json"))
    parser.add_argument("--item-source", default="onestopqa", choices=["onestopqa", "race_pool"],
                         help="onestopqa (default): the 1,296 OneStopQA/RACE items with real human "
                              "responses. race_pool: the ~87K-question RACE training pool QG actually "
                              "trains on (no human data, no pre-existing difficulty) -- for enriching "
                              "cascade-IRT coverage at QG-training scale, see redo_irt_excluding_broken_gold.py's "
                              "module docstring and cascade_common.load_race_training_pool()'s docstring.")
    parser.add_argument("--sample-level", default="question", choices=["question", "passage"],
                         help="question (default): sample --n-items individual questions uniformly at "
                              "random across the whole pool -- for race_pool, this scatters across "
                              "passages (RACE averages ~3.65 questions/passage), so most sampled "
                              "passages end up with only 1 of their real questions included, which "
                              "makes passage-mean-centering degenerate for them (see "
                              "question_generation/scripts/prepare_pilot_irt_difficulty_data.py's "
                              "docstring for why centering needs >=2 questions/passage). passage: sample "
                              "whole passages instead, including ALL of each sampled passage's "
                              "questions -- guarantees every included passage supports centering, at "
                              "the cost of landing somewhat above --n-items (stops once the cumulative "
                              "item count reaches --n-items, so the last passage's full question set "
                              "is always included even if it overshoots slightly).")
    args = parser.parse_args()

    items = load_race_training_pool() if args.item_source == "race_pool" else load_items()
    print(f"{len(items)} items available ({args.item_source})\n")

    random.seed(args.seed)
    if args.sample_level == "passage":
        by_passage: dict[str, list[str]] = {}
        for item_id, it in items.items():
            by_passage.setdefault(it["passage"], []).append(item_id)
        passages = list(by_passage.keys())
        random.shuffle(passages)
        sample_ids = []
        n_passages_used = 0
        for passage in passages:
            if len(sample_ids) >= args.n_items:
                break
            sample_ids.extend(by_passage[passage])
            n_passages_used += 1
        print(f"sampled {len(sample_ids)} items from {n_passages_used} whole passages "
              f"(target was {args.n_items})")
    else:
        sample_ids = random.sample(list(items.keys()), min(args.n_items, len(items)))

    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    include = [n.strip() for n in args.include.split(",") if n.strip()]
    exclude = [n.strip() for n in args.exclude.split(",") if n.strip()]
    specs = candidate_specs(precisions, types, include, exclude, args.aws_profile, args.aws_region)
    print(f"cascade: {[name for name, _ in specs]}\n")

    results = run_cascade(items, sample_ids, specs, Path(args.output))
    print(f"\nsaved to {args.output}")
    print("Run score_cascade_results.py on this file to add is_correct and print accuracy.")


if __name__ == "__main__":
    main()
