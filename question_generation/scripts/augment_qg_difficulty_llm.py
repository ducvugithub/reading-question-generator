#!/usr/bin/env python3
"""
Generate synthetic same-passage, cross-difficulty QG training examples via LLM.

Problem this fixes: every RACE passage belongs to exactly one difficulty
subset (RACE-middle->EASY, RACE-high->MEDIUM, RACE-C->HARD), so
diff-control-race never sees the same passage paired with more than one
difficulty token during training -- there's no contrastive signal forcing the
model to learn "the token controls output style" rather than "this passage's
vocabulary/style correlates with this token." See training_details.md's
"Known limitation" section for the full writeup.

This script asks an LLM to write additional questions for each passage at the
difficulty levels it does NOT naturally have -- e.g. an EASY (RACE-middle)
passage gets synthetic MEDIUM- and HARD-style questions written for it too,
in a SINGLE call per passage covering both missing levels at once. Mixing
these into training gives every passage all 3 difficulty tokens with
genuinely different targets: the missing contrastive signal.

Only augments the TRAIN split -- val/test stay 100% real data so evaluation
remains honest and comparable to the non-augmented baseline.

Supports both the direct Anthropic API and AWS Bedrock, via the same
LangChain wrappers already used elsewhere in this org (ChatAnthropic /
ChatBedrockConverse) -- not the raw anthropic SDK.

Usage:
  # Direct API (needs ANTHROPIC_API_KEY env var)
  python question_generation/scripts/augment_qg_difficulty_llm.py \
      --backend anthropic --model claude-opus-4-8 \
      --questions-per-level 2 --limit 50   # smoke test, 50 passages

  # AWS Bedrock (needs AWS credentials configured, e.g. via `aws sso login`)
  python question_generation/scripts/augment_qg_difficulty_llm.py \
      --backend bedrock --model haiku \
      --region us-east-1 --questions-per-level 2 --max-workers 8

Cost warning: with no --limit, this processes every unique train-split
passage (~23,894 for RACE) -- one LLM call each, asking for
--questions-per-level questions at each of its 2 missing difficulty levels.
Start with a small --limit to sanity check quality and cost before running
the full set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

_DIFFICULTIES = ["EASY", "MEDIUM", "HARD"]

# --backend bedrock only: friendly aliases for --model, resolved against
# `aws bedrock list-inference-profiles` output for this project's account/region.
# Bare foundation-model IDs (no prefix) fail with "on-demand throughput isn't
# supported" for these models -- must use an inference profile ID instead.
# Using the `global.` prefix to match the existing convention in
# simulation-platform's services/{studio,analyser}/core/llm_factory.py.
# Verify against your own account before trusting these — Bedrock model
# availability and IDs are account/region-specific.
BEDROCK_MODEL_ALIASES = {
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet-4.6": "global.anthropic.claude-sonnet-4-6",
    "sonnet-5": "global.anthropic.claude-sonnet-5",
}

_MC_PATTERNS = ("which of the following", "which one of the following", "which of these")


def _is_mc_question(question: str) -> bool:
    """RACE questions like "Which of the following is NOT true..." require answer
    options to make sense -- they're multiple-choice stems, incompatible with the
    open-ended/cloze format we want the LLM to write and mimic."""
    q = question.lower()
    return any(p in q for p in _MC_PATTERNS)


_DIFFICULTY_DESCRIPTIONS = {
    "EASY": "middle-school level: the answer is a single fact stated explicitly in one sentence of the passage",
    "MEDIUM": ("high-school level: requires connecting two or more sentences that are not adjacent, "
               "or a cause-and-effect the passage implies but never states in one place — must not be "
               "answerable by copying or paraphrasing a single sentence"),
    "HARD": ("college-entrance level: requires synthesizing the whole passage into a judgment not "
             "stated anywhere directly — the author's attitude, an implied theme/moral, or a "
             "generalization from multiple examples"),
}

_SYSTEM_PROMPT = ("You write RACE-style English reading-comprehension questions. Mix natural "
                  'WH-questions ("Why did...", "What can we infer...") with cloze/fill-in-the-blank '
                  'stems ("The man was surprised because _ .") — match whichever form the target '
                  "difficulty's typical openers call for, don't force everything into one style. "
                  "No multiple-choice options.")


class DifficultyQuestion(BaseModel):
    difficulty: str
    question: str


class SyntheticQuestions(BaseModel):
    questions: list[DifficultyQuestion]


def _build_prompt(passage: str, native_difficulty: str, other_difficulties: list[str],
                   existing_questions: list[str], exemplars: dict[str, list[str]],
                   questions_per_level: int) -> str:
    existing_block = "\n".join(f"- {q}" for q in existing_questions) if existing_questions else "(none)"

    level_blocks = []
    for d in other_difficulties:
        exemplar_lines = "\n".join(f"  - {q}" for q in exemplars.get(d, [])) or "  (no examples available)"
        level_blocks.append(
            f"- {d} ({_DIFFICULTY_DESCRIPTIONS[d]})\n"
            f"  Example {d}-level questions from OTHER passages (style reference only, don't reuse content):\n"
            f"{exemplar_lines}"
        )
    levels_block = "\n".join(level_blocks)

    return f"""Passage:
\"\"\"
{passage}
\"\"\"

This passage's native difficulty is {native_difficulty} ({_DIFFICULTY_DESCRIPTIONS[native_difficulty]}), \
with these existing question(s):
{existing_block}

Write {questions_per_level} new question(s) for EACH of these other difficulty levels:
{levels_block}

Each new question must differ from the existing ones above and from each other — no duplicates or \
near-paraphrases."""


def _build_llm(backend: str, model: str, region: str, temperature: float, max_tokens: int,
               profile: str | None = None):
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)

    if backend == "bedrock":
        import boto3
        from langchain_aws import ChatBedrockConverse

        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        client = session.client("bedrock-runtime", region_name=region)
        return ChatBedrockConverse(model=model, region_name=region, temperature=temperature,
                                    max_tokens=max_tokens, client=client)

    raise ValueError(f"Unknown backend: {backend}")


def _call_llm(structured_llm, passage: str, native_difficulty: str, other_difficulties: list[str],
              existing_questions: list[str], exemplars: dict[str, list[str]],
              questions_per_level: int, max_retries: int = 3) -> list[DifficultyQuestion]:
    prompt = _build_prompt(passage, native_difficulty, other_difficulties, existing_questions,
                           exemplars, questions_per_level)
    messages = [("system", _SYSTEM_PROMPT), ("user", prompt)]
    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke(messages)
            return result.questions
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [!] Failed after {max_retries} attempts: {e}", file=sys.stderr)
                return []
            time.sleep(2 ** attempt)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["anthropic", "bedrock"], default="anthropic",
                        help="anthropic = direct Anthropic API (needs ANTHROPIC_API_KEY); "
                             "bedrock = AWS Bedrock (needs AWS credentials configured)")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="Model ID. For --backend anthropic, use the Anthropic model string "
                             "(e.g. claude-opus-4-8). For --backend bedrock, use one of the "
                             f"BEDROCK_MODEL_ALIASES ({', '.join(BEDROCK_MODEL_ALIASES)}) or a full "
                             "Bedrock model ID from your AWS console.")
    parser.add_argument("--region", default="us-east-1", help="AWS region, --backend bedrock only")
    parser.add_argument("--profile", default=None,
                        help="AWS named profile, --backend bedrock only (e.g. from `aws sso login "
                             "--profile <name>`). Defaults to boto3's normal credential chain if unset.")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Higher = more variety across the questions-per-level draws")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--raw-dir", default="data/qg/raw/race")
    parser.add_argument("--output", default="data/qg/raw/race/train_synthetic.jsonl")
    parser.add_argument("--questions-per-level", type=int, default=2,
                        help="Synthetic questions to generate per missing difficulty level per passage")
    parser.add_argument("--num-exemplars", type=int, default=3,
                        help="Real few-shot example questions per difficulty level shown in the prompt")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent LLM requests")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N unique passages (for smoke testing / cost control)")
    args = parser.parse_args()

    if args.backend == "bedrock":
        resolved = BEDROCK_MODEL_ALIASES.get(args.model, args.model)
        if resolved != args.model:
            print(f"Resolved --model alias '{args.model}' -> '{resolved}'")
        args.model = resolved

    llm = _build_llm(args.backend, args.model, args.region, args.temperature, args.max_tokens, args.profile)
    structured_llm = llm.with_structured_output(SyntheticQuestions)

    train_path = Path(args.raw_dir) / "train.jsonl"
    if not train_path.exists():
        print(f"Error: {train_path} not found — run prepare_qg_test_sets.py first", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(l) for l in train_path.open(encoding="utf-8") if l.strip()]

    # Group existing real questions by passage, and note each passage's native difficulty.
    # Every record for a passage counts toward its difficulty (so a passage isn't dropped
    # just because all its questions happen to be MC-style), but MC-style questions
    # ("Which of the following...") are excluded from the DISPLAYED question list — they
    # need answer options to make sense, incompatible with the open-ended format we want
    # the LLM to mimic. A passage can end up with an empty (but present) questions list.
    by_passage: dict[str, dict] = {}
    for rec in records:
        p = rec["passage"]
        by_passage.setdefault(p, {"difficulty": rec["difficulty"], "questions": []})
        if not _is_mc_question(rec["question"]):
            by_passage[p]["questions"].append(rec["question"])

    passages = list(by_passage.items())
    if args.limit:
        passages = passages[:args.limit]
    print(f"Passages to augment: {len(passages)} (of {len(by_passage)} total in train split)")

    # Sample real exemplar questions per difficulty level, for prompt style grounding
    by_difficulty_questions = defaultdict(list)
    for rec in records:
        if _is_mc_question(rec["question"]):
            continue
        by_difficulty_questions[rec["difficulty"]].append(rec["question"])
    exemplars = {
        d: random.sample(qs, min(args.num_exemplars, len(qs)))
        for d, qs in by_difficulty_questions.items()
    }

    # Resume support: skip (passage, difficulty) pairs already in the output file
    output_path = Path(args.output)
    done_keys = set()
    if output_path.exists():
        for line in output_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["passage"], r["difficulty"]))
        print(f"Resuming — {len(done_keys)} (passage, difficulty) pairs already done")

    tasks = []
    for passage, info in passages:
        native_difficulty = info["difficulty"]
        other_difficulties = [d for d in _DIFFICULTIES
                              if d != native_difficulty and (passage, d) not in done_keys]
        if other_difficulties:
            tasks.append((passage, native_difficulty, other_difficulties, info["questions"]))

    print(f"Passages needing generation: {len(tasks)}")
    print(f"Estimated LLM calls: {len(tasks)} (one call per passage covers all its missing levels)")

    def _work(task):
        passage, native_difficulty, other_difficulties, existing_questions = task
        questions = _call_llm(structured_llm, passage, native_difficulty, other_difficulties,
                              existing_questions, exemplars, args.questions_per_level)
        return passage, questions

    fout = output_path.open("a", encoding="utf-8")
    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_work, t): t for t in tasks}
        for future in as_completed(futures):
            passage, questions = future.result()
            for dq in questions:
                out_rec = {
                    "passage": passage,
                    "question": dq.question,
                    "difficulty": dq.difficulty,
                    "source": "llm_synthetic",
                    "synthetic": True,
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            completed += 1
            if completed % 20 == 0:
                print(f"  {completed}/{len(tasks)} passages done", flush=True)
    fout.close()

    print(f"\n✓ Synthetic questions saved to {output_path}")
    print("\nNext: merge into training data before running prepare_qg_data.py, e.g.:")
    print(f"  cat {output_path} >> data/qg/raw/race/train.jsonl")


if __name__ == "__main__":
    main()
