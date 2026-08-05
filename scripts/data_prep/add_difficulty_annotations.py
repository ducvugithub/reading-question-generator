#!/usr/bin/env python3
"""
Annotate QG dataset with LLM-based difficulty scores.

Reads existing JSONL files, calls DifficultyAnnotator (one LLM call per unique passage),
and writes annotated files in-place. Each run adds a new key under llm_diff_judge keyed
by model name — running with Haiku then Sonnet accumulates both without overwriting.

Generated QA pairs (to fill passages to target_count) are appended to the split file.

Usage:
  # Bedrock Haiku (default)
  python scripts/add_difficulty_annotations.py --langs en fi ru

  # Specific model / provider
  python scripts/add_difficulty_annotations.py --langs en --provider anthropic --model claude-sonnet-4-6
  python scripts/add_difficulty_annotations.py --langs en --provider openai --model gpt-4o-mini

  # Dry-run: process first N passages only
  python scripts/add_difficulty_annotations.py --langs en --limit 5 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _sentence_count(text: str) -> int:
    return len(re.findall(r'[.!?]+', text))

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_llm(provider: str, model: str, region: str):
    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model=model, region_name=region, max_tokens=1500, temperature=0.3)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=1500, temperature=0.3)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=1500, temperature=0.3)
    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=1500,
            temperature=0.3,
        )
        return ChatHuggingFace(llm=endpoint)
    raise ValueError(f"Unknown provider: {provider}. Choose bedrock / anthropic / openai / huggingface")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def annotate_split(
    path: Path,
    annotator,
    target_count: int,
    limit: int | None,
    dry_run: bool,
    verbose: bool = False,
    min_sentences: int = 0,
) -> tuple[int, int, int]:
    """Returns (n_passages, n_scored, n_generated)."""
    from question_generation.difficulty.annotator import DifficultyAnnotator

    records = load_jsonl(path)
    if not records:
        return 0, 0, 0

    # Group by passage — one LLM call per unique passage
    passage_groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        passage_groups[rec["passage"]].append(i)

    model_name = annotator.model_name

    passages = list(passage_groups.items())
    if min_sentences > 0:
        passages = [(p, idx) for p, idx in passages if _sentence_count(p) >= min_sentences]
    # Skip passages where every existing record already has this model's scores
    passages = [
        (p, idx) for p, idx in passages
        if not all(model_name in records[i].get("llm_diff_judge", {}) for i in idx)
    ]
    if limit:
        passages = passages[:limit]

    n_scored = n_generated = 0

    checkpoint_every = 100
    for i, (passage_text, indices) in enumerate(tqdm(passages, desc=f"{path.parent.name}/{path.stem}", unit="passage")):
        first = records[indices[0]]
        qa_pairs = [(records[i]["question"], records[i]["answer"]) for i in indices]

        try:
            result = annotator.annotate(
                passage=passage_text,
                qa_pairs=qa_pairs,
                kg_raw=first.get("kg_raw"),
                kg_coref=first.get("kg_coref"),
            )
        except Exception as e:
            print(f"  [skip] {e.__class__.__name__}: {e}", flush=True)
            continue

        if verbose:
            print(f"\n  passage: {passage_text[:100]}...")
            print(f"  reasoning: {result.get('reasoning', '')[:120]}")
            print(f"  pass_readability={result['passage_readability']:.2f}  pass_vocab={result['passage_vocab_diff']:.2f}")
            for s in result.get("scored", []):
                print(f"  [scored] cog={s['question_cognitive_diff']:.2f} vocab={s['question_vocab_diff']:.2f}  Q: {s['question'][:80]}")
                print(f"           A: {s['answer'][:80]}")
            for g in result.get("generated", []):
                print(f"  [gen]    cog={g['question_cognitive_diff']:.2f} vocab={g.get('question_vocab_diff', 0):.2f}  Q: {g['question'][:80]}")
                print(f"           A: {g['answer'][:80]}")
            print(flush=True)

        passage_scores = {
            "passage_readability": result["passage_readability"],
            "passage_vocab_diff":  result["passage_vocab_diff"],
        }

        for idx, scored in zip(indices, result.get("scored", [])):
            records[idx].setdefault("llm_diff_judge", {})[model_name] = {
                **passage_scores,
                "question_cognitive_diff": scored["question_cognitive_diff"],
                "question_vocab_diff":     scored["question_vocab_diff"],
            }
            n_scored += 1

        for gen in result.get("generated", []):
            new_rec = {
                **{k: first[k] for k in ("passage", "kg_raw", "kg_coref",
                                          "source", "lang", "cefr") if k in first},
                "question":    gen["question"],
                "answer":      gen["answer"],
                "generated":   True,
                "generated_by": model_name,
                "llm_diff_judge": {
                    model_name: {
                        **passage_scores,
                        "question_cognitive_diff": gen["question_cognitive_diff"],
                        "question_vocab_diff":     gen["question_vocab_diff"],
                    }
                },
            }
            records.append(new_rec)
            n_generated += 1

        if not dry_run and (i + 1) % checkpoint_every == 0:
            save_jsonl(path, records)

    if not dry_run:
        save_jsonl(path, records)

    return len(passages), n_scored, n_generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     default="data/training")
    parser.add_argument("--langs",        nargs="+", default=["en", "fi", "ru"])
    parser.add_argument("--splits",       nargs="+", default=["train", "eval"])
    parser.add_argument("--provider",     default="bedrock",
                        choices=["bedrock", "anthropic", "openai", "huggingface"])
    parser.add_argument("--model",        default="anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--region",       default="us-east-1")
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--min-sentences", type=int, default=0,
                        help="Skip passages with fewer than N sentences")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Process only first N passages (for testing)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Run inference but do not write files")
    parser.add_argument("--verbose",      action="store_true",
                        help="Print each passage result to stdout")
    args = parser.parse_args()

    print(f"Provider: {args.provider}  Model: {args.model}", flush=True)
    llm = build_llm(args.provider, args.model, args.region)

    from question_generation.difficulty.annotator import DifficultyAnnotator
    annotator = DifficultyAnnotator(llm, target_count=args.target_count)
    print(f"Model name (for llm_diff_judge key): {annotator.model_name}\n", flush=True)

    data_dir = Path(args.data_dir)
    for lang in args.langs:
        for split in args.splits:
            path = data_dir / lang / f"{split}.jsonl"
            if not path.exists():
                continue
            n_pass, n_scored, n_gen = annotate_split(
                path, annotator, args.target_count, args.limit, args.dry_run, args.verbose,
                min_sentences=args.min_sentences,
            )
            status = "(dry-run)" if args.dry_run else "written"
            print(f"  {n_pass} passages, {n_scored} scored, {n_gen} generated — {status}",
                  flush=True)


if __name__ == "__main__":
    main()
