#!/usr/bin/env python3
"""
Infer question difficulty using LLM (zero-shot), save predictions to file.

Prompts an LLM to rate each (passage, question, answer) triple as EASY / MEDIUM / HARD.
Saves predictions to JSONL (one per line with true/pred labels for later evaluation).

No training required. Use with evaluate_qde.py --llm-predictions to compare against trained methods.

Supports: Claude API (default), AWS Bedrock, local models (Ollama/vLLM).

Usage:
  # Claude API (cheapest, ~$0.94 for 3755 examples)
  export ANTHROPIC_API_KEY=sk-ant-...
  python question_difficulty/scripts/qde_infer_llm.py \
    --output question_difficulty/results/predictions_llm_verdict.jsonl --limit 3755

  # AWS Bedrock
  python question_difficulty/scripts/qde_infer_llm.py --backend bedrock \
    --output predictions.jsonl --limit 3755

  # Local model (Ollama)
  ollama run llama2
  python question_difficulty/scripts/qde_infer_llm.py --backend local \
    --model llama2 --output predictions.jsonl --limit 3755

Then evaluate with: evaluate_qde.py --llm-predictions predictions.jsonl
"""
from __future__ import annotations

import abc
import argparse
import json
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm

_LABEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
_LABEL_NAMES = ["EASY", "MEDIUM", "HARD"]

_SYSTEM_PROMPT = """\
You are an expert in reading comprehension assessment.
Given a passage, a question, and the correct answer, classify the question difficulty.

Respond with EXACTLY one word on the first line: EASY, MEDIUM, or HARD.
Then on the next lines give a one-sentence justification.

Difficulty guide:
  EASY   — Factual recall; answer is directly stated in the passage; simple vocabulary.
  MEDIUM — Requires inference or connecting two pieces of information; moderate vocabulary.
  HARD   — Requires multi-step reasoning, synthesis, or abstract/critical thinking; complex vocabulary.
"""

_USER_TEMPLATE = """\
PASSAGE:
{passage}

QUESTION: {question}

ANSWER: {answer}

Difficulty (EASY / MEDIUM / HARD):"""


class LLMBackend(abc.ABC):
    """Abstract base for LLM backends."""

    @abc.abstractmethod
    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 64) -> str:
        """Query the LLM and return the response text."""
        pass


class AnthropicBackend(LLMBackend):
    """Claude API via Anthropic SDK."""

    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError:
            print("Error: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        self.client = anthropic.Anthropic()
        self.model = model

    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 64) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()


class BedrockBackend(LLMBackend):
    """AWS Bedrock backend using langchain."""

    def __init__(self, model: str, region: str = "us-east-1", profile: str | None = None):
        try:
            from langchain_aws import ChatBedrockConverse
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:
            print("Error: langchain-aws and langchain-core required. Run: pip install langchain-aws langchain-core")
            sys.exit(1)
        self.llm = ChatBedrockConverse(
            model=model,
            region_name=region,
            temperature=0,
            max_tokens=64,
        )
        self.SystemMessage = SystemMessage
        self.HumanMessage = HumanMessage

    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 64) -> str:
        messages = [
            self.SystemMessage(content=system_prompt),
            self.HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()


class LocalBackend(LLMBackend):
    """Local model backend (Ollama, vLLM, etc.)."""

    def __init__(self, model: str, url: str = "http://localhost:11434"):
        try:
            import requests
        except ImportError:
            print("Error: requests package not installed. Run: pip install requests")
            sys.exit(1)
        self.model = model
        self.url = url.rstrip("/")
        self.requests = requests

    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 64) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        result = response.json()
        return result.get("response", "").strip()


def _parse_label(text: str) -> int | None:
    """Extract the first occurrence of EASY/MEDIUM/HARD from the response."""
    m = re.search(r'\b(EASY|MEDIUM|HARD)\b', text.upper())
    if m:
        return _LABEL_MAP[m.group(1).lower()]
    return None


def evaluate_split(
    records: list[dict],
    backend: LLMBackend,
    max_passage_words: int,
    delay: float,
) -> tuple[list[int], list[int]]:
    """Call the LLM for each record; return (true_labels, pred_labels)."""
    true_labels, pred_labels = [], []
    n = len(records)

    for i, rec in tqdm(enumerate(records), total=n, desc="Inferring"):
        true_label = _LABEL_MAP.get(rec["difficulty"].lower())
        if true_label is None:
            continue

        # Truncate very long passages to save tokens
        words = rec["passage"].split()
        passage = " ".join(words[:max_passage_words]) if len(words) > max_passage_words else rec["passage"]

        user_msg = _USER_TEMPLATE.format(
            passage=passage,
            question=rec["question"],
            answer=rec["answer"],
        )

        text = None
        try:
            text = backend.query(_SYSTEM_PROMPT, user_msg, max_tokens=64)
            pred = _parse_label(text)
        except Exception as e:
            print(f"  [!] API error on record {i}: {e}", flush=True)
            pred = None

        if pred is None:
            if text:
                print(f"  [!] Unparseable response on record {i}: {text!r}", flush=True)
            pred = 1  # fallback to MEDIUM

        true_labels.append(true_label)
        pred_labels.append(pred)

        if delay > 0:
            time.sleep(delay)

    return true_labels, pred_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend",     default="anthropic",
                        choices=["anthropic", "bedrock", "local"],
                        help="LLM backend (default: anthropic)")
    parser.add_argument("--model",       default=None,
                        help="Model ID (default: claude-haiku-4-5 for anthropic, anthropic.claude-haiku-4-5-20251001-v1:0 for bedrock)")
    parser.add_argument("--bedrock-region", default="us-east-1",
                        help="AWS region for Bedrock (default: us-east-1)")
    parser.add_argument("--aws-profile",  default=None,
                        help="AWS profile name (uses default if not specified)")
    parser.add_argument("--local-url",   default="http://localhost:11434",
                        help="URL for local model server (default: http://localhost:11434)")
    parser.add_argument("--data-dir",    default="data/qde")
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max records to evaluate (cost control)")
    parser.add_argument("--max-passage-words", type=int, default=400,
                        help="Truncate passages to this many words")
    parser.add_argument("--delay",       type=float, default=0.1,
                        help="Seconds between API calls (rate limiting)")
    parser.add_argument("--output",      default=None,
                        help="Write per-record predictions to this JSONL file")
    args = parser.parse_args()

    # Set default model based on backend
    if args.model is None:
        if args.backend == "anthropic":
            args.model = "claude-haiku-4-5"
        elif args.backend == "bedrock":
            args.model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        elif args.backend == "local":
            args.model = "llama2"

    # Initialize backend
    if args.backend == "anthropic":
        backend = AnthropicBackend(args.model)
    elif args.backend == "bedrock":
        backend = BedrockBackend(args.model, region=args.bedrock_region, profile=args.aws_profile)
    elif args.backend == "local":
        backend = LocalBackend(args.model, url=args.local_url)
    else:
        print(f"Error: unknown backend {args.backend}")
        sys.exit(1)

    data_path = Path(args.data_dir) / f"{args.split}.jsonl"
    if not data_path.exists():
        print(f"Error: {data_path} not found — run prepare_qde_data.py first")
        sys.exit(1)

    records = [json.loads(l) for l in data_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        import random
        random.Random(42).shuffle(records)
        records = records[:args.limit]

    print(f"Backend : {args.backend}")
    print(f"Model  : {args.model}")
    print(f"Split  : {args.split}  ({len(records)} records)")
    if args.output:
        print(f"Output : {args.output}")
    print(f"Inferring...", flush=True)

    true_labels, pred_labels = evaluate_split(
        records, backend, args.max_passage_words, args.delay
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec, true, pred in zip(records, true_labels, pred_labels):
                f.write(json.dumps({
                    "passage":   rec["passage"],
                    "question":  rec["question"],
                    "answer":    rec["answer"],
                    "difficulty": rec["difficulty"],
                    "true_label": _LABEL_NAMES[true],
                    "pred_label": _LABEL_NAMES[pred],
                    "correct": true == pred,
                }, ensure_ascii=False) + "\n")
        print(f"✓ Predictions saved to {out_path}")
        print(f"\nNext: python question_difficulty/scripts/evaluate_qde.py --llm-predictions {out_path}")


if __name__ == "__main__":
    main()
