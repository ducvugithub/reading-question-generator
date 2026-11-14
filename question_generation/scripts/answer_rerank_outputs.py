#!/usr/bin/env python3
"""
Generate a free-text gold answer for each sample_rerank_diffcond.py output,
via Claude on Bedrock (same client pattern as ClaudeBedrockAnswerer in
question_answering/answerer.py, but open-ended -- no 4 options to pick
from, since these are newly generated questions with none).

This is the missing piece needed to score a generated question's IRT
difficulty with question_difficulty/models/encoder_irt_pilot_regression:
that encoder's input format is "question [answer: ...]" + passage, so an
answer string is required, but NOT 4 distractor options -- much cheaper
than building a full MCQ option-generation step.

Usage:
  python question_generation/scripts/answer_rerank_outputs.py \
    --input question_generation/results/sample_rerank.jsonl \
    --output question_generation/results/sample_rerank_with_answers.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "eu.anthropic.claude-sonnet-5-5"


def answer_question(client, passage: str, question: str) -> str:
    prompt = (
        f"Passage: {passage}\n\nQuestion: {question}\n\n"
        "Answer the question using only information from the passage. "
        "Respond with just the answer, one sentence, no preamble."
    )
    response = client.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 64},
        additionalModelRequestFields={"thinking": {"type": "between_tools"}},
    )
    for block in response["output"]["message"]["content"]:
        if "text" in block:
            return block["text"].strip()
    raise ValueError(f"No text block in response content: {response['output']['message']['content']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aws-profile", default="fsecure-golden-retriever-ci")
    parser.add_argument("--aws-region", default="eu-west-1")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.aws_profile, region_name=args.aws_region)
    client = session.client("bedrock-runtime")

    records = [json.loads(l) for l in (REPO_ROOT / args.input).open(encoding="utf-8") if l.strip()]
    out_path = REPO_ROOT / args.output
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rec["answer"] = answer_question(client, rec["passage"], rec["question"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(records)}] [{rec['difficulty']}] Q: {rec['question'][:70]}  A: {rec['answer'][:70]}")

    print(f"\nWrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
