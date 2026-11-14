#!/usr/bin/env python3
"""
Re-scores an already-run cascade_results*.json using a bigger local Qwen
model as a semantic judge, instead of score_cascade_results.py's
letter-mapping + exact-text-match pipeline.

Why: run_answerer_cascade.py's blind answerers (LocalDecoderAnswerer/
LocalExtractiveAnswerer) never see the 4 options -- they generate free text
(saved as answers[name]["raw_text"]), which is then mapped to the closest
option via embedding cosine similarity (semantic_matcher.best_option_index).
That mapping step is a known noise source on paraphrase-heavy options (see
answerer.py's docstrings and semantic_matcher.py's module docstring). This
script skips the letter-mapping step entirely and asks a judge model
directly: "does this free-text answer mean the same thing as the gold
answer?" -- True/False, no embedding threshold involved.

No new cascade run needed -- raw_text was already saved per item per
candidate specifically so this kind of re-scoring fix could happen without
re-running the (expensive) answerer models (see run_answerer_cascade.py's
run_cascade() comment, 2026-11-11).

Judge model: Qwen2.5-7B-Instruct, fp16 -- deliberately bigger than qwen15
(Qwen2.5-1.5B-Instruct, one of the 13 cascade candidates being judged) so
the judge isn't just as noisy as what it's grading.

Usage (on Roihu, GPU):
  python question_difficulty/scripts/qwen_judge_rescore.py \\
      question_difficulty/scripts/cascade_results_race_pool_clean13k.json \\
      --item-source race_pool \\
      --output question_difficulty/scripts/cascade_results_race_pool_clean13k_qwenjudge.json

Resumable: writes a checkpoint every CHECKPOINT_EVERY items and skips any
item whose candidates are all already judged in an existing --output file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade_common import load_items, load_race_training_pool, classify_refusal, is_extractive_candidate  # noqa: E402

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_EVERY = 200
BATCH_SIZE = 16

_VERDICT_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def build_prompt(tokenizer, passage: str, question: str, raw_text: str, gold: str) -> str:
    user_text = (
        "You are judging whether a candidate's answer to a reading-comprehension "
        "question means the same thing as the correct answer. Ignore wording/style "
        "differences -- judge meaning only.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n"
        f"Correct answer: {gold}\n"
        f"Candidate's answer: {raw_text}\n\n"
        "Does the candidate's answer convey the same meaning as the correct answer? "
        "Reply with exactly one word: True or False."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True,
    )


def parse_verdict(text: str) -> bool | None:
    m = _VERDICT_RE.search(text)
    if not m:
        return None
    return m.group(1).lower() == "true"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return tokenizer, model, torch, device


def judge_batch(tokenizer, model, torch, device, prompts: list[str]) -> list[bool | None]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=4, do_sample=False)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [parse_verdict(t) for t in texts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file")
    parser.add_argument("--item-source", default="race_pool", choices=["onestopqa", "race_pool"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = json.loads(Path(args.results_file).read_text())
    items = load_race_training_pool() if args.item_source == "race_pool" else load_items()
    gold_by_item = {iid: it["correct_answer_text"] for iid, it in items.items()}

    output_path = Path(args.output) if args.output else Path(args.results_file).with_name(
        Path(args.results_file).stem + "_qwenjudge.json")

    judged: dict = json.loads(output_path.read_text()) if output_path.exists() else {}

    # Build the flat worklist of (item_id, candidate_name, raw_text, gold) that
    # still needs judging -- skips refusals (same convention as
    # score_cascade_results.py: "needs_options" is excluded entirely, nothing
    # to judge; "content_gap" is a real wrong answer, scored False without a
    # judge call since there's no raw answer content to compare against gold).
    work = []
    for item_id, row in results.items():
        gold = gold_by_item.get(item_id)
        if gold is None:
            continue
        judged.setdefault(item_id, {"gold_answer": gold, "answers": {}})
        for name, answer in row.get("answers", {}).items():
            if name in judged[item_id]["answers"]:
                continue  # already judged (resume)
            raw_text = answer.get("raw_text")
            refusal_kind = None if is_extractive_candidate(name) else classify_refusal(raw_text)
            if refusal_kind == "needs_options":
                judged[item_id]["answers"][name] = {"qwen_is_correct": None, "skipped": "needs_options"}
                continue
            if refusal_kind == "content_gap":
                judged[item_id]["answers"][name] = {"qwen_is_correct": False, "skipped": "content_gap"}
                continue
            if not raw_text or not raw_text.strip():
                judged[item_id]["answers"][name] = {"qwen_is_correct": False, "skipped": "empty_raw_text"}
                continue
            work.append((item_id, name, raw_text, gold))

    print(f"{len(work)} (item, candidate) pairs need judging "
          f"({sum(len(r['answers']) for r in judged.values())} already done/skipped)", flush=True)
    if not work:
        output_path.write_text(json.dumps(judged, indent=2))
        print(f"nothing to do, wrote {output_path}")
        return

    print(f"loading judge model {JUDGE_MODEL}...", flush=True)
    tokenizer, model, torch, device = load_model()
    print("loaded, judging...", flush=True)

    items_by_id = items
    done_since_checkpoint = 0
    for start in range(0, len(work), BATCH_SIZE):
        chunk = work[start:start + BATCH_SIZE]
        prompts = [
            build_prompt(tokenizer, items_by_id[iid]["passage"], items_by_id[iid]["question"], raw_text, gold)
            for iid, _, raw_text, gold in chunk
        ]
        verdicts = judge_batch(tokenizer, model, torch, device, prompts)
        for (item_id, name, raw_text, gold), verdict in zip(chunk, verdicts):
            judged[item_id]["answers"][name] = {"qwen_is_correct": verdict, "raw_text": raw_text}
        done_since_checkpoint += len(chunk)

        if done_since_checkpoint >= CHECKPOINT_EVERY or start + BATCH_SIZE >= len(work):
            output_path.write_text(json.dumps(judged, indent=2))
            print(f"  {min(start + BATCH_SIZE, len(work))}/{len(work)} judged, checkpoint saved", flush=True)
            done_since_checkpoint = 0

    print(f"done, wrote {output_path}")

    accuracy: dict[str, list[bool]] = {}
    for row in judged.values():
        for name, answer in row["answers"].items():
            v = answer.get("qwen_is_correct")
            if v is not None:
                accuracy.setdefault(name, []).append(v)
    print("\nAccuracy per candidate (qwen-judged):")
    for name, flags in sorted(accuracy.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        n_correct = sum(flags)
        print(f"  {name:16s} {n_correct}/{len(flags)} ({n_correct/len(flags):.1%})")


if __name__ == "__main__":
    main()
