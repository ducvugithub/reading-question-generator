#!/usr/bin/env python3
"""
Prepare T5 QG training data for Steps 0, 2, 3.

  Step 0 (baseline):  context: {passage} → question
                      Sources: RACE-middle + RACE-high + RACE-C + HotpotQA + MultiRC
                      No conditioning — unconditional QG baseline.

  Step 2 (diff QG):   difficulty: {EASY|MEDIUM|HARD} context: {passage} → question
                      Sources: RACE-middle→EASY, RACE-high→MEDIUM, RACE-C→HARD

  Step 3 (span QG):   focus: {evidence_sentences} context: {passage} → question
                      Sources: HotpotQA (type==comparison) + MultiRC (allenai/multirc)

  Step 4 (M6 full):   Not implemented here — requires QDE-enriched data from Step 1.
                      Run after QDE training; enrich HotpotQA/MultiRC with EASY/MEDIUM/HARD labels.

Passage-level deterministic 80/10/10 split via MD5.

Usage:
  python question_generation/scripts/prepare_qg_data.py --steps step0
  python question_generation/scripts/prepare_qg_data.py --steps step2
  python question_generation/scripts/prepare_qg_data.py --steps step3
  python question_generation/scripts/prepare_qg_data.py --steps step0 step2 step3
  python question_generation/scripts/prepare_qg_data.py --steps step0 --limit 500  # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator


def _split_bucket(passage: str, train_r: float, val_r: float) -> str:
    h = int(hashlib.md5(passage.encode()).hexdigest(), 16) % 100
    if h < train_r * 100:
        return "train"
    if h < (train_r + val_r) * 100:
        return "val"
    return "test"


def _format_input(step: str, passage: str, difficulty: str = "", focus: str = "") -> str:
    if step == "step0":
        return f"generate question: context: {passage}"
    if step == "step2":
        return f"generate question: difficulty: {difficulty} context: {passage}"
    if step == "step3":
        return f"generate question: focus: {focus} context: {passage}"
    if step == "step4":
        return f"generate question: difficulty: {difficulty} focus: {focus} context: {passage}"
    raise ValueError(f"Unknown step: {step}")


# ── dataset iterators ────────────────────────────────────────────────────────

def _iter_race(subset: str, difficulty: str, limit: int | None) -> Iterator[dict]:
    """ehovy/race middle or high — answer field is letter A/B/C/D."""
    from datasets import load_dataset
    ds = load_dataset("ehovy/race", subset, split="train")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    count = 0
    for rec in ds:
        idx = letter_to_idx.get(rec["answer"])
        if idx is None or idx >= len(rec["options"]):
            continue
        yield {"passage": rec["article"], "question": rec["question"],
               "difficulty": difficulty, "source": f"race_{subset}"}
        count += 1
        if limit and count >= limit:
            break


def _iter_race_c(limit: int | None) -> Iterator[dict]:
    """tasksource/race-c — answer field is integer 0–3."""
    from datasets import load_dataset
    ds = load_dataset("tasksource/race-c", split="train")
    count = 0
    for rec in ds:
        if rec["label"] is None or rec["label"] >= len(rec["option"]):
            continue
        yield {"passage": rec["article"], "question": rec["question"],
               "difficulty": "HARD", "source": "race_c"}
        count += 1
        if limit and count >= limit:
            break


def _gold_passage(rec: dict) -> str:
    """Concatenate the 2 supporting passages from a HotpotQA record."""
    gold_titles = set(rec["supporting_facts"]["title"])
    parts = []
    for title, sents in zip(rec["context"]["title"], rec["context"]["sentences"]):
        if title in gold_titles:
            parts.append(" ".join(sents))
    return " ".join(parts)


def _focus_span(rec: dict) -> str:
    """Extract supporting_facts sentences as the focus span."""
    sf_map: dict[str, set] = {}
    for title, sid in zip(rec["supporting_facts"]["title"], rec["supporting_facts"]["sent_id"]):
        sf_map.setdefault(title, set()).add(sid)
    parts = []
    for title, sents in zip(rec["context"]["title"], rec["context"]["sentences"]):
        if title in sf_map:
            for i, s in enumerate(sents):
                if i in sf_map[title]:
                    parts.append(s.strip())
    return " ".join(parts)


def _iter_hotpotqa(step: str, limit: int | None) -> Iterator[dict]:
    """hotpotqa/hotpot_qa distractor split.
    step0: all types, gold passage as context.
    step3: comparison type only, gold passage + supporting_facts as focus.
    """
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    count = 0
    for rec in ds:
        if step == "step3" and rec["type"] != "comparison":
            continue
        passage = _gold_passage(rec)
        if not passage:
            continue
        entry: dict = {"passage": passage, "question": rec["question"], "source": "hotpotqa"}
        if step == "step3":
            span = _focus_span(rec)
            if not span:
                continue
            entry["focus"] = span
        yield entry
        count += 1
        if limit and count >= limit:
            break


def _iter_multirc(step: str, limit: int | None) -> Iterator[dict]:
    """allenai/multirc — original dataset with evidence annotations.

    The allenai/multirc dataset has nested structure:
      paragraph.text, paragraph.questions[].question,
      paragraph.questions[].sentences_used (evidence sentence indices)

    Falls back to aps/super_glue multirc (no evidences) for step0 only.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("allenai/multirc", split="train", trust_remote_code=True)
        use_allenai = True
    except Exception:
        print("  [warn] allenai/multirc unavailable — falling back to aps/super_glue multirc "
              "(no evidence annotations; step3 will be skipped for MultiRC)", flush=True)
        ds = load_dataset("aps/super_glue", "multirc", split="train")
        use_allenai = False

    if step == "step3" and not use_allenai:
        print("  [warn] MultiRC step3 skipped: evidence field requires allenai/multirc")
        return

    seen_questions: set[str] = set()
    count = 0

    if use_allenai:
        for item in ds:
            para = item["paragraph"]
            text = para["text"]
            # Split paragraph into sentences for evidence extraction
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            for q in para["questions"]:
                qkey = f"{text[:80]}||{q['question']}"
                if qkey in seen_questions:
                    continue
                seen_questions.add(qkey)
                entry: dict = {"passage": text, "question": q["question"], "source": "multirc"}
                if step == "step3":
                    ev_idxs = q.get("sentences_used", [])
                    if not ev_idxs:
                        continue
                    focus = " ".join(sentences[i] for i in ev_idxs if i < len(sentences))
                    if not focus:
                        continue
                    entry["focus"] = focus
                yield entry
                count += 1
                if limit and count >= limit:
                    return
    else:
        # SuperGLUE version — one record per answer candidate; deduplicate to question level
        for rec in ds:
            qkey = f"{rec['paragraph'][:80]}||{rec['question']}"
            if qkey in seen_questions:
                continue
            seen_questions.add(qkey)
            yield {"passage": rec["paragraph"], "question": rec["question"], "source": "multirc_sg"}
            count += 1
            if limit and count >= limit:
                return


# ── per-step preparation ─────────────────────────────────────────────────────

def _sources_for_step(step: str, limit: int | None) -> list[tuple[str, Iterator[dict]]]:
    if step == "step0":
        return [
            ("RACE-middle",  _iter_race("middle", "EASY",   limit)),
            ("RACE-high",    _iter_race("high",   "MEDIUM", limit)),
            ("RACE-C",       _iter_race_c(limit)),
            ("HotpotQA",     _iter_hotpotqa("step0", limit)),
            ("MultiRC",      _iter_multirc("step0", limit)),
        ]
    if step == "step2":
        return [
            ("RACE-middle → EASY",   _iter_race("middle", "EASY",   limit)),
            ("RACE-high   → MEDIUM", _iter_race("high",   "MEDIUM", limit)),
            ("RACE-C      → HARD",   _iter_race_c(limit)),
        ]
    if step == "step3":
        return [
            ("HotpotQA (comparison)", _iter_hotpotqa("step3", limit)),
            ("MultiRC",               _iter_multirc("step3", limit)),
        ]
    raise ValueError(f"Unknown step: {step}")


def prepare_step(step: str, out_dir: Path, train_r: float, val_r: float, limit: int | None) -> None:
    step_dir = out_dir / step
    step_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for name, iterator in _sources_for_step(step, limit):
        print(f"  Loading {name}...", flush=True)
        n = 0
        for rec in iterator:
            out = {
                "input_text":  _format_input(step, rec["passage"],
                                             difficulty=rec.get("difficulty", ""),
                                             focus=rec.get("focus", "")),
                "target_text": rec["question"],
                "source":      rec.get("source", ""),
                "step":        step,
            }
            if rec.get("difficulty"):
                out["difficulty"] = rec["difficulty"]
            bucket = _split_bucket(rec["passage"], train_r, val_r)
            splits[bucket].append(out)
            n += 1
        print(f"    {n} records", flush=True)

    for split_name, records in splits.items():
        if not records:
            continue
        path = step_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in splits.values())
    print(f"  [{step}] train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])}  total={total}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+",
                        choices=["step0", "step2", "step3"],
                        default=["step0", "step2", "step3"])
    parser.add_argument("--output-dir", default="data/qg")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records per source (smoke tests)")
    args = parser.parse_args()

    if abs(sum(args.split) - 1.0) > 0.01:
        print("Error: split ratios must sum to 1.0")
        sys.exit(1)

    train_r, val_r, _ = args.split
    out_dir = Path(args.output_dir)

    for step in args.steps:
        print(f"\n=== {step.upper()} ===", flush=True)
        prepare_step(step, out_dir, train_r, val_r, args.limit)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
