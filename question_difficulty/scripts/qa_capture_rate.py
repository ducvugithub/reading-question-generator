#!/usr/bin/env python3
"""
QA capture rate: what fraction of real (passage, question, gold answer)
samples does a QA model actually answer correctly -- i.e. how much data
would survive the "captured_correct" gate (token_f1 and recall_overlap
thresholds, with a length guard on the recall_overlap fallback to block
"answered with a big chunk of the passage" from scoring as correct) before
that sample's attention entropy can be trusted as a QG difficulty signal?
Reported per model, per source (RACE-middle/high/C, OneStopQA, SQuAD), and
across a threshold sensitivity grid.

Samples ONCE (shared across all models given via --models, so every model
is scored against the exact same questions -- a fair comparison), then
writes one markdown report per model plus a combined summary comparing
yield across models.

Usage:
  python question_difficulty/scripts/qa_capture_rate.py \
      --models deepset/roberta-base-squad2 deepset/deberta-v3-base-squad2 \
      --n-per-source 50 \
      --output-dir question_difficulty/results/qa_capture_rate
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from question_answering.qa_evaluator import QAEvaluator
from question_answering.qa_model import ExtractiveQAModel

_MC_PATTERNS = ("which of the following", "which one of the following", "which of these")


def _is_mc(question: str) -> bool:
    return any(p in question.lower() for p in _MC_PATTERNS)


def sample_race(subset: str, source_label: str, n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("ehovy/race", subset, split="train")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    idxs = random.sample(range(len(ds)), min(n * 20, len(ds)))
    out = []
    for i in idxs:
        rec = ds[i]
        idx = letter_to_idx.get(rec["answer"])
        if idx is None or idx >= len(rec["options"]) or _is_mc(rec["question"]):
            continue
        out.append({"source": source_label, "passage": rec["article"],
                     "question": rec["question"], "answer": rec["options"][idx]})
        if len(out) >= n:
            break
    return out


def sample_race_c(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("tasksource/race-c", split="train")
    idxs = random.sample(range(len(ds)), min(n * 20, len(ds)))
    out = []
    for i in idxs:
        rec = ds[i]
        if rec["label"] is None or rec["label"] >= len(rec["option"]) or _is_mc(rec["question"]):
            continue
        out.append({"source": "RACE-C", "passage": rec["article"],
                     "question": rec["question"], "answer": rec["option"][rec["label"]]})
        if len(out) >= n:
            break
    return out


def sample_onestopqa(n: int) -> list[dict]:
    """NOTE assumption: answers[0] = correct answer per the STARC annotation
    scheme (a_span points at it), answers[1:] = distractors -- not
    independently re-verified beyond schema inspection."""
    from datasets import load_dataset

    ds = load_dataset("malmaud/onestop_qa", split="train")
    idxs = random.sample(range(len(ds)), min(n * 5, len(ds)))
    out = []
    for i in idxs:
        rec = ds[i]
        if _is_mc(rec["question"]) or not rec["answers"]:
            continue
        out.append({"source": "OneStopQA", "passage": rec["paragraph"],
                     "question": rec["question"], "answer": rec["answers"][0]})
        if len(out) >= n:
            break
    return out


def sample_squad(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="train")
    idxs = random.sample(range(len(ds)), n)
    out = []
    for i in idxs:
        rec = ds[i]
        if not rec["answers"]["text"]:
            continue
        out.append({"source": "SQuAD", "passage": rec["context"],
                     "question": rec["question"], "answer": rec["answers"]["text"][0]})
    return out


def build_samples(n_per_source: int, seed: int) -> list[dict]:
    random.seed(seed)
    samples = []
    samples += sample_race("middle", "RACE-middle", n_per_source)
    samples += sample_race("high", "RACE-high", n_per_source)
    samples += sample_race_c(n_per_source)
    samples += sample_onestopqa(n_per_source)
    samples += sample_squad(n_per_source)
    return samples


def score_samples(samples: list[dict], model_name: str, qa_evaluator: QAEvaluator) -> list[dict]:
    """Returns a NEW list of dicts (samples + per-model prediction/scores) --
    doesn't mutate `samples`, so the same base sample set can be scored by
    multiple models without cross-contamination."""
    qa_model = ExtractiveQAModel(model_name)
    scored = []
    for s in samples:
        pred, conf = qa_model.predict_answer(s["passage"], s["question"])
        f1 = qa_evaluator.token_f1(pred, s["answer"])
        recall_overlap = qa_evaluator.recall_overlap(pred, s["answer"])
        pred_frac = len(pred.split()) / max(1, len(s["passage"].split()))
        scored.append({**s, "pred": pred, "conf": conf, "f1": f1,
                        "recall_overlap": recall_overlap, "pred_frac_of_passage": pred_frac})
    return scored


F1_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
RECALL_THRESHOLDS = [0.5, 0.7, 0.8, 1.0]
LENGTH_GUARD_CAP = 0.30  # recall_overlap pass only counts if pred <= 30% of passage


def _captured(s: dict, f1_thresh: float, ro_thresh: float) -> bool:
    return s["f1"] >= f1_thresh or (s["recall_overlap"] >= ro_thresh and s["pred_frac_of_passage"] <= LENGTH_GUARD_CAP)


def render_model_report(model_name: str, scored: list[dict]) -> str:
    n = len(scored)
    lines = [f"# Threshold Sensitivity: `{model_name}`", "",
             f"n={n} samples ({', '.join(f'{k}={v}' for k, v in sorted(_counts_by_source(scored).items()))})",
             "", "Length guard on `recall_overlap`: prediction must be <= "
             f"{LENGTH_GUARD_CAP*100:.0f}% of the passage's word count.", ""]

    lines += ["## Yield by `token_f1` threshold alone", "",
              "| f1_thresh | n_captured | pct |", "|---|---|---|"]
    for ft in F1_THRESHOLDS:
        c = sum(1 for s in scored if s["f1"] >= ft)
        lines.append(f"| {ft:.2f} | {c} | {100*c/n:.1f}% |")

    lines += ["", "## Yield by `recall_overlap` threshold, with vs. without length guard", "",
              "| ro_thresh | no_guard_pct | with_guard_pct | n_blocked_by_guard |", "|---|---|---|---|"]
    for rt in RECALL_THRESHOLDS:
        passing = [s for s in scored if s["recall_overlap"] >= rt]
        guarded = sum(1 for s in passing if s["pred_frac_of_passage"] <= LENGTH_GUARD_CAP)
        blocked = len(passing) - guarded
        lines.append(f"| {rt:.2f} | {100*len(passing)/n:.1f}% | {100*guarded/n:.1f}% | {blocked} |")

    lines += ["", "## Combined gate: `captured_correct` = (f1 >= f1_thresh) OR "
              "(recall_overlap >= ro_thresh AND length-guarded)", "",
              "| f1_thresh | " + " | ".join(f"ro>={rt:.1f}" for rt in RECALL_THRESHOLDS) + " |",
              "|---|" + "---|" * len(RECALL_THRESHOLDS)]
    for ft in F1_THRESHOLDS:
        row = [f"{100*sum(1 for s in scored if _captured(s, ft, rt))/n:.1f}%" for rt in RECALL_THRESHOLDS]
        lines.append(f"| {ft:.2f} | " + " | ".join(row) + " |")

    lines += ["", "## Yield by source (f1_thresh=0.5, ro_thresh=0.8)", "",
              "| source | n_captured | n_total | pct |", "|---|---|---|---|"]
    by_source = defaultdict(list)
    for s in scored:
        by_source[s["source"]].append(s)
    for src, items in sorted(by_source.items()):
        c = sum(1 for s in items if _captured(s, 0.5, 0.8))
        lines.append(f"| {src} | {c} | {len(items)} | {100*c/len(items):.1f}% |")

    return "\n".join(lines) + "\n"


def render_summary(model_names: list[str], all_scored: dict[str, list[dict]]) -> str:
    lines = ["# Threshold Sensitivity: Model Comparison Summary", "",
              "Combined gate at f1_thresh=0.5, ro_thresh=0.8, length-guard="
              f"{LENGTH_GUARD_CAP*100:.0f}% -- same {len(next(iter(all_scored.values())))} "
              "samples scored by every model.", "",
              "## Overall yield", "", "| model | n_captured | n_total | pct |", "|---|---|---|---|"]
    for name in model_names:
        scored = all_scored[name]
        c = sum(1 for s in scored if _captured(s, 0.5, 0.8))
        lines.append(f"| `{name}` | {c} | {len(scored)} | {100*c/len(scored):.1f}% |")

    lines += ["", "## Yield by source", "",
              "| source | " + " | ".join(f"`{n}`" for n in model_names) + " |",
              "|---|" + "---|" * len(model_names)]
    sources = sorted({s["source"] for s in next(iter(all_scored.values()))})
    for src in sources:
        row = []
        for name in model_names:
            items = [s for s in all_scored[name] if s["source"] == src]
            c = sum(1 for s in items if _captured(s, 0.5, 0.8))
            row.append(f"{100*c/len(items):.1f}%" if items else "-")
        lines.append(f"| {src} | " + " | ".join(row) + " |")

    lines += ["", "## Yield by RACE difficulty level (EASY/MEDIUM/HARD)", "",
              "Same EASY=RACE-middle, MEDIUM=RACE-high, HARD=RACE-C mapping "
              "`validate_difficulty_signals.py` uses. Cells show pct (n_captured/n_total). "
              "OneStopQA has its own separate elementary/intermediate/advanced scale, "
              "not tracked per-level in this script's sampling -- would need "
              "`sample_onestopqa` to sample+tag each level separately to break down "
              "the same way. SQuAD has no difficulty tiers.", "",
              "| level | " + " | ".join(f"`{n}`" for n in model_names) + " |",
              "|---|" + "---|" * len(model_names)]
    for level, src in [("EASY", "RACE-middle"), ("MEDIUM", "RACE-high"), ("HARD", "RACE-C")]:
        row = []
        for name in model_names:
            items = [s for s in all_scored[name] if s["source"] == src]
            c = sum(1 for s in items if _captured(s, 0.5, 0.8))
            row.append(f"{100*c/len(items):.1f}% ({c}/{len(items)})" if items else "-")
        lines.append(f"| {level} | " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def _counts_by_source(samples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s["source"]] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        help="HF model IDs, e.g. deepset/roberta-base-squad2 deepset/deberta-v3-base-squad2")
    parser.add_argument("--n-per-source", type=int, default=50,
                        help="Samples per source (RACE-middle/high/C, OneStopQA, SQuAD)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="question_difficulty/results/qa_capture_rate")
    args = parser.parse_args()

    print(f"Sampling {args.n_per_source} per source (seed={args.seed})...", flush=True)
    samples = build_samples(args.n_per_source, args.seed)
    print(f"Sampled {len(samples)} total: {_counts_by_source(samples)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_evaluator = QAEvaluator()
    all_scored: dict[str, list[dict]] = {}
    for model_name in args.models:
        print(f"\nScoring with {model_name}...", flush=True)
        scored = score_samples(samples, model_name, qa_evaluator)
        all_scored[model_name] = scored

        slug = model_name.replace("/", "__")
        report_path = output_dir / f"{slug}.md"
        report_path.write_text(render_model_report(model_name, scored), encoding="utf-8")
        print(f"  wrote {report_path}", flush=True)

    if len(args.models) > 1:
        summary_path = output_dir / "summary.md"
        summary_path.write_text(render_summary(args.models, all_scored), encoding="utf-8")
        print(f"\nwrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
