#!/usr/bin/env python3
"""
EDA on all datasets used in the QDE and QG pipelines.

Datasets analysed:
  RACE-middle   ehovy/race middle         — QDE EASY
  RACE-high     ehovy/race high           — QDE MEDIUM
  RACE-C        tasksource/race-c         — QDE HARD
  HotpotQA      hotpotqa/hotpot_qa        — focus-span QG (Step 3/4)
  MultiRC       aps/super_glue multirc    — focus-span QG (Step 3/4)

Usage:
  python scripts/eda_datasets.py
  python scripts/eda_datasets.py --sample 5000
  python scripts/eda_datasets.py --plot --out-dir data/eda
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _pct(arr, ps=(5, 25, 50, 75, 95)):
    a = np.array(arr, dtype=float)
    return {p: float(np.percentile(a, p)) for p in ps}


def _row(label, arr, fmt=".0f"):
    p = _pct(arr)
    f = fmt
    print(f"    {label:<24}  "
          f"p5={p[5]:{f}}  p25={p[25]:{f}}  "
          f"median={p[50]:{f}}  p75={p[75]:{f}}  "
          f"p95={p[95]:{f}}  mean={np.mean(arr):{f}}")


def _wc(text):
    return len(text.split())


def _sc(text):
    return max(1, len(re.split(r'(?<=[.!?])\s+', text.strip())))


def _qw(q):
    t = q.strip().lower()
    for kw in ['why', 'how many', 'how much', 'how', 'which', 'what', 'who', 'when', 'where']:
        if t.startswith(kw):
            return kw
    if re.match(r'^(is|are|was|were|do|does|did|can|could|will|would)\b', t):
        return 'yes/no'
    return 'other'


def _header(title):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")


# ── per-dataset EDA ───────────────────────────────────────────────────────────

def eda_race_ehovy(subset: str, difficulty: str, sample: int):
    from datasets import load_dataset
    _header(f"RACE-{subset}  (ehovy/race / {subset})  —  QDE label: {difficulty}")

    ds = load_dataset("ehovy/race", subset, split="train")
    n_total = len(ds)
    recs = list(ds.select(range(min(sample, n_total))))

    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    passages  = [r["article"] for r in recs]
    questions = [r["question"] for r in recs]
    answers   = [r["options"][letter_to_idx[r["answer"]]]
                 for r in recs if letter_to_idx.get(r["answer"]) is not None
                 and letter_to_idx[r["answer"]] < len(r["options"])]
    options   = [r["options"] for r in recs]

    a_in_p = sum(1 for p, a in zip(passages, answers) if a.lower() in p.lower())

    print(f"  Size (train): {n_total:,}   sampled: {len(recs):,}")
    print(f"  Focus span  : NO")
    print(f"  Difficulty  : YES — fixed label '{difficulty}'")
    print(f"  Answer type : multiple-choice (4 options, non-span)")
    print(f"  a_in_passage: {a_in_p}/{len(answers)} ({100*a_in_p/len(answers):.1f}%)")
    ans_dist = Counter(r["answer"] for r in recs)
    print(f"  MC answer distribution: " + "  ".join(f"{k}:{v}" for k, v in sorted(ans_dist.items())))

    print("\n  Passage")
    _row("char length",     [len(p) for p in passages])
    _row("word count",      [_wc(p) for p in passages])
    _row("sentence count",  [_sc(p) for p in passages])

    print("\n  Question")
    _row("char length",  [len(q) for q in questions])
    _row("word count",   [_wc(q) for q in questions])
    qw = Counter(_qw(q) for q in questions)
    print(f"    {'question word':<24}  " + "  ".join(f"{w}:{c}" for w, c in qw.most_common(8)))

    print("\n  Answer options")
    _row("option word count", [_wc(o) for opts in options for o in opts])
    _row("correct ans wc",   [_wc(a) for a in answers])


def eda_race_c(sample: int):
    from datasets import load_dataset
    _header("RACE-C  (tasksource/race-c)  —  QDE label: HARD")

    ds = load_dataset("tasksource/race-c", split="train")
    n_total = len(ds)
    recs = list(ds.select(range(min(sample, n_total))))

    passages  = [r["article"] for r in recs]
    questions = [r["question"] for r in recs]
    answers   = [r["option"][r["label"]] for r in recs if r["label"] < len(r["option"])]
    options   = [r["option"] for r in recs]

    a_in_p = sum(1 for p, a in zip(passages, answers) if a.lower() in p.lower())

    print(f"  Size (train): {n_total:,}   sampled: {len(recs):,}")
    print(f"  Focus span  : NO")
    print(f"  Difficulty  : YES — fixed label 'HARD'")
    print(f"  Answer type : multiple-choice (4 options, non-span)")
    print(f"  a_in_passage: {a_in_p}/{len(answers)} ({100*a_in_p/len(answers):.1f}%)")
    ans_dist = Counter(r["label"] for r in recs)
    print(f"  MC answer distribution (0-3): " + "  ".join(f"{k}:{v}" for k, v in sorted(ans_dist.items())))

    print("\n  Passage")
    _row("char length",     [len(p) for p in passages])
    _row("word count",      [_wc(p) for p in passages])
    _row("sentence count",  [_sc(p) for p in passages])

    print("\n  Question")
    _row("char length",  [len(q) for q in questions])
    _row("word count",   [_wc(q) for q in questions])
    qw = Counter(_qw(q) for q in questions)
    print(f"    {'question word':<24}  " + "  ".join(f"{w}:{c}" for w, c in qw.most_common(8)))

    print("\n  Answer options")
    _row("option word count", [_wc(o) for opts in options for o in opts])
    _row("correct ans wc",   [_wc(a) for a in answers])


def eda_hotpotqa(sample: int):
    from datasets import load_dataset
    _header("HotpotQA  (hotpotqa/hotpot_qa distractor)  —  focus-span QG")

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    n_total = len(ds)
    recs = list(ds.select(range(min(sample, n_total))))

    def _gold_passage(r):
        gold = set(r["supporting_facts"]["title"])
        return " ".join(
            " ".join(sents)
            for title, sents in zip(r["context"]["title"], r["context"]["sentences"])
            if title in gold
        )

    def _focus_span(r):
        sf_map: dict[str, set] = {}
        for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
            sf_map.setdefault(t, set()).add(i)
        parts = []
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            if title in sf_map:
                for i, s in enumerate(sents):
                    if i in sf_map[title]:
                        parts.append(s)
        return " ".join(parts)

    passages   = [_gold_passage(r) for r in recs]
    full_ctx   = [" ".join(" ".join(s) for s in r["context"]["sentences"]) for r in recs]
    questions  = [r["question"] for r in recs]
    answers    = [r["answer"] for r in recs]
    spans      = [_focus_span(r) for r in recs]
    types      = Counter(r["type"] for r in recs)

    yesno  = sum(1 for a in answers if a.lower() in ("yes", "no"))
    n_sf   = [len(set(r["supporting_facts"]["title"])) for r in recs]
    n_sf_s = [len(r["supporting_facts"]["title"]) for r in recs]

    span_wc   = [_wc(s) for s in spans if s]
    pass_wc   = [_wc(p) for p in passages if p]
    span_frac = [sw / pw if pw > 0 else 0 for sw, pw in zip(span_wc, pass_wc)]

    print(f"  Size (train): {n_total:,}   sampled: {len(recs):,}")
    print(f"  Focus span  : YES — supporting_facts (sentence-level, gold passages only)")
    print(f"  Difficulty  : NO explicit labels")
    print(f"  Answer type : yes/no={yesno} ({100*yesno/len(recs):.1f}%)  "
          f"span={len(recs)-yesno} ({100*(len(recs)-yesno)/len(recs):.1f}%)")
    print(f"  Q type      : " + "  ".join(f"{k}:{v}" for k, v in types.most_common()))

    print("\n  Gold passage (2 supporting passages concatenated)")
    _row("char length",    [len(p) for p in passages if p])
    _row("word count",     [_wc(p) for p in passages if p])
    _row("sentence count", [_sc(p) for p in passages if p])

    print("\n  Full context (all 10 passages)")
    _row("word count", [_wc(c) for c in full_ctx])

    print("\n  Question")
    _row("char length", [len(q) for q in questions])
    _row("word count",  [_wc(q) for q in questions])
    qw = Counter(_qw(q) for q in questions)
    print(f"    {'question word':<24}  " + "  ".join(f"{w}:{c}" for w, c in qw.most_common(8)))

    print("\n  Answer")
    _row("answer word count", [_wc(a) for a in answers])

    print("\n  Focus span (supporting_facts sentences)")
    _row("span word count",          span_wc)
    _row("span % of gold passage",   [int(f*100) for f in span_frac])
    print(f"    {'num gold passages':<24}  min={min(n_sf)}  max={max(n_sf)}  "
          f"mean={np.mean(n_sf):.2f}  (always 2)")
    print(f"    {'num supporting sents':<24}  min={min(n_sf_s)}  max={max(n_sf_s)}  "
          f"mean={np.mean(n_sf_s):.2f}")


def eda_multirc(sample: int):
    from datasets import load_dataset
    _header("MultiRC  (aps/super_glue multirc)  —  focus-span QG")

    ds = load_dataset("aps/super_glue", "multirc", split="train")
    n_total = len(ds)
    recs = list(ds.select(range(min(sample, n_total))))

    # Each record = one answer candidate for one question over one paragraph
    passages   = {r["idx"]["paragraph"]: r["paragraph"] for r in recs}
    questions  = [r["question"] for r in recs]
    ans_texts  = [r["answer"] for r in recs]
    labels     = Counter(r["label"] for r in recs)

    # Check what fields exist
    sample_fields = list(recs[0].keys())

    unique_pids = len(passages)
    unique_qids = len(set((r["idx"]["paragraph"], r["idx"]["question"]) for r in recs))

    p_list = list(passages.values())
    print(f"  Size (train records): {n_total:,}   sampled: {len(recs):,}")
    print(f"  Unique paragraphs   : {unique_pids}")
    print(f"  Unique questions    : {unique_qids}")
    print(f"  Fields available    : {sample_fields}")
    print(f"  Focus span          : YES — 'evidences' field (sentence-level)")
    print(f"  Difficulty          : NO explicit labels")
    print(f"  Answer type         : binary per candidate")
    print(f"  Label distribution  : true={labels[1]} ({100*labels[1]/len(recs):.1f}%)  "
          f"false={labels[0]} ({100*labels[0]/len(recs):.1f}%)")

    print("\n  Paragraph")
    _row("char length",    [len(p) for p in p_list])
    _row("word count",     [_wc(p) for p in p_list])
    _row("sentence count", [_sc(p) for p in p_list])

    print("\n  Question")
    _row("char length", [len(q) for q in questions])
    _row("word count",  [_wc(q) for q in questions])
    qw = Counter(_qw(q) for q in questions)
    print(f"    {'question word':<24}  " + "  ".join(f"{w}:{c}" for w, c in qw.most_common(8)))

    print("\n  Answer candidates")
    _row("char length", [len(a) for a in ans_texts])
    _row("word count",  [_wc(a) for a in ans_texts])

    # Check evidences field if available
    if "evidences" in sample_fields:
        ev_counts = [len(r["evidences"]) for r in recs]
        print("\n  Evidences (focus span sentences per question)")
        _row("num evidence sents", ev_counts)
    else:
        print("\n  Note: 'evidences' field not in super_glue split — "
              "use original MultiRC from allenai/multirc for evidence annotations.")


# ── summary table ─────────────────────────────────────────────────────────────

def summary():
    print(f"\n{'═'*72}")
    print("  SUMMARY: Dataset capabilities at a glance")
    print(f"{'─'*72}")
    rows = [
        ("Dataset",       "Size(train)", "Focus span",          "Difficulty",           "Answer type"),
        ("RACE-middle",   "~25K",        "NO",                  "YES (EASY label)",     "MC non-span"),
        ("RACE-high",     "~62K",        "NO",                  "YES (MEDIUM label)",   "MC non-span"),
        ("RACE-C",        "~12.7K",      "NO",                  "YES (HARD label)",     "MC non-span"),
        ("HotpotQA",      "~90K",        "YES (sentence-level)","NO",                   "yes/no + span"),
        ("MultiRC",       "~27K Q",      "YES (sentence-level)","NO",                   "binary/candidate"),
    ]
    col_w = [14, 13, 25, 24, 18]
    for row in rows:
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
    print()
    print("  Pipeline role:")
    print("    QDE (Step 1)        : RACE-middle + RACE-high + RACE-C")
    print("    Baseline QG (Step 0): RACE-middle + RACE-high + RACE-C + HotpotQA + MultiRC (no conditioning)")
    print("    Difficulty QG (Step 2): RACE-middle + RACE-high + RACE-C (with difficulty token)")
    print("    Focus-span QG (Step 3): HotpotQA (comparison) + MultiRC (with focus span)")
    print("    Full M6 (Step 4)    : Steps 2+3 data, enriched via QDE")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=5000,
                        help="Records to sample per dataset (default 5000)")
    parser.add_argument("--datasets", nargs="+",
                        choices=["race-middle", "race-high", "race-c", "hotpotqa", "multirc", "all"],
                        default=["all"])
    args = parser.parse_args()

    run_all = "all" in args.datasets
    s = args.sample

    if run_all or "race-middle" in args.datasets:
        eda_race_ehovy("middle", "EASY",   s)
    if run_all or "race-high" in args.datasets:
        eda_race_ehovy("high",   "MEDIUM", s)
    if run_all or "race-c" in args.datasets:
        eda_race_c(s)
    if run_all or "hotpotqa" in args.datasets:
        eda_hotpotqa(s)
    if run_all or "multirc" in args.datasets:
        eda_multirc(s)

    summary()


if __name__ == "__main__":
    main()
