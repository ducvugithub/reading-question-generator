#!/usr/bin/env python3
"""
Small, human-inspectable sample: N *captured-correct* passages per difficulty
level per dataset, scored by one QA model, with sentence- and token-level
attention entropy PLUS two entropy alternatives -- participation ratio
(effective outcome count, robust to a noisy attention tail unlike entropy)
and top-K cumulative mass (K=1/2/3 for sentences, K=5/10/15 for tokens, since
tokens are a much finer-grained unit). Each passage is grouped with EVERY
question the source dataset actually has for it (not just one), so you can
compare these metrics across multiple real questions on the same fixed text.

Unlike a plain random sample, this oversamples each group and scores
candidates on the fly, keeping only passages where at least one of their
questions passes the `captured_correct` gate (f1 >= F1_THRESH OR
(recall_overlap >= RO_THRESH AND length-guarded)), until N_PER_GROUP such
passages are collected (or the candidate pool for that group is exhausted).
The point: these metrics are only a valid difficulty signal for questions
the model actually got right -- and even then, the two summary tables are
just a pointer to where to look; the difficulty label is passage-level
(RACE, OneStopQA), not per-question, so manual review of the full per-
question detail below the tables is still required, not optional.

Difficulty levels:
  RACE-middle -> EASY, RACE-high -> MEDIUM, RACE-C -> HARD
    (same convention as validate_difficulty_signals.py / qa_capture_rate.py)
  OneStopQA -> its own Ele/Int/Adv scale, mapped Ele=EASY, Int=MEDIUM, Adv=HARD
  SQuAD -> no difficulty tiers, sampled as a single unlabeled group

Usage:
  python question_difficulty/scripts/entropy_by_difficulty_sample.py \
      --model consciousAI/question-answering-roberta-base-s-v2 \
      --n-per-group 6 \
      --max-candidates-per-group 300
  # writes to question_difficulty/results/entropy_by_difficulty_sample/<model_slug>.md
  # by default (model name with "/" -> "__"); pass --output to override.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from question_answering.qa_evaluator import QAEvaluator
from question_answering.qa_model import ExtractiveQAModel

_MC_PATTERNS = ("which of the following", "which one of the following", "which of these")
LENGTH_GUARD_CAP = 0.30
F1_THRESH = 0.5
RO_THRESH = 0.8
MAX_QUESTIONS_PER_PASSAGE = 5  # keep the report readable for passages with many questions
SENT_TOPKS = (1, 2, 3)      # "top-k sentences" -- a meaningful unit at sentence granularity
TOK_TOPKS = (5, 10, 15)     # tokens are much finer-grained, so use bigger k's
TOPIC_QUESTION = "What is the main topic of the passage?"


def _is_mc(question: str) -> bool:
    return any(p in question.lower() for p in _MC_PATTERNS)


def _group_by_passage(records: Iterable[dict], passage_field: str,
                       get_qa_pair) -> Iterator[dict]:
    """Groups dataset records by passage text, keeping every (question, answer)
    pair `get_qa_pair(rec)` returns (None to skip a record), deduped by
    question text, capped at MAX_QUESTIONS_PER_PASSAGE, then yields
    {"passage": ..., "qa_pairs": [(q, a), ...]} in shuffled passage order."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for rec in records:
        qa = get_qa_pair(rec)
        if qa is None:
            continue
        passage = rec[passage_field]
        groups.setdefault(passage, []).append(qa)
    passages = list(groups.keys())
    random.shuffle(passages)
    for passage in passages:
        seen: set[str] = set()
        qa_pairs = []
        for q, a in groups[passage]:
            if q in seen:
                continue
            seen.add(q)
            qa_pairs.append((q, a))
            if len(qa_pairs) >= MAX_QUESTIONS_PER_PASSAGE:
                break
        yield {"passage": passage, "qa_pairs": qa_pairs}


def iter_race(subset: str, level: str, source_label: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("ehovy/race", subset, split="train")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    def get_qa(rec):
        idx = letter_to_idx.get(rec["answer"])
        if idx is None or idx >= len(rec["options"]) or _is_mc(rec["question"]):
            return None
        return rec["question"], rec["options"][idx]

    for cand in _group_by_passage(ds, "article", get_qa):
        yield {"source": source_label, "level": level, **cand}


def iter_race_c(level: str) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("tasksource/race-c", split="train")

    def get_qa(rec):
        if rec["label"] is None or rec["label"] >= len(rec["option"]) or _is_mc(rec["question"]):
            return None
        return rec["question"], rec["option"][rec["label"]]

    for cand in _group_by_passage(ds, "article", get_qa):
        yield {"source": "RACE-C", "level": level, **cand}


_ONESTOP_LEVEL_TO_TAG = {"Ele": "EASY", "Int": "MEDIUM", "Adv": "HARD"}


def iter_onestopqa(target_tag: str) -> Iterator[dict]:
    """NOTE assumption: answers[0] = correct answer per the STARC annotation
    scheme (a_span points at it), answers[1:] = distractors -- not
    independently re-verified beyond schema inspection."""
    from datasets import load_dataset

    ds = load_dataset("malmaud/onestop_qa", split="train")
    level_names = ds.features["level"].names  # ClassLabel int -> name, e.g. ['Adv', 'Int', 'Ele']

    def get_qa(rec):
        tag = _ONESTOP_LEVEL_TO_TAG.get(level_names[rec["level"]])
        if tag != target_tag or _is_mc(rec["question"]) or not rec["answers"]:
            return None
        return rec["question"], rec["answers"][0]

    records = (rec for rec in ds if _ONESTOP_LEVEL_TO_TAG.get(level_names[rec["level"]]) == target_tag)
    for cand in _group_by_passage(records, "paragraph", get_qa):
        yield {"source": "OneStopQA", "level": target_tag, **cand}


def iter_squad() -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="train")

    def get_qa(rec):
        if not rec["answers"]["text"]:
            return None
        return rec["question"], rec["answers"]["text"][0]

    for cand in _group_by_passage(ds, "context", get_qa):
        yield {"source": "SQuAD", "level": "N/A", **cand}


def _shape_metrics(dist: list[float], qa_model: ExtractiveQAModel, topks: tuple[int, ...],
                    prefix: str) -> dict:
    """Attention-shape metrics beyond entropy, prefixed (e.g. "sent_"/"tok_"):
    top-k cumulative mass for each k in topks, plus participation ratio
    (effective outcome count, noise-floor-robust unlike entropy)."""
    sorted_dist = sorted(dist, reverse=True)
    out = {f"{prefix}top{k}_mass": (sum(sorted_dist[:k]) if sorted_dist else None) for k in topks}
    out[f"{prefix}pr_norm"] = qa_model.normalized_participation_ratio(dist)
    return out


def score_passage(cand: dict, qa_model: ExtractiveQAModel, layer: int,
                   qa_evaluator: QAEvaluator) -> dict:
    """Scores every (question, answer) pair attached to this passage.
    `any_captured` is True if at least one real question passes the
    captured_correct gate."""
    passage = cand["passage"]
    scored_questions = []
    any_captured = False
    for question, answer in cand["qa_pairs"]:
        pred, conf = qa_model.predict_answer(passage, question)
        f1 = qa_evaluator.token_f1(pred, answer)
        recall_overlap = qa_evaluator.recall_overlap(pred, answer)
        pred_frac = len(pred.split()) / max(1, len(passage.split()))
        captured = f1 >= F1_THRESH or (recall_overlap >= RO_THRESH and pred_frac <= LENGTH_GUARD_CAP)
        dist = qa_model.get_attention_distribution(passage, question, layer=layer)

        scored_questions.append({
            "question": question, "answer": answer, "pred": pred, "conf": conf, "f1": f1,
            "recall_overlap": recall_overlap, "pred_frac_of_passage": pred_frac,
            "captured_correct": captured,
            "num_sentences": len(dist["sentences"]), "num_tokens": dist["num_tokens"],
            "sent_entropy_norm": dist["entropy_norm"], "tok_entropy_norm": dist["tok_entropy_norm"],
            **_shape_metrics(dist["distribution"], qa_model, SENT_TOPKS, "sent_"),
            **_shape_metrics(dist["token_distribution"], qa_model, TOK_TOPKS, "tok_"),
        })
        any_captured = any_captured or captured

    probe_pred, probe_conf = qa_model.predict_answer(passage, TOPIC_QUESTION)
    probe_dist = qa_model.get_attention_distribution(passage, TOPIC_QUESTION, layer=layer)
    probe = {
        "question": TOPIC_QUESTION, "pred": probe_pred, "conf": probe_conf,
        "num_sentences": len(probe_dist["sentences"]), "num_tokens": probe_dist["num_tokens"],
        "sent_entropy_norm": probe_dist["entropy_norm"], "tok_entropy_norm": probe_dist["tok_entropy_norm"],
        "sent_pr_norm": qa_model.normalized_participation_ratio(probe_dist["distribution"]),
        "tok_pr_norm": qa_model.normalized_participation_ratio(probe_dist["token_distribution"]),
    }

    return {"source": cand["source"], "level": cand["level"], "passage": passage,
            "questions": scored_questions, "any_captured": any_captured, "probe": probe}


def collect_captured(group_label: str, candidates: Iterator[dict], qa_model: ExtractiveQAModel,
                      layer: int, qa_evaluator: QAEvaluator, n_needed: int,
                      max_candidates: int) -> tuple[list[dict], int]:
    """Scores passage candidates one at a time, keeping only ones where at
    least one question passes captured_correct, until n_needed are collected
    or max_candidates have been tried. Returns (captured_rows, n_candidates_tried)."""
    captured: list[dict] = []
    tried = 0
    for cand in candidates:
        if not cand["qa_pairs"]:
            continue
        tried += 1
        scored = score_passage(cand, qa_model, layer, qa_evaluator)
        if scored["any_captured"]:
            captured.append(scored)
        if len(captured) >= n_needed or tried >= max_candidates:
            break
    print(f"  {group_label}: {len(captured)}/{n_needed} captured after trying {tried} candidates",
          flush=True)
    return captured, tried


def build_groups() -> list[tuple[str, str, Iterator[dict]]]:
    """(source, level, candidate_iterator) for every group, in report order."""
    return [
        ("RACE-middle", "EASY", iter_race("middle", "EASY", "RACE-middle")),
        ("RACE-high", "MEDIUM", iter_race("high", "MEDIUM", "RACE-high")),
        ("RACE-C", "HARD", iter_race_c("HARD")),
        ("OneStopQA", "EASY", iter_onestopqa("EASY")),
        ("OneStopQA", "MEDIUM", iter_onestopqa("MEDIUM")),
        ("OneStopQA", "HARD", iter_onestopqa("HARD")),
        ("SQuAD", "N/A", iter_squad()),
    ]


def _avg(entries: list[dict], key: str) -> float:
    return sum(e[key] for e in entries) / len(entries)


def _esc(s) -> str:
    """Escape "|" so arbitrary text is safe to drop into a markdown table cell."""
    return str(s).replace("|", "/")


def render_report(model_name: str, layer: int, n_per_group: int, max_candidates: int,
                   groups: dict[tuple[str, str], list[dict]],
                   tried_counts: dict[tuple[str, str], int]) -> str:
    header = [f"# Entropy and attention-shape metrics on captured-correct samples: `{model_name}`", "",
              f"layer={layer}, target {n_per_group} captured_correct passages per group "
              f"(up to {max_candidates} candidates tried per group, up to "
              f"{MAX_QUESTIONS_PER_PASSAGE} questions shown per passage). "
              f"`captured_correct` = f1>={F1_THRESH} OR (recall_overlap>={RO_THRESH} "
              f"AND pred<= {int(LENGTH_GUARD_CAP*100)}% of passage). "
              "A passage is included if at least one of its real questions passes the gate -- "
              "these metrics are only a trustworthy difficulty signal on captured_correct=Y rows. "
              "`pr_norm` (participation ratio, normalized) is an alternative to entropy_norm that's "
              "robust to a noisy attention tail -- see the two summary tables below. "
              "`topK_mass` = cumulative attention mass on the K most-attended sentences/tokens. "
              "These are aggregate numbers only, meant to point you at which passages/questions to "
              "read manually below -- they don't replace manual review, since the EASY/MEDIUM/HARD "
              "label is passage-level (RACE, OneStopQA), not per-question.",
              ""]

    order = [("RACE-middle", "EASY"), ("RACE-high", "MEDIUM"), ("RACE-C", "HARD"),
             ("OneStopQA", "EASY"), ("OneStopQA", "MEDIUM"), ("OneStopQA", "HARD"),
             ("SQuAD", "N/A")]

    detail_lines = []
    summary_rows = []
    for key in order:
        items = groups.get(key, [])
        source, level = key
        tried = tried_counts.get(key, 0)
        detail_lines += [f"## {source} / {level}", "",
                          f"{len(items)}/{n_per_group} captured (tried {tried} candidates)", ""]

        group_captured_entries: list[dict] = []
        for i, item in enumerate(items, 1):
            captured_questions = [q for q in item["questions"] if q["captured_correct"]]
            probe = item["probe"]
            detail_lines += [f"### Passage {i}", "", "**Passage:**", "", f"> {item['passage']}", "",
                              f"{probe['num_sentences']} sentences, {probe['num_tokens']} tokens. "
                              f"{len(captured_questions)}/{len(item['questions'])} of this passage's "
                              "questions were captured_correct -- only those are shown below, plus a "
                              "fixed topic-probe question for comparison (no gold answer, so no f1).", ""]
            detail_lines += ["| # | question | gold | pred | f1 | sent_entropy_norm | sent_pr_norm | "
                             "tok_entropy_norm | tok_pr_norm |",
                             "|---|---|---|---|---|---|---|---|---|"]
            for j, q in enumerate(captured_questions, 1):
                detail_lines.append(
                    f"| {j} | {_esc(q['question'])} | {_esc(q['answer'])} | {_esc(q['pred'])} | "
                    f"{q['f1']:.2f} | {q['sent_entropy_norm']:.3f} | {q['sent_pr_norm']:.3f} | "
                    f"{q['tok_entropy_norm']:.3f} | {q['tok_pr_norm']:.3f} |")
            detail_lines.append(
                f"| probe | {_esc(probe['question'])} | -- | {_esc(probe['pred'])} | -- | "
                f"{probe['sent_entropy_norm']:.3f} | {probe['sent_pr_norm']:.3f} | "
                f"{probe['tok_entropy_norm']:.3f} | {probe['tok_pr_norm']:.3f} |")
            detail_lines.append("")
            group_captured_entries.extend(captured_questions)

        if group_captured_entries:
            e = group_captured_entries
            detail_lines.append(
                f"Avg over {len(e)} captured=Y question(s) -- "
                f"sentence: entropy_norm={_avg(e,'sent_entropy_norm'):.3f}, pr_norm={_avg(e,'sent_pr_norm'):.3f}, "
                f"top1={_avg(e,'sent_top1_mass'):.3f}, top2={_avg(e,'sent_top2_mass'):.3f}, "
                f"top3={_avg(e,'sent_top3_mass'):.3f}; token: entropy_norm={_avg(e,'tok_entropy_norm'):.3f}, "
                f"pr_norm={_avg(e,'tok_pr_norm'):.3f}, top5={_avg(e,'tok_top5_mass'):.3f}, "
                f"top10={_avg(e,'tok_top10_mass'):.3f}, top15={_avg(e,'tok_top15_mass'):.3f}")
            summary_rows.append((source, level, e, len(items)))
        else:
            detail_lines.append("No captured_correct rows found for this group.")
        detail_lines.append("")

    sent_table = ["## Summary (sentence-level)", "",
                  "| source | level | entropy_norm | pr_norm | top1_mass | top2_mass | top3_mass | "
                  "n_questions | n_passages |",
                  "|---|---|---|---|---|---|---|---|---|"]
    for source, level, e, n_p in summary_rows:
        sent_table.append(f"| {source} | {level} | {_avg(e,'sent_entropy_norm'):.3f} | "
                          f"{_avg(e,'sent_pr_norm'):.3f} | {_avg(e,'sent_top1_mass'):.3f} | "
                          f"{_avg(e,'sent_top2_mass'):.3f} | {_avg(e,'sent_top3_mass'):.3f} | "
                          f"{len(e)} | {n_p} |")
    sent_table.append("")

    tok_table = ["## Summary (token-level)", "",
                 "| source | level | entropy_norm | pr_norm | top5_mass | top10_mass | top15_mass | "
                 "n_questions | n_passages |",
                 "|---|---|---|---|---|---|---|---|---|"]
    for source, level, e, n_p in summary_rows:
        tok_table.append(f"| {source} | {level} | {_avg(e,'tok_entropy_norm'):.3f} | "
                         f"{_avg(e,'tok_pr_norm'):.3f} | {_avg(e,'tok_top5_mass'):.3f} | "
                         f"{_avg(e,'tok_top10_mass'):.3f} | {_avg(e,'tok_top15_mass'):.3f} | "
                         f"{len(e)} | {n_p} |")
    tok_table.append("")

    return "\n".join(header + sent_table + tok_table + detail_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="consciousAI/question-answering-roberta-base-s-v2")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--n-per-group", type=int, default=6)
    ap.add_argument("--max-candidates-per-group", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None,
                     help="defaults to question_difficulty/results/entropy_by_difficulty_sample/<model_slug>.md")
    args = ap.parse_args()

    if args.output is None:
        slug = args.model.replace("/", "__")
        args.output = f"question_difficulty/results/entropy_by_difficulty_sample/{slug}.md"

    random.seed(args.seed)

    print(f"Loading QA model ({args.model})...", flush=True)
    qa_model = ExtractiveQAModel(args.model)
    qa_evaluator = QAEvaluator()

    print(f"Collecting {args.n_per_group} captured_correct passages per group "
          f"(seed={args.seed}, max {args.max_candidates_per_group} candidates/group)...", flush=True)
    groups: dict[tuple[str, str], list[dict]] = {}
    tried_counts: dict[tuple[str, str], int] = {}
    for source, level, candidates in build_groups():
        key = (source, level)
        captured, tried = collect_captured(f"{source}/{level}", candidates, qa_model, args.layer,
                                            qa_evaluator, args.n_per_group, args.max_candidates_per_group)
        groups[key] = captured
        tried_counts[key] = tried

    report = render_report(args.model, args.layer, args.n_per_group, args.max_candidates_per_group,
                            groups, tried_counts)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
