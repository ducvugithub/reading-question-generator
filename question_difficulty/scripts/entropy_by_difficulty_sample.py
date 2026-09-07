#!/usr/bin/env python3
"""
Small, human-inspectable sample: N *captured-correct* passages per difficulty
level per dataset, scored by one QA model, with sentence- and token-level
attention entropy (raw + length-normalized). Each passage is grouped with
EVERY question the source dataset actually has for it (not just one), so you
can compare entropy across multiple real questions on the same fixed text.

Unlike a plain random sample, this oversamples each group and scores
candidates on the fly, keeping only passages where at least one of their
questions passes the `captured_correct` gate (f1 >= F1_THRESH OR
(recall_overlap >= RO_THRESH AND length-guarded)), until N_PER_GROUP such
passages are collected (or the candidate pool for that group is exhausted).
The point: entropy is only a valid difficulty signal for questions the model
actually got right, so this is the sample to look at when checking whether
entropy tracks difficulty at all.

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
TOPIC_QUESTION = "What is the main topic of the passage?"
MAX_QUESTIONS_PER_PASSAGE = 5  # keep the report readable for passages with many questions


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


def _locate_sentence(passage: str, span_text: str, sent_spans: list[tuple[int, int]]) -> int | None:
    """Which sentence (by index into sent_spans) contains the first
    occurrence of span_text in passage, or None if not found."""
    idx = passage.find(span_text)
    if idx < 0:
        idx = passage.lower().find(span_text.lower())
    if idx < 0:
        return None
    for i, (s, e) in enumerate(sent_spans):
        if s <= idx < e:
            return i
    return None


def score_passage(cand: dict, qa_model: ExtractiveQAModel, layer: int,
                   qa_evaluator: QAEvaluator) -> dict:
    """Scores every (question, answer) pair attached to this passage, plus
    the fixed topic-probe question once. `any_captured` is True if at least
    one real question passes the captured_correct gate."""
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

        sent_spans = qa_model.sentence_spans(passage)
        ans_idx = _locate_sentence(passage, pred, sent_spans) if pred else None
        if ans_idx is not None and dist["distribution"]:
            ans_share = dist["distribution"][ans_idx]
            ans_rank = 1 + sum(1 for d in dist["distribution"] if d > ans_share)
        else:
            ans_share, ans_rank = None, None

        scored_questions.append({
            "question": question, "answer": answer, "pred": pred, "conf": conf, "f1": f1,
            "recall_overlap": recall_overlap, "pred_frac_of_passage": pred_frac,
            "captured_correct": captured, "sent_entropy": dist["entropy"],
            "sent_entropy_norm": dist["entropy_norm"], "tok_entropy": dist["tok_entropy"],
            "tok_entropy_norm": dist["tok_entropy_norm"], "num_sentences": len(dist["sentences"]),
            "answer_sentence_attention_share": ans_share, "answer_sentence_rank": ans_rank,
        })
        any_captured = any_captured or captured

    pred, conf = qa_model.predict_answer(passage, TOPIC_QUESTION)
    dist = qa_model.get_attention_distribution(passage, TOPIC_QUESTION, layer=layer)
    probe = {"pred": pred, "conf": conf, "sent_entropy": dist["entropy"],
             "sent_entropy_norm": dist["entropy_norm"], "tok_entropy": dist["tok_entropy"],
             "tok_entropy_norm": dist["tok_entropy_norm"]}

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


def _combined(entry: dict) -> float:
    """Simple average of sent_entropy_norm and tok_entropy_norm."""
    return (entry["sent_entropy_norm"] + entry["tok_entropy_norm"]) / 2


def render_report(model_name: str, layer: int, n_per_group: int, max_candidates: int,
                   groups: dict[tuple[str, str], list[dict]],
                   tried_counts: dict[tuple[str, str], int],
                   passage_detail_group: tuple[str, str] = ("RACE-C", "HARD")) -> str:
    header = [f"# Entropy on captured-correct samples only: `{model_name}`", "",
              f"layer={layer}, target {n_per_group} captured_correct passages per group "
              f"(up to {max_candidates} candidates tried per group, up to "
              f"{MAX_QUESTIONS_PER_PASSAGE} questions shown per passage). "
              f"`captured_correct` = f1>={F1_THRESH} OR (recall_overlap>={RO_THRESH} "
              f"AND pred<= {int(LENGTH_GUARD_CAP*100)}% of passage). "
              "A passage is included if at least one of its real questions passes the gate -- "
              "entropy is only a trustworthy difficulty signal on rows where captured=Y. "
              "`combined` is the simple average of sent_entropy_norm and tok_entropy_norm. "
              f"Each passage also gets a fixed topic-probe question (\"{TOPIC_QUESTION}\") for "
              "comparison -- it has no gold answer, so no f1/captured_correct for it.",
              ""]

    order = [("RACE-middle", "EASY"), ("RACE-high", "MEDIUM"), ("RACE-C", "HARD"),
             ("OneStopQA", "EASY"), ("OneStopQA", "MEDIUM"), ("OneStopQA", "HARD"),
             ("SQuAD", "N/A")]

    detail_lines = []
    summary_rows = []
    passage_detail_rows = []
    for key in order:
        items = groups.get(key, [])
        source, level = key
        tried = tried_counts.get(key, 0)
        detail_lines += [f"## {source} / {level}", "",
                          f"{len(items)}/{n_per_group} captured (tried {tried} candidates)", ""]

        group_captured_entries: list[dict] = []
        group_probe_entries: list[dict] = []
        for i, item in enumerate(items, 1):
            captured_questions = [q for q in item["questions"] if q["captured_correct"]]
            detail_lines += [f"### Passage {i}", "", "**Passage:**", "", f"> {item['passage']}", "",
                              f"({len(captured_questions)}/{len(item['questions'])} of this passage's "
                              "questions were captured_correct -- only those are shown below)", ""]
            for j, q in enumerate(captured_questions, 1):
                share = q["answer_sentence_attention_share"]
                rank = q["answer_sentence_rank"]
                loc_str = (f"share={share:.3f}, rank={rank}/{q['num_sentences']}"
                          if share is not None else "answer sentence not located")
                detail_lines += [
                    f"**Question {j}:** {q['question']}", "",
                    f"**Gold answer:** {q['answer']}", "",
                    f"**Predicted answer:** {q['pred']} (f1={q['f1']:.2f}, conf={q['conf']:.3f})", "",
                    f"**Entropy:** sent_entropy_norm={q['sent_entropy_norm']:.3f}, "
                    f"tok_entropy_norm={q['tok_entropy_norm']:.3f}, combined={_combined(q):.3f} "
                    f"(sent_entropy={q['sent_entropy']:.3f}, tok_entropy={q['tok_entropy']:.3f}, "
                    f"num_sentences={q['num_sentences']})",
                    "",
                    f"**Answer-sentence attention:** {loc_str} -- how much attention mass landed on "
                    "the sentence containing the model's own predicted answer, and its rank among "
                    "all sentences (1 = most-attended sentence IS the answer sentence).",
                    "",
                ]
                group_captured_entries.append(q)
            probe = item["probe"]
            detail_lines += [
                f"**Probe question:** {TOPIC_QUESTION}", "",
                f"**Probe predicted answer:** {probe['pred']} (conf={probe['conf']:.3f})", "",
                f"**Probe entropy:** sent_entropy_norm={probe['sent_entropy_norm']:.3f}, "
                f"tok_entropy_norm={probe['tok_entropy_norm']:.3f}, combined={_combined(probe):.3f}",
                "",
            ]
            group_probe_entries.append(probe)

            if captured_questions and key == passage_detail_group:
                p_sent = sum(q["sent_entropy_norm"] for q in captured_questions) / len(captured_questions)
                p_tok = sum(q["tok_entropy_norm"] for q in captured_questions) / len(captured_questions)
                p_combined = sum(_combined(q) for q in captured_questions) / len(captured_questions)
                located = [q for q in captured_questions if q["answer_sentence_attention_share"] is not None]
                p_share = (sum(q["answer_sentence_attention_share"] for q in located) / len(located)
                          if located else None)
                p_rank = sum(q["answer_sentence_rank"] for q in located) / len(located) if located else None
                snippet = item["passage"][:70].replace("|", "/") + "..."
                passage_detail_rows.append((i, snippet, len(captured_questions), len(item["questions"]),
                                            p_sent, p_tok, p_combined, _combined(probe), p_share, p_rank))

        if group_captured_entries:
            avg_sent = sum(q["sent_entropy_norm"] for q in group_captured_entries) / len(group_captured_entries)
            avg_tok = sum(q["tok_entropy_norm"] for q in group_captured_entries) / len(group_captured_entries)
            avg_combined = sum(_combined(q) for q in group_captured_entries) / len(group_captured_entries)
            avg_probe_combined = sum(_combined(p) for p in group_probe_entries) / len(group_probe_entries)
            located = [q for q in group_captured_entries if q["answer_sentence_attention_share"] is not None]
            avg_share = sum(q["answer_sentence_attention_share"] for q in located) / len(located) if located else None
            avg_rank = sum(q["answer_sentence_rank"] for q in located) / len(located) if located else None
            share_str = f"{avg_share:.3f}" if avg_share is not None else "n/a"
            rank_str = f"{avg_rank:.2f}" if avg_rank is not None else "n/a"
            detail_lines.append(f"Avg over {len(group_captured_entries)} captured=Y question(s): "
                                f"sent_entropy_norm={avg_sent:.3f}, tok_entropy_norm={avg_tok:.3f}, "
                                f"combined={avg_combined:.3f} (probe combined avg={avg_probe_combined:.3f}); "
                                f"answer_sentence_attention_share={share_str}, answer_sentence_rank={rank_str} "
                                f"(n_located={len(located)})")
            summary_rows.append((source, level, avg_sent, avg_tok, avg_combined,
                                 avg_probe_combined, avg_share, avg_rank, len(group_captured_entries), len(items)))
        else:
            detail_lines.append("No captured_correct rows found for this group.")
        detail_lines.append("")

    def _fmt(v, spec=".3f"):
        return format(v, spec) if v is not None else "n/a"

    summary_table = ["## Summary: avg entropy by group (captured_correct questions only)", "",
                      "| source | level | sent_entropy_norm | tok_entropy_norm | combined | "
                      "probe combined | ans_sentence_share | ans_sentence_rank | n_questions | n_passages |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
    for source, level, avg_sent, avg_tok, avg_combined, avg_probe_combined, avg_share, avg_rank, n_q, n_p in summary_rows:
        summary_table.append(f"| {source} | {level} | {avg_sent:.3f} | {avg_tok:.3f} | {avg_combined:.3f} | "
                             f"{avg_probe_combined:.3f} | {_fmt(avg_share)} | {_fmt(avg_rank, '.2f')} | "
                             f"{n_q} | {n_p} |")
    summary_table.append("")

    pd_source, pd_level = passage_detail_group
    passage_table = [f"## Per-passage detail: {pd_source} / {pd_level} "
                      "(avg over each passage's captured_correct questions)", "",
                      "`ans_sentence_share` = avg fraction of attention mass on the sentence containing "
                      "the model's own predicted answer; `ans_sentence_rank` = avg rank of that sentence "
                      "by attention mass (1 = most-attended sentence IS the answer sentence -- lower is "
                      "\"more correctly focused\").", "",
                      "| passage | snippet | n_captured/n_total | sent_entropy_norm | "
                      "tok_entropy_norm | combined | probe combined | ans_sentence_share | ans_sentence_rank |",
                      "|---|---|---|---|---|---|---|---|---|"]
    for i, snippet, n_cap, n_tot, p_sent, p_tok, p_combined, p_probe, p_share, p_rank in passage_detail_rows:
        passage_table.append(f"| {i} | {snippet} | {n_cap}/{n_tot} | {p_sent:.3f} | {p_tok:.3f} | "
                             f"{p_combined:.3f} | {p_probe:.3f} | {_fmt(p_share)} | {_fmt(p_rank, '.2f')} |")
    passage_table.append("")

    return "\n".join(header + summary_table + passage_table + detail_lines) + "\n"


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
