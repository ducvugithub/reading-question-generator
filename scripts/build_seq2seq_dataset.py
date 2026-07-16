#!/usr/bin/env python3
"""
Build seq2seq training data for question generation from SQuAD 2.0, Natural Questions, and TyDiQA.

Pipeline per sample:
  1. Load (passage, question) from source dataset.
  2. Optionally filter to short passages (--max-passage-length).
  3. Run KG extractor on the passage → triples.
  4. Linearize all triples (up to --max-triples) with a CEFR control prefix.
  5. Assign CEFR level via RuleBasedEstimator (passage readability).

The model learns: given KG triples of a passage → generate a question.
This matches inference time where you extract the passage KG and ask for questions at a target level.

Output JSONL format:
  {"input": "generate question level=B1: Beyoncé | born_in | Houston . Beyoncé | performed_with | Destiny's Child",
   "target": "When did Beyoncé begin her solo career?",
   "cefr": "B1", "lang": "en", "source": "squad"}

Split: 80% train / 20% eval at passage level (same passage never in both splits).

Usage:
  python scripts/build_seq2seq_dataset.py --sources squad tydiqa --output-dir data/training
  python scripts/build_seq2seq_dataset.py --sources nq --limit 5000 --max-passage-length 600
  python scripts/build_seq2seq_dataset.py --sources squad_fi tydiqa --lang fi --output-dir data/training
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_graph.extractor import KnowledgeGraphExtractor, Triple
from question_generation.difficulty import RuleBasedEstimator
from question_generation.methods.seq2seq.linearizer import linearize


# ---------------------------------------------------------------------------
# Source loaders — each yields (passage_id, context, question, answer, answer_start, lang)
# ---------------------------------------------------------------------------

def _squad_samples(
    limit: int | None, lang_filter: str | None
) -> Iterator[tuple[str, str, str, str, int, str]]:
    """SQuAD 2.0 — English only. Skip unanswerable questions."""
    if lang_filter and lang_filter != "en":
        return
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
    count = 0
    for s in ds:
        if not s["answers"]["text"]:
            continue  # unanswerable
        yield (
            s["id"], s["context"], s["question"],
            s["answers"]["text"][0], s["answers"]["answer_start"][0], "en",
        )
        count += 1
        if limit and count >= limit:
            break


def _tydiqa_samples(
    limit: int | None, lang_filter: str | None
) -> Iterator[tuple[str, str, str, str, int, str]]:
    """TyDiQA GoldP secondary task — Finnish and English samples."""
    from datasets import load_dataset

    _LANG_MAP = {"english": "en", "finnish": "fi"}
    ds = load_dataset(
        "google-research-datasets/tydiqa", "secondary_task", split="train", streaming=True
    )
    count = 0
    for s in ds:
        if not s["answers"]["text"]:
            continue
        raw_lang = s["id"].split("--")[0]  # e.g. "finnish--7633..."
        lang = _LANG_MAP.get(raw_lang, raw_lang[:2])
        if lang_filter and lang != lang_filter:
            continue
        yield (
            s["id"], s["context"], s["question"],
            s["answers"]["text"][0], s["answers"]["answer_start"][0], lang,
        )
        count += 1
        if limit and count >= limit:
            break


def _nq_samples(
    limit: int | None, lang_filter: str | None
) -> Iterator[tuple[str, str, str, str, int, str]]:
    """Natural Questions — English only.

    Reconstructs passage from the long-answer token span (filtering HTML tokens).
    Uses the first annotator that provides both a long answer and a short answer.
    """
    if lang_filter and lang_filter != "en":
        return
    from datasets import load_dataset

    ds = load_dataset(
        "google-research-datasets/natural_questions", split="train", streaming=True
    )
    count = 0
    for s in ds:
        ann = s["annotations"]
        cand_idx = -1
        short_text = None

        for la, sa in zip(ann["long_answer"], ann["short_answers"]):
            if sa["text"] and la["candidate_index"] != -1:
                cand_idx = la["candidate_index"]
                short_text = sa["text"][0]
                break

        if not short_text or cand_idx == -1:
            continue

        candidates = s["long_answer_candidates"]
        if cand_idx >= len(candidates):
            continue

        cand = candidates[cand_idx]
        start, end = cand["start_token"], cand["end_token"]
        doc_tokens = s["document"]["tokens"]
        passage_tokens = [
            doc_tokens["token"][i]
            for i in range(start, end)
            if not doc_tokens["html_token"][i]
        ]
        if not passage_tokens:
            continue

        passage = " ".join(passage_tokens)
        answer_start = passage.find(short_text)
        yield str(s["id"]), passage, s["question"]["text"], short_text, max(answer_start, 0), "en"
        count += 1
        if limit and count >= limit:
            break


def _squad_fi_samples(
    limit: int | None, lang_filter: str | None
) -> Iterator[tuple[str, str, str, str, int, str]]:
    """SQuAD v2 machine-translated to Finnish. Same format as SQuAD v2."""
    if lang_filter and lang_filter != "fi":
        return
    from datasets import load_dataset

    ds = load_dataset("ilmariky/SQuAD_v2_fi", split="train", streaming=True)
    count = 0
    for s in ds:
        if not s["answers"]["text"]:
            continue  # unanswerable
        yield (
            s["id"], s["context"], s["question"],
            s["answers"]["text"][0], s["answers"]["answer_start"][0], "fi",
        )
        count += 1
        if limit and count >= limit:
            break


_LOADERS = {
    "squad": _squad_samples,
    "squad_fi": _squad_fi_samples,
    "tydiqa": _tydiqa_samples,
    "nq": _nq_samples,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentences_around(context: str, answer_start: int, window: int = 2) -> str:
    """Return ±window sentences around the answer character offset."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", context)
    pos = 0
    target_idx = 0
    for i, sent in enumerate(sentences):
        if pos + len(sent) >= answer_start:
            target_idx = i
            break
        pos += len(sent) + 1
    start = max(0, target_idx - window)
    end = min(len(sentences), target_idx + window + 1)
    return " ".join(sentences[start:end])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    estimator = RuleBasedEstimator()
    extractors: dict[str, KnowledgeGraphExtractor] = {}

    # Collect records per language: {lang: [(passage_id, record)]}
    by_lang: dict[str, list[tuple[str, dict]]] = defaultdict(list)

    total_in = total_filtered_len = total_no_triples = total_ok = 0

    for source in args.sources:
        loader = _LOADERS[source]
        print(f"[{source}] loading...", flush=True)
        source_count = 0

        for passage_id, context, question, answer, answer_start, lang in loader(args.limit, args.lang):
            total_in += 1

            if args.max_passage_length and len(context) > args.max_passage_length:
                total_filtered_len += 1
                continue

            if lang not in extractors:
                print(f"  initialising KG extractor for lang={lang}...", flush=True)
                extractors[lang] = KnowledgeGraphExtractor(lang=lang, coref=args.coref)

            triples = extractors[lang].extract(context)
            if len(triples) < args.min_triples:
                total_no_triples += 1
                continue

            # Limit triples to cap model input length
            selected_triples = triples[: args.max_triples]

            s_read = estimator.score_readability(context)
            s_type = estimator.score_type("object")
            cefr = estimator.estimate(s_type, 0.2, 0.0, s_read)

            record = {
                "input": linearize(selected_triples, cefr),
                "target": question,
                "cefr": cefr,
                "lang": lang,
                "source": source,
                "coref_resolved": args.coref,
            }

            if args.verbose:
                window = _sentences_around(context, answer_start)
                ans_lower = answer.lower()
                print(f"\n{'─'*60}")
                print(f"PASSAGE ({len(context)} chars, showing ±2 sentences around answer):")
                print(f"  {window}")
                print(f"\nANSWER: {answer!r}")
                print(f"\nFULL KG ({len(triples)} triples, using first {len(selected_triples)}):")
                for t in selected_triples:
                    subj_l = t.subject.lower()
                    obj_l = t.object.lower()
                    subj_covers = subj_l in ans_lower or ans_lower in subj_l
                    obj_covers = obj_l in ans_lower or ans_lower in obj_l
                    if subj_covers or obj_covers:
                        marker = " ◄ answer"
                    else:
                        marker = ""
                    print(f"  {t.subject!r:30s} | {t.relation:25s} | {t.object!r}{marker}")
                print(f"\nCEFR: {cefr}  (readability={s_read:.2f})")
                print(f"TARGET: {question}")

            by_lang[lang].append((passage_id, record))
            total_ok += 1
            source_count += 1

            if not args.verbose and source_count % 500 == 0:
                print(f"  [{source}] processed {source_count} samples", flush=True)

        print(f"[{source}] done: {source_count} records kept", flush=True)

    print(
        f"\nSummary: {total_in} in → "
        f"{total_filtered_len} too-long, {total_no_triples} too-few-triples, "
        f"{total_ok} written"
    )

    # Passage-level 80/20 split per language
    for lang, items in by_lang.items():
        passage_ids = list({pid for pid, _ in items})
        random.shuffle(passage_ids)
        split = int(len(passage_ids) * 0.8)
        train_ids = set(passage_ids[:split])

        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        train_path = lang_dir / "train.jsonl"
        eval_path = lang_dir / "eval.jsonl"

        n_train = n_eval = 0
        with open(train_path, "w", encoding="utf-8") as ft, \
             open(eval_path, "w", encoding="utf-8") as fe:
            for pid, record in items:
                if pid in train_ids:
                    ft.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_train += 1
                else:
                    fe.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_eval += 1

        print(f"[{lang}] → {train_path} ({n_train} train) | {eval_path} ({n_eval} eval)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build seq2seq QG training data from SQuAD, NQ, and TyDiQA."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(_LOADERS),
        default=["squad", "tydiqa"],
        help="Datasets to load (default: squad tydiqa).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training",
        help="Root output directory (default: data/training).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "fi"],
        default=None,
        help="Restrict to one language (default: all languages in each source).",
    )
    parser.add_argument(
        "--max-passage-length",
        type=int,
        default=None,
        metavar="CHARS",
        help="Skip passages longer than CHARS characters (e.g. 600).",
    )
    parser.add_argument(
        "--min-triples",
        type=int,
        default=2,
        metavar="N",
        help="Minimum KG triples required to keep a sample (default: 2).",
    )
    parser.add_argument(
        "--max-triples",
        type=int,
        default=15,
        metavar="N",
        help="Max triples included in model input (default: 15, ~150 tokens).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max samples per source (useful for testing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split (default: 42).",
    )
    parser.add_argument(
        "--coref",
        action="store_true",
        help=(
            "Enable Stanza coreference resolution (English only). "
            "Replaces pronoun subjects/objects with their antecedent. "
            "Requires: stanza.download('en', processors='coref'). "
            "Use this flag to build the coref-resolved ablation dataset."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print passage, full KG, and CEFR for each sample.",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    build(args)


if __name__ == "__main__":
    main()
