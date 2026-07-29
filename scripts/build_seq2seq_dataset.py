#!/usr/bin/env python3
"""
Build unified QG training data from SQuAD 2.0 and TyDiQA.

Each output record contains:
  - passage, answer, question  (for seq2seq baseline)
  - kg_raw     KG triples extracted without coreference resolution
  - kg_coref   KG triples with Stanza coref (English only; null for other languages)
  - source, lang, cefr

Training scripts pick the fields they need. Coref ablation is just a field
switch — no separate dataset build required.

Output JSONL:
  {"passage": "...", "answer": "...", "question": "...",
   "kg_raw": [["Beyoncé", "born_in", "Houston"], ...],
   "kg_coref": [["Beyoncé", "perform_with", "Destiny's Child"], ...],
   "source": "squad", "lang": "en", "cefr": "B1"}

Split: 80% train / 20% eval at passage level.

Usage:
  python scripts/build_seq2seq_dataset.py --sources squad tydiqa --output-dir data/training
  python scripts/build_seq2seq_dataset.py --sources squad --limit 500 --verbose
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_graph.extractor import KnowledgeGraphExtractor, Triple
from question_generation.difficulty import RuleBasedEstimator


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

    ds = load_dataset("rajpurkar/squad_v2", split="train")
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
        "google-research-datasets/tydiqa", "secondary_task", split="train"
    )
    count = 0
    for s in ds:
        if not s["answers"]["text"]:
            continue
        raw_lang = s["id"].split("--")[0]
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
    """Natural Questions — English only."""
    if lang_filter and lang_filter != "en":
        return
    from datasets import load_dataset

    ds = load_dataset(
        "google-research-datasets/natural_questions", split="train"
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
    """SQuAD v2 machine-translated to Finnish."""
    if lang_filter and lang_filter != "fi":
        return
    from datasets import load_dataset

    ds = load_dataset("ilmariky/SQuAD_v2_fi", split="train")
    count = 0
    for s in ds:
        if not s["answers"]["text"]:
            continue
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


def _triples_to_list(triples: list[Triple]) -> list[list[str]]:
    return [[t.subject, t.relation, t.object] for t in triples]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _is_train(passage_id: str) -> bool:
    """Deterministic 80/20 passage-level split via MD5 hash — no in-memory accumulation."""
    return int(hashlib.md5(passage_id.encode()).hexdigest(), 16) % 10 < 8


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    estimator = RuleBasedEstimator()
    extractors: dict[str, KnowledgeGraphExtractor] = {}

    # Stream records directly to disk — one (train, eval) file pair per language
    out_files: dict[str, tuple] = {}  # lang → (train_fh, eval_fh)
    counts: dict[str, list[int]] = {}  # lang → [n_train, n_eval]

    total_in = total_filtered_len = total_no_triples = total_ok = total_failed_kg = 0

    try:
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
                    use_coref = (lang == "en") and not args.no_coref
                    extractors[lang] = KnowledgeGraphExtractor(lang=lang, coref=use_coref)

                if lang not in out_files:
                    lang_dir = output_dir / lang
                    lang_dir.mkdir(parents=True, exist_ok=True)
                    out_files[lang] = (
                        open(lang_dir / "train.jsonl", "w", encoding="utf-8"),
                        open(lang_dir / "eval.jsonl",  "w", encoding="utf-8"),
                    )
                    counts[lang] = [0, 0]

                try:
                    triples_raw, triples_coref = extractors[lang].extract_both(context)
                except Exception as exc:
                    triples_raw, triples_coref = None, None
                    total_failed_kg += 1
                    print(f"  [kg-fail] {exc.__class__.__name__}: {exc}", flush=True)

                if triples_raw is not None and len(triples_raw) < args.min_triples:
                    total_no_triples += 1
                    continue

                raw   = triples_raw[:args.max_triples]   if triples_raw   is not None else None
                coref = triples_coref[:args.max_triples] if (triples_coref is not None and not args.no_coref) else None

                s_read = estimator.score_readability(context)
                s_type = estimator.score_type("object")
                cefr   = estimator.estimate(s_type, 0.2, 0.0, s_read)

                record = {
                    "passage":  context,
                    "answer":   answer,
                    "question": question,
                    "kg_raw":   _triples_to_list(raw)   if raw   is not None else None,
                    "kg_coref": _triples_to_list(coref) if (lang == "en" and coref is not None) else None,
                    "source": source,
                    "lang":   lang,
                    "cefr":   cefr,
                }

                if args.verbose:
                    window = _sentences_around(context, answer_start)
                    ans_lower = answer.lower()
                    print(f"\n{'─'*60}")
                    print(f"PASSAGE ({len(context)} chars, ±2 sentences around answer):")
                    print(f"  {window}")
                    print(f"\nANSWER: {answer!r}")
                    n_raw   = len(triples_raw) if triples_raw else 0
                    n_shown = len(raw)         if raw         else 0
                    print(f"\nKG_RAW ({n_raw} triples, using first {n_shown}):")
                    for t in (raw or []):
                        subj_l = t.subject.lower()
                        obj_l  = t.object.lower()
                        marker = " ◄ answer" if (subj_l in ans_lower or ans_lower in subj_l
                                                  or obj_l in ans_lower or ans_lower in obj_l) else ""
                        print(f"  {t.subject!r:30s} | {t.relation:25s} | {t.object!r}{marker}")
                    if lang == "en":
                        print(f"\nKG_COREF (resolved):")
                        for t in (coref or []):
                            print(f"  {t.subject!r:30s} | {t.relation:25s} | {t.object!r}")
                    print(f"\nCEFR: {cefr}  (readability={s_read:.2f})")
                    print(f"TARGET: {question}")

                ft, fe = out_files[lang]
                line = json.dumps(record, ensure_ascii=False) + "\n"
                if _is_train(passage_id):
                    ft.write(line)
                    counts[lang][0] += 1
                else:
                    fe.write(line)
                    counts[lang][1] += 1

                total_ok += 1
                source_count += 1

                if not args.verbose and source_count % 500 == 0:
                    print(f"  [{source}] processed {source_count} samples", flush=True)

            print(f"[{source}] done: {source_count} records kept", flush=True)

    finally:
        for ft, fe in out_files.values():
            ft.close()
            fe.close()

    print(
        f"\nSummary: {total_in} in → "
        f"{total_filtered_len} too-long, {total_no_triples} too-few-triples, "
        f"{total_failed_kg} kg-failed (stored with null kg), "
        f"{total_ok} written"
    )
    for lang, (n_train, n_eval) in counts.items():
        print(f"  {lang}: {n_train} train / {n_eval} eval")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified QG training data from SQuAD and TyDiQA."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(_LOADERS),
        default=["squad", "tydiqa"],
    )
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument("--lang", choices=["en", "fi"], default=None)
    parser.add_argument("--max-passage-length", type=int, default=None, metavar="CHARS")
    parser.add_argument("--min-triples", type=int, default=2, metavar="N")
    parser.add_argument("--max-triples", type=int, default=50, metavar="N")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-coref", action="store_true",
                        help="Skip coreference resolution (kg_coref will be null).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    build(args)


if __name__ == "__main__":
    main()
