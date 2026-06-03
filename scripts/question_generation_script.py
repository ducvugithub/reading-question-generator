"""
question_generation_script.py  —  CLI: text file → questions + knowledge graph → Markdown report

Usage:
    python scripts/question_generation_script.py <input.txt> [--output report.md] [--max-questions N]
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from langdetect import detect, LangDetectException

from knowledge_graph import KnowledgeGraph, KnowledgeGraphExtractor, resolve_coreferences
from question_generation import QuestionGenerator

_LANG_NAMES = {"en": "English", "fi": "Finnish"}
_LEVEL_LABELS = {
    "preA1": "Beginner", "A1": "Elementary", "A2": "Pre-intermediate",
    "B1": "Intermediate", "B2": "Upper-intermediate",
    "C1": "Advanced", "C2": "Proficiency", "C2+": "Mastery",
}
_FORM_NAMES = {
    "object": "wh-",
    "subject": "wh-",
    "chain": "wh-",      # multi-hop wh-, distinguished by hop count in Difficulty column
    "yesno": "yes/no",
    "comparison": "comparison",
    "which": "which",
    "aggregation": "list",
}


def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "en"
    return lang if lang in _LANG_NAMES else "en"


def build_markdown(text: str, lang: str, kg: KnowledgeGraph, questions: list) -> str:
    lines = []

    lines += [
        f"# Question Generation Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Language: **{_LANG_NAMES.get(lang, lang)}**  ",
        f"Questions: **{len(questions)}**",
        f"",
    ]

    # ── Input text ──────────────────────────────────────────────────────────
    lines += ["## Input Text", ""]
    for sentence in text.strip().split(". "):
        if sentence.strip():
            lines.append(sentence.strip().rstrip(".") + ".")
    lines.append("")

    # ── Questions ────────────────────────────────────────────────────────────
    lines += ["## Questions", ""]
    lines += ["| Type | Text diff | Question diff | Question | Answer |"]
    lines += ["|------|-----------|---------------|----------|--------|"]

    for q in questions:
        form = _FORM_NAMES.get(q.masked, q.masked)
        hops = f" · {q.hop_count}-hop" if q.masked == "chain" else ""
        t_diff = f"**{q.text_difficulty}** {_LEVEL_LABELS.get(q.text_difficulty, '')}{hops}"
        q_diff = f"**{q.question_difficulty}** {_LEVEL_LABELS.get(q.question_difficulty, '')}"
        raw_answer = " / ".join(q.answer_list) if q.answer_list else q.answer
        answer = raw_answer.replace("|", "\\|")
        question = q.text.replace("|", "\\|")
        lines.append(f"| {form} | {t_diff} | {q_diff} | {question} | {answer} |")

    lines.append("")

    # ── Knowledge Graph ───────────────────────────────────────────────────────
    n_nodes = kg._g.number_of_nodes()
    n_edges = kg._g.number_of_edges()
    lines += [
        "## Knowledge Graph",
        "",
        f"**{n_nodes}** nodes · **{n_edges}** edges",
        "",
    ]

    lines += ["### Nodes", ""]
    lines += ["| Entity | Type |"]
    lines += ["|--------|------|"]
    for node, data in kg.nodes:
        etype = data.get("entity_type") or "—"
        lines.append(f"| `{node}` | {etype} |")
    lines.append("")

    lines += ["### Edges", ""]
    lines += ["| Subject | Relation | Object |"]
    lines += ["|---------|----------|--------|"]
    for src, dst, data in kg.edges:
        lines.append(f"| `{src}` | {data['relation']} | `{dst}` |")
    lines.append("")

    return "\n".join(lines)


def run(input_path: Path, output_path: Path, max_questions: int) -> None:
    text = input_path.read_text(encoding="utf-8").strip()
    print(f"Input : {input_path} ({len(text.split())} words)")

    lang = detect_lang(text)
    print(f"Language: {_LANG_NAMES.get(lang, lang)}")

    print("Extracting knowledge graph…")
    extractor = KnowledgeGraphExtractor(lang=lang)
    triples = resolve_coreferences(extractor.extract(text), lang=lang)
    kg = KnowledgeGraph()
    kg.add_triples(triples)
    print(f"  {kg._g.number_of_nodes()} nodes, {kg._g.number_of_edges()} edges")

    print("Generating questions…")
    generator = QuestionGenerator(lang=lang)
    limit = max_questions if max_questions > 0 else 10_000
    questions = generator.generate(triples, kg, num_questions=limit)
    print(f"  {len(questions)} questions")

    for q in questions:
        kind = _FORM_NAMES.get(q.masked, q.masked)
        answer_str = " / ".join(q.answer_list) if q.answer_list else q.answer
        print(f"  [T:{q.text_difficulty:<5}/Q:{q.question_difficulty:<5}][{kind:12}] {q.text!r}  ->  {answer_str!r}")

    md = build_markdown(text, lang, kg, questions)
    output_path.write_text(md, encoding="utf-8")
    print(f"\nReport saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate questions from a text file.")
    parser.add_argument("input", type=Path, help="Path to input .txt file")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .md file (default: same name as input with .md extension)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=0,
        help="Max questions to generate (0 = no limit, default: 0)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_suffix(".md")
    run(args.input, output, args.max_questions)


if __name__ == "__main__":
    main()
