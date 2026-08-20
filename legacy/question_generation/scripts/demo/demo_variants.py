#!/usr/bin/env python3
"""
Demo script: Generate questions with difficulty variants from a passage.

Usage:
    INPUT=data/fi/demo.txt OUTPUT=out.md TARGET_CEFR=C1 python scripts/demo_variants.py
    INPUT=passage.txt python scripts/demo_variants.py
    python scripts/demo_variants.py --input passage.txt --output out.md --target-cefr B1
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph import KnowledgeGraphExtractor, KnowledgeGraph, resolve_coreferences
from question_generation.generator import QuestionGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate questions with difficulty variants")
    parser.add_argument("--input", "-i", help="Input file path", default=os.getenv("INPUT", "passage.txt"))
    parser.add_argument("--output", "-o", help="Output file path (auto-generated if not provided)", default=os.getenv("OUTPUT"))
    parser.add_argument("--target-cefr", "-c", help="Target CEFR level", default=os.getenv("TARGET_CEFR", "B1"))
    parser.add_argument("--num-questions", "-n", type=int, help="Number of questions", default=os.getenv("NUM_QUESTIONS", 30))
    parser.add_argument("--verbose", "-v", action="store_true", help="Print questions to terminal")
    return parser.parse_args()


def read_passage(filepath):
    """Read passage from file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)


def write_output(filepath, content):
    """Write output to file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Output written to: {filepath}")


def generate_demo(passage, lang="en", target_cefr="B1", num_questions=30):
    """Generate questions with variants from passage."""

    print("⚙️  Extracting knowledge graph...")
    extractor = KnowledgeGraphExtractor(lang=lang)
    triples = extractor.extract(passage)
    triples = resolve_coreferences(triples, lang=lang)

    kg = KnowledgeGraph()
    kg.add_triples(triples)

    print(f"✓ Extracted {len(triples)} triples")
    print(f"✓ Knowledge graph: {kg._g.number_of_nodes()} entities, {kg._g.number_of_edges()} relations")

    print(f"\n🔄 Generating questions WITH difficulty variants (target: {target_cefr})...")
    gen = QuestionGenerator(lang=lang)
    questions = gen.generate(
        triples, kg,
        num_questions=num_questions,
        create_variants=True,
        target_template_cefr=target_cefr
    )

    print(f"✓ Generated {len(questions)} questions\n")
    return questions, kg


def format_output(passage, questions, kg, target_cefr="B1", lang="en"):
    """Format output as markdown with tables."""
    from datetime import datetime

    output = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header
    output.append("# Question Generation Report\n\n")
    output.append(f"Generated: {timestamp}  \n")
    output.append(f"Language: **{lang.upper()}**  \n")
    output.append(f"Questions: **{len(questions)}**\n\n")

    # Input section
    output.append("## Input Text\n\n")
    output.append(passage)
    output.append("\n\n")

    # Questions table (remove duplicates)
    output.append("## Questions\n\n")
    output.append("| Level | Type | Question | Answer |\n")
    output.append("|-------|------|----------|--------|\n")

    seen = set()
    for q in questions:
        if q.text in seen:
            continue
        seen.add(q.text)
        level = q.difficulty or "?"
        qtype = q.masked or "?"
        answer = ", ".join(q.answer_list) if q.answer_list else q.answer
        # Escape pipe characters in question/answer
        question_text = q.text.replace("|", "\\|")
        answer_text = answer.replace("|", "\\|")
        output.append(f"| **[{level}]** | {qtype} | {question_text} | {answer_text} |\n")

    output.append("\n")

    # Knowledge graph section
    output.append("## Knowledge Graph\n\n")
    output.append(f"**{kg._g.number_of_nodes()}** nodes · **{kg._g.number_of_edges()}** edges\n\n")

    # Nodes table
    output.append("### Nodes\n\n")
    output.append("| Entity | Type |\n")
    output.append("|--------|------|\n")
    for node, data in sorted(kg.nodes):
        etype = data.get("entity_type", "UNKNOWN")
        output.append(f"| `{node}` | {etype} |\n")

    output.append("\n")

    # Edges table
    output.append("### Edges\n\n")
    output.append("| Subject | Relation | Object |\n")
    output.append("|---------|----------|--------|\n")
    for src, dst, data in sorted(kg.edges):
        rel = data.get("relation", "unknown")
        output.append(f"| `{src}` | {rel} | `{dst}` |\n")

    return "".join(output)


def main():
    args = parse_args()

    print("=" * 80)
    print("QUESTION GENERATION WITH DIFFICULTY VARIANTS")
    print("=" * 80)

    # Read input passage
    print(f"\n📖 Reading passage from: {args.input}")
    passage = read_passage(args.input)
    print(f"✓ Passage length: {len(passage)} characters\n")

    # Detect language from filename (e.g., microsoft_en.txt -> "en")
    filename = Path(args.input).stem
    lang = "en"  # default
    if "_en" in filename:
        lang = "en"
    elif "_fi" in filename:
        lang = "fi"

    # Auto-generate output filename if not provided
    output_path = args.output
    if not output_path:
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}_{lang}_{args.target_cefr.lower()}.md"

    # Generate questions
    questions, kg = generate_demo(passage, lang=lang, target_cefr=args.target_cefr, num_questions=args.num_questions)

    # Format and output
    output = format_output(passage, questions, kg, target_cefr=args.target_cefr, lang=lang)

    # Print to terminal if verbose
    if args.verbose:
        print("\n" + "=" * 80)
        print("GENERATED QUESTIONS")
        print("=" * 80)
        print(output)

    # Write to file
    write_output(output_path, output)

    print("=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
