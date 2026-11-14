#!/usr/bin/env python3
"""One-off: reconstruct the sample_rerank_diffcond.py (job 2983916) output
as jsonl, using the same deterministic passage selection (seed=0) plus the
exact generated texts/margins already captured from that job's log."""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

records = [json.loads(l) for l in (REPO_ROOT / "data/qg/raw/race_irt_10k_blind/test.jsonl").open(encoding="utf-8") if l.strip()]
by_passage: dict[str, list[dict]] = {}
for r in records:
    by_passage.setdefault(r["passage"], []).append(r)
passages = list(by_passage.keys())
random.Random(0).shuffle(passages)
picked = passages[:8]

GENERATED = [
    [("EASY", 0.95, "The author decided to write his memoir mainly _ ."),
     ("MEDIUM", 1.72, "The main purpose of the text is to _ ."),
     ("HARD", 2.34, "The principal uses the author's words in Para. 2 as an example to show that _ .")],
    [("EASY", 2.00, "What is the author's opinion of the reaction to violence?"),
     ("MEDIUM", 0.59, "The author mentions the famous lawyer to _ ."),
     ("HARD", 0.91, "What does the author think of the rational argument over violence?")],
    [("EASY", 0.70, "When the writer was free, _ ."),
     ("MEDIUM", 1.91, "According to the text, the policemen _ ."),
     ("HARD", 1.72, "Why did the writer return to the back porch when she returned home?")],
    [("EASY", 2.18, 'The word "crude"(Line 4, Para. 1) probably means _ .'),
     ("MEDIUM", 0.68, "Mei Jing failed to get the required qualifications _ ."),
     ("HARD", 0.57, "Which would be the best title for the text?")],
    [("EASY", 0.89, "What happened to the Internet in the 1970s?"),
     ("MEDIUM", 1.95, "Which of the following is true according to the passage ?"),
     ("HARD", 0.63, "Why was the Internet cheap in 1970s?")],
    [("EASY", 0.82, "Why did Maggie pay no attention to old Dan's words?"),
     ("MEDIUM", 0.94, "What is Maggie supposed to do now ?"),
     ("HARD", 1.14, "Who bought Maggie a penny?")],
    [("EASY", 0.75, "What is the best title for the passage?"),
     ("MEDIUM", 1.52, "The public reaction to Laila's success of the boxing match was _ ."),
     ("HARD", 0.33, "Why did Laila change her attitude toward boxing?")],
    [("EASY", 0.75, "What's the best title of the passage?"),
     ("MEDIUM", 1.27, "The _ people have their favorite way of traveling."),
     ("HARD", 0.97, "Who likes traveling fastest?")],
]

out_path = REPO_ROOT / "question_generation/results/sample_rerank.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for passage, triples in zip(picked, GENERATED):
        for difficulty, margin, question in triples:
            f.write(json.dumps({
                "passage": passage, "difficulty": difficulty,
                "question": question, "margin": margin,
            }, ensure_ascii=False) + "\n")

print(f"Wrote {sum(len(t) for t in GENERATED)} records to {out_path}")
