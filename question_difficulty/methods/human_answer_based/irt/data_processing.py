"""
Real human difficulty ground truth from berzak/onestop-qa's Prolific
reading-comprehension experiment -- an alternative to attention-entropy-
based difficulty signals (see question_difficulty/notebooks/
entropy_manual_review.ipynb for why those turned out to be unreliable).

Requires a local clone of https://github.com/berzak/onestop-qa (pass its
path to HumanResponseBank).

qa.tsv itself has no correct-answer column -- every design decision below
(article-number-to-title mapping, 1-indexed paragraph/question lookup,
answer_response indexing directly into `options` rather than via
`new_options`, substring-match scoring for a handful of tsv truncation
artifacts) was empirically verified against the real data; see
question_difficulty/notebooks/human_difficulty_onestopqa.ipynb for that
verification (100% of 3880 rows resolved, ~76-81% accuracy -- the wrong
answer_response interpretation gives ~26-28%, i.e. chance level).
"""
from __future__ import annotations

import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# article number (from item_id "os<N>_...") -> filename, per
# human_experiments/README.md's table in the onestop-qa repo.
_NUM_TO_FNAME = {
    1: "Google-introduces-its-driverless-car",
    2: "Love-hormone-helps-autistic-children-bond-with-others",
    3: "Spain's-robin-hood",
    4: "Will-drones-soon-be-delivering-packages-to-your-doorstep",
    5: "Philip-pullman-illegal-downloading-is-moral-squalor",
    6: "Swarthy-blue-eyed-caveman-revealed",
    7: "Inky-the-octopus-escapes-from-aquarium",
    8: "Autumn-born-children-better-at-sport-says-study",
    9: "Why-you-should-start-work-at-10am",
    10: "The-Greek-island-where-time-is-running-out",
    11: "The-secrets-of-the-mystery-shopper",
    12: "On-the-trail-of-the-wolf",
    13: "Why-is-Sweden-closing-its-prisons",
    14: "Four-new-elements-find-a-place-on-periodic-table",
    15: "A-new-form-of-lie-detector-test",
    16: "Vienna-named-worlds-top-city-for-quality-of-life",
    17: "Wealth-therapy-for-the-rich",
    18: "Rwandan-women-whip-up-popular-ice-cream-business",
    19: "Bolivians-demand-the-right-to-chew-coca-leaves",
    20: "Insects-could-be-the-planets-next-food-source",
}
# JSON title for #8 says "Sports" (plural); README filename says "sport" (singular).
_TITLE_OVERRIDES = {8: "Autumn-Born Children Better at Sports Says Study"}
_ITEM_ID_RE = re.compile(r"os(\d+)_(\d+)_(\d+)_(\w+)")
_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _norm_text(s: str) -> str:
    return (s.replace("‘", "'").replace("’", "'")
             .replace("“", '"').replace("”", '"').strip().lower())


def _texts_match(a: str, b: str) -> bool:
    a, b = _norm_text(a), _norm_text(b)
    return a in b or b in a  # substring: qa.tsv occasionally truncates a character


class HumanResponseBank:
    """Loads berzak/onestop-qa's qa.tsv (RACE + OneStopQA Prolific reading-
    comprehension responses) and resolves each response's correctness
    against the source datasets. `self.rows` is the list of raw tsv rows
    (as dicts), each augmented with "correct_answer_text" and "is_correct"
    (None if unresolvable)."""

    def __init__(self, onestopqa_repo: str | Path):
        self.repo = Path(onestopqa_repo)
        qa_tsv = self.repo / "human_experiments/prolific/qa.tsv"
        onestop_zip = self.repo / "annotations/onestop_qa.zip"
        if not qa_tsv.exists():
            raise FileNotFoundError(f"{qa_tsv} not found -- clone berzak/onestop-qa first")

        with open(qa_tsv) as f:
            self.rows = [r for r in csv.DictReader(f, delimiter="\t")
                         if r["trial_type"] == "reading_comprehension"]

        with zipfile.ZipFile(onestop_zip) as zf:
            onestop_data = json.loads(zf.read("onestop_qa.json"))["data"]
        self._num_to_article = self._build_article_map(onestop_data)
        self._race_question_to_answer: dict[str, str] | None = None  # lazy: needs `datasets`

        self._resolve_correctness()

    @staticmethod
    def _build_article_map(onestop_data: list[dict]) -> dict[int, dict]:
        num_to_article = {}
        for num, fname in _NUM_TO_FNAME.items():
            target = _TITLE_OVERRIDES.get(num, fname)
            matches = [a for a in onestop_data if _norm_title(a["title"]) == _norm_title(target)]
            if len(matches) != 1:
                raise ValueError(f"article #{num} ({fname!r}) matched {len(matches)} entries")
            num_to_article[num] = matches[0]
        return num_to_article

    def _resolve_onestop_correct_answer(self, item_id: str) -> str | None:
        """STARC convention: answers[0] is always the correct answer."""
        m = _ITEM_ID_RE.match(item_id)
        if not m:
            return None
        art_num, para_num, q_num = int(m.group(1)), int(m.group(2)), int(m.group(3))
        article = self._num_to_article.get(art_num)
        if article is None:
            return None
        try:
            qas = article["paragraphs"][para_num - 1]["qas"][q_num - 1]
        except IndexError:
            return None
        return qas["answers"][0]

    def _ensure_race_lookup(self) -> None:
        if self._race_question_to_answer is not None:
            return
        from datasets import load_dataset
        lookup: dict[str, str] = {}
        for subset in ("middle", "high"):
            for rec in load_dataset("ehovy/race", subset, split="test"):
                idx = _LETTER_TO_IDX.get(rec["answer"])
                if idx is None or idx >= len(rec["options"]):
                    continue
                lookup.setdefault(rec["question"].strip(), rec["options"][idx])
        self._race_question_to_answer = lookup

    def _resolve_correctness(self) -> None:
        if any(r["source"] == "RACE" for r in self.rows):
            self._ensure_race_lookup()

        for r in self.rows:
            if r["source"] == "Onestop":
                correct_text = self._resolve_onestop_correct_answer(r["item_id"])
            elif r["source"] == "RACE":
                correct_text = self._race_question_to_answer.get(r["question"].strip())
            else:
                correct_text = None
            r["correct_answer_text"] = correct_text

            if correct_text is None:
                r["is_correct"] = None
                continue
            options = r["options"].split("|")
            resp_idx = int(r["answer_response"])
            if resp_idx >= len(options):
                r["is_correct"] = None
                continue
            r["is_correct"] = _texts_match(options[resp_idx], correct_text)

    def rows_by_item(self) -> dict[str, list[dict]]:
        by_item = defaultdict(list)
        for r in self.rows:
            by_item[r["item_id"]].append(r)
        return dict(by_item)

    def item_accuracy(self) -> dict[str, float]:
        """Per-item fraction of participants who answered correctly.
        Excludes items whose correctness couldn't be resolved."""
        by_item = defaultdict(list)
        for r in self.rows:
            if r["is_correct"] is not None:
                by_item[r["item_id"]].append(r["is_correct"])
        return {item_id: sum(vals) / len(vals) for item_id, vals in by_item.items() if vals}

    def item_metadata(self) -> dict[str, tuple[str, str]]:
        """item_id -> (source, difficulty), from the first row seen for that item."""
        meta: dict[str, tuple[str, str]] = {}
        for r in self.rows:
            meta.setdefault(r["item_id"], (r["source"], r["difficulty"]))
        return meta
