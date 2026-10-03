#!/usr/bin/env python3
"""
Refits every IRT estimate (Rasch + Bayesian; human-only, model-only,
combined) EXCLUDING the 66 items whose `correct_answer_text` doesn't match
any of their own options -- see irt_human_vs_model.ipynb §7.2 for how these
were found and confirmed (RACE answer-key resolution bug in
HumanResponseBank, not a real difficulty signal). Those 66 items score
0.000 accuracy across all 38 cascade candidates by construction (no
answerer can match an unmatchable gold string), and the same bug makes
every human response to them resolve to is_correct=False too -- so they
were previously included in every fit's response matrix as "everyone got
this wrong", contaminating not just their own difficulty estimate but,
via joint estimation, every person's fitted ability (and so, indirectly,
every other item's difficulty).

irt_human_vs_model.ipynb §7.2 only EXCLUDED these items from its
correlation checks after the fact -- the fits themselves still saw them.
This script redoes the fits properly, with the 66 items removed from the
response matrix before fitting, and overwrites the cache files
irt_human_vs_model.ipynb reads. Existing caches are backed up first
(suffix "_with_broken_items") rather than silently discarded.

The underlying data (qa.tsv, cascade_results_all_scored.json) is NOT
modified -- only these derived IRT cache files are regenerated. The root
HumanResponseBank bug itself is left unfixed (separate, larger task).

Usage:
  python question_difficulty/scripts/redo_irt_excluding_broken_gold.py --rasch-only
  python question_difficulty/scripts/redo_irt_excluding_broken_gold.py --person-sets human,model,combined
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = REPO_ROOT / "question_difficulty/notebooks"
ONESTOPQA_REPO = REPO_ROOT.parent / "onestop-qa"
IRT_DIR = REPO_ROOT / "question_difficulty/methods/human_answer_based/irt"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(IRT_DIR))

from data_processing import HumanResponseBank, _texts_match  # noqa: E402
from rasch_model import fit_rasch  # noqa: E402


def find_broken_gold_items() -> set[str]:
    """Items where correct_answer_text matches NONE of the options, even
    after the pipeline's own normalization -- see
    irt_human_vs_model.ipynb §7.2.

    Deliberately does NOT use cascade_common.load_items() -- that filters
    to items already present in RASCH_DIFFICULTY_CACHE, which is one of
    the files THIS script overwrites. Re-running this script a second
    time would then see an already-fixed (smaller) cache, filter the scan
    down to the already-clean items, and silently find 0 broken items --
    a circular dependency that clobbered a correct first run back to the
    broken 1296-item state during development of this script. Reading
    HumanResponseBank directly sidesteps the cache entirely."""
    bank = HumanResponseBank(ONESTOPQA_REPO)
    broken = set()
    for item_id, rows in bank.rows_by_item().items():
        r = rows[0]
        options = r["options"].split("|")
        correct = r["correct_answer_text"]
        if not any(_texts_match(opt, correct) for opt in options):
            broken.add(item_id)
    return broken


def build_human_triples(exclude: set[str]) -> list[tuple[str, str, bool]]:
    """Same (age_group|education bucket, item_id, is_correct) construction
    as question_difficulty_with_irt.ipynb §7/§8, minus `exclude`."""
    import pandas as pd

    bank = HumanResponseBank(ONESTOPQA_REPO)
    all_rows = bank.rows

    df = pd.DataFrame(all_rows)
    person_df = (
        df.groupby("startTime")
        .agg(age=("age", "first"), education=("education", "first"))
        .reset_index()
    )
    person_df["age"] = pd.to_numeric(person_df["age"], errors="coerce")
    person_df["education"] = person_df["education"].replace("", "Unknown")
    age_bins = [18, 25, 35, 45, 55, 76]
    age_labels = ["18-24", "25-34", "35-44", "45-54", "55+"]
    person_df["age_group"] = pd.cut(person_df["age"], bins=age_bins, labels=age_labels, right=False)
    # .astype(str) on a Categorical with NaN entries can yield an actual float NaN
    # (not the string "nan") depending on pandas/string-dtype version -- normalize
    # explicitly so every bucket key is a plain, sortable string.
    age_group_str = person_df["age_group"].astype(object).where(person_df["age_group"].notna(), "Unknown").astype(str)
    person_bucket = dict(zip(person_df["startTime"], age_group_str + "|" + person_df["education"].astype(str)))

    return [
        (person_bucket[r["startTime"]], r["item_id"], r["is_correct"])
        for r in all_rows
        if r["is_correct"] is not None and r["startTime"] in person_bucket
        and r["item_id"] not in exclude
    ]


def build_model_triples(exclude: set[str]) -> list[tuple[str, str, bool]]:
    cascade_results = json.loads((SCRIPTS_DIR / "cascade_results_all_scored.json").read_text())
    return [
        (candidate, item_id, ans["is_correct"])
        for item_id, row in cascade_results.items()
        if item_id not in exclude
        for candidate, ans in row["answers"].items()
    ]


def backup(path: Path) -> None:
    if path.exists():
        backup_path = path.with_name(path.stem + "_with_broken_items" + path.suffix)
        if not backup_path.exists():  # don't clobber a backup from a previous run of this script
            shutil.copy(path, backup_path)
            print(f"  backed up {path.name} -> {backup_path.name}")


def save(path: Path, data: dict) -> None:
    backup(path)
    path.write_text(json.dumps(data))
    print(f"  wrote {path}")


def run_rasch(person_set: str, triples: list[tuple[str, str, bool]]) -> None:
    print(f"\n[{person_set}] Rasch: {len(triples)} triples, "
          f"{len(set(t[0] for t in triples))} persons, {len(set(t[1] for t in triples))} items")
    result = fit_rasch(triples)
    print(f"  converged={result.converged} loss={result.loss:.2f}")

    if person_set == "human":
        save(NOTEBOOKS_DIR / "irt_rasch_difficulty_cache.json", result.item_difficulty)
    else:
        save(SCRIPTS_DIR / f"irt_rasch_{person_set}_only_difficulty.json"
             if person_set == "model" else SCRIPTS_DIR / "irt_rasch_combined_difficulty.json",
             result.item_difficulty)
        save(SCRIPTS_DIR / f"irt_rasch_{person_set}_only_ability.json"
             if person_set == "model" else SCRIPTS_DIR / "irt_rasch_combined_ability.json",
             result.person_ability)


def run_bayesian(person_set: str, triples: list[tuple[str, str, bool]]) -> None:
    from bayesian_irt_model import fit_bayesian_irt

    print(f"\n[{person_set}] Bayesian: {len(triples)} triples, "
          f"{len(set(t[0] for t in triples))} persons, {len(set(t[1] for t in triples))} items")
    result = fit_bayesian_irt(triples)
    print(f"  divergences={result.n_divergences} max_rhat={result.max_rhat:.4f} min_ess={result.min_ess:.0f}")

    prefix = "human_only" if person_set == "human" else f"{person_set}_only" if person_set == "model" else "combined"
    save(SCRIPTS_DIR / f"irt_bayesian_{prefix}_difficulty.json", result.item_difficulty)
    save(SCRIPTS_DIR / f"irt_bayesian_{prefix}_difficulty_std.json", result.item_difficulty_std)
    save(SCRIPTS_DIR / f"irt_bayesian_{prefix}_ability.json", result.person_ability)
    save(SCRIPTS_DIR / f"irt_bayesian_{prefix}_ability_std.json", result.person_ability_std)
    save(SCRIPTS_DIR / f"irt_bayesian_{prefix}_discrimination.json", result.item_discrimination)
    save(SCRIPTS_DIR / f"irt_bayesian_{prefix}_discrimination_std.json", result.item_discrimination_std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person-sets", default="human,model,combined")
    parser.add_argument("--rasch-only", action="store_true", help="skip the slower Bayesian refit")
    args = parser.parse_args()
    person_sets = args.person_sets.split(",")

    broken = find_broken_gold_items()
    print(f"excluding {len(broken)} broken-gold-answer items from every fit below")

    human_triples = build_human_triples(broken) if "human" in person_sets else None
    model_triples = build_model_triples(broken) if "model" in person_sets or "combined" in person_sets else None
    combined_triples = (human_triples or build_human_triples(broken)) + model_triples \
        if "combined" in person_sets else None

    triples_by_set = {"human": human_triples, "model": model_triples, "combined": combined_triples}

    for person_set in person_sets:
        run_rasch(person_set, triples_by_set[person_set])

    if not args.rasch_only:
        for person_set in person_sets:
            run_bayesian(person_set, triples_by_set[person_set])


if __name__ == "__main__":
    main()
