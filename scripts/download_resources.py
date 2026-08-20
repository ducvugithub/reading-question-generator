#!/usr/bin/env python3
"""Download HuggingFace datasets to cache.
Run once on the login node before submitting sbatch jobs.
"""


def main():
    print("=== Downloading HuggingFace datasets ===", flush=True)
    from datasets import load_dataset

    # QDE training data (Step 1) + QG baseline/difficulty (Steps 0, 2)
    print("  race-middle ...", flush=True)
    load_dataset("ehovy/race", "middle", split="train")
    print("  race-high ...", flush=True)
    load_dataset("ehovy/race", "high", split="train")
    print("  race-c (college/Gaokao) ...", flush=True)
    load_dataset("tasksource/race-c", split="train")

    # QG focus-span training data (Steps 3, 4)
    print("  hotpotqa (distractor) ...", flush=True)
    load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # QG focus-span (Step 3)
    print("  multirc (super_glue) ...", flush=True)
    load_dataset("aps/super_glue", "multirc", split="train")
    load_dataset("aps/super_glue", "multirc", split="validation")

    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
