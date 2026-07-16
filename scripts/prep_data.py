#!/usr/bin/env python3
"""Download HuggingFace datasets and Stanza models to cache.
Run once on the login node before submitting sbatch jobs.
"""
import os
import sys

def main():
    print("=== Downloading HuggingFace datasets ===", flush=True)
    from datasets import load_dataset
    print("  squad_v2 ...", flush=True)
    load_dataset("rajpurkar/squad_v2", split="train")
    print("  tydiqa ...", flush=True)
    load_dataset("google-research-datasets/tydiqa", "secondary_task", split="train")
    print("  SQuAD_v2_fi ...", flush=True)
    load_dataset("ilmariky/SQuAD_v2_fi", split="train")

    print("=== Downloading Stanza models ===", flush=True)
    import stanza
    stanza.download("en")
    stanza.download("fi")

    print("=== Done ===", flush=True)

if __name__ == "__main__":
    main()
