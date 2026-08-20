#!/bin/bash
# Run on the Mahti login node before submitting training jobs.
# Downloads HF datasets to cache, then prepares all training data files.
#
# Usage:
#   bash scripts/prepare_all_data.sh
#   bash scripts/prepare_all_data.sh --qde-only
#   bash scripts/prepare_all_data.sh --qg-only

set -e

QDE=1
QG=1
for arg in "$@"; do
    case $arg in
        --qde-only) QG=0 ;;
        --qg-only)  QDE=0 ;;
    esac
done

export HF_HOME=/scratch/project_2006600/ducvu/hf_cache
export HF_DATASETS_CACHE=/scratch/project_2006600/ducvu/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/project_2006600/ducvu/hf_cache

cd /scratch/project_2006600/ducvu/reading-question-generator

echo "=== Downloading HF datasets ==="
python scripts/download_resources.py

if [ $QDE -eq 1 ]; then
    echo ""
    echo "=== Preparing QDE data (balanced, ~38K total) ==="
    python question_difficulty/scripts/prepare_qde_data.py \
        --balanced \
        --output-dir data/qde
fi

if [ $QG -eq 1 ]; then
    echo ""
    echo "=== Preparing QG data (step0 + step2 + step3) ==="
    python question_generation/scripts/prepare_qg_data.py \
        --steps step0 step2 step3 \
        --output-dir data/qg
fi

echo ""
echo "=== Data sizes ==="
for f in data/qde/train.jsonl data/qde/val.jsonl data/qde/test.jsonl; do
    [ -f $f ] && echo "  $f: $(wc -l < $f) records"
done
for step in step0 step2 step3; do
    for split in train val test; do
        f=data/qg/$step/$split.jsonl
        [ -f $f ] && echo "  $f: $(wc -l < $f) records"
    done
done

echo ""
echo "Done. Submit jobs with:"
echo "  sbatch question_difficulty/slurms/train_qde_feature_based.job"
echo "  sbatch --array=0-1 question_difficulty/slurms/train_qde_encoder.job"
echo "  sbatch --array=0-1 question_difficulty/slurms/train_qde_contrastive.job"
echo "  sbatch --array=0-2 question_generation/slurms/train_qg_t5base.job"
