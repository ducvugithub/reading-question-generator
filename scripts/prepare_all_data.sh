#!/bin/bash
# Run on the Roihu login node before submitting training jobs.
# Downloads HF datasets to cache, then prepares all training data files.
#
# ⚠️  BEFORE SSH-ING IN: renew your CSC certificate (valid 24h) if needed:
#   cp ~/Downloads/cert*.pub ~/.ssh/my_csc_key-cert.pub
#   ssh-keygen -L -f ~/.ssh/my_csc_key-cert.pub | grep Valid   # check expiry
#   Download fresh cert from: https://my.csc.fi → Profile → SSH PUBLIC KEYS
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

export HF_HOME=/scratch/project_2006601/ducvu/hf_cache
export HF_DATASETS_CACHE=/scratch/project_2006601/ducvu/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/project_2006601/ducvu/hf_cache

module purge
module load python-pytorch/2.10

cd /scratch/project_2006601/ducvu/reading-question-generator

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
    echo "=== Preparing QG data (baseline + diff-control + focus-span-control) ==="
    python question_generation/scripts/prepare_qg_data.py \
        --steps baseline diff-control focus-span-control \
        --output-dir data/qg
fi

echo ""
echo "=== Data sizes ==="
for f in data/qde/train.jsonl data/qde/val.jsonl data/qde/test.jsonl; do
    [ -f $f ] && echo "  $f: $(wc -l < $f) records"
done
for step in baseline diff-control focus-span-control; do
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
