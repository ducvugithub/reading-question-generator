#!/bin/bash
# Run QDE evaluation on Roihu GPU login node.
# Usage: bash scripts/evaluate_qde.sh

module purge
module load python-pytorch/2.10

cd /scratch/project_2006600/ducvu/reading-question-generator

pip install --user wordfreq --quiet

python question_difficulty/scripts/evaluate_qde.py "$@"
