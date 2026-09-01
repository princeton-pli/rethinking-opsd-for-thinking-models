#!/bin/bash
#SBATCH --job-name=eval_student
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/rl/slurm-%j.out
#
# Held-out solve rate for ONE student checkpoint (MODEL/TAG env).
# Primary outcome for the OPSD arms; bare-question prompts, k=8, temp 1.0.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

: "${MODEL:?set MODEL}" "${TAG:?set TAG}"
python rl/eval_student_solve.py \
    --model "$MODEL" --tag "$TAG" \
    --heldout data/raw/countdown_opsd_heldout.parquet \
    --k 8 --out "rl/solve_${TAG}.jsonl" || exit 1
echo "STUDENT EVAL DONE: $TAG"
