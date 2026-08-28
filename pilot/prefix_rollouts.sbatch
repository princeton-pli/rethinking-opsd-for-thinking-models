#!/bin/bash
#SBATCH --job-name=cd_prefix
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Thinking-retained prefix rollouts for grader-RL episodes (SPEC §5).
# PAIRS selects the harvest; default is the pattern-balanced (primary) set.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

PAIRS=${PAIRS:-pilot/countdown_pairs_balanced.parquet}
OUT=${OUT:-pilot/prefix_rollouts_balanced.jsonl}
[ -f "$PAIRS" ] || { echo "pairs parquet missing: $PAIRS"; exit 1; }

python pilot/gen_prefix_rollouts.py --pairs "$PAIRS" --k 4 \
    --max_tokens 16384 --out "$OUT" || exit 1
echo "PREFIX ROLLOUTS DONE"
