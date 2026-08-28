#!/bin/bash
#SBATCH --job-name=cd_hard
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Hard-Countdown gate run (SPEC_grader_rl.md §4): harvest (x+, x-) pairs from
# generated 5-6 number instances, then the 3x2 adjudication probe on them.
#
# Proceed to RL only if (thresholds from the spec):
#   usable-pair rate >= 25%   (3-4 number baseline: 7.9%)
#   adjudication-when-copying >= 65% nothink on hard pairs
#
# max_tokens 16384 (not the 8192 default): harder search means longer thinking;
# a truncated rollout is excluded as incomplete, and a high truncation rate
# would silently starve the pair pool (the Opus thinking-ceiling lesson).
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

HARD=data/raw/countdown_hard_3k.parquet
[ -f "$HARD" ] || { echo "hard parquet missing -- run pilot/gen_hard_countdown.py first"; exit 1; }

# Idempotent: the harvest parquet is written last by countdown_pairs.py, so
# its existence implies a complete harvest -- a resubmit goes straight to the
# probe instead of redoing 4+ GPU-hours (CR finding).
if [ ! -f pilot/countdown_pairs_hard.parquet ]; then
    python pilot/countdown_pairs.py --countdown "$HARD" --n_problems 2000 \
        --max_tokens 16384 --out pilot/countdown_pairs_hard.jsonl || exit 1
fi
python pilot/probe_adjudication.py --parquet pilot/countdown_pairs_hard.parquet \
    --n_problems 300 --out pilot/probe_countdown_hard.jsonl || exit 1
echo "HARD COUNTDOWN GATE RUN DONE"
