#!/bin/bash
#SBATCH --job-name=cd_bal
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Pattern-balanced hard-Countdown gate run (SPEC_grader_rl.md §3-amended, §4):
# harvest (x+, x-) pairs from RL-skill-comp-style balanced instances (n=5,6),
# then the 3x2 adjudication probe. This is the PRIMARY gate set; the
# tree-generator slice (cd_hard job) is the secondary datapoint.
# Same thresholds: usable >= 25%, adjudication-when-copying >= 65% nothink.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

BAL=data/raw/countdown_balanced_3k.parquet
[ -f "$BAL" ] || { echo "balanced parquet missing -- run pilot/gen_countdown_balanced.py first"; exit 1; }

# Idempotent: harvest parquet is written last, so its existence implies a
# complete harvest; a resubmit goes straight to the probe.
if [ ! -f pilot/countdown_pairs_balanced.parquet ]; then
    python pilot/countdown_pairs.py --countdown "$BAL" --n_problems 2000 \
        --max_tokens 16384 --out pilot/countdown_pairs_balanced.jsonl || exit 1
fi
python pilot/probe_adjudication.py --parquet pilot/countdown_pairs_balanced.parquet \
    --n_problems 300 --out pilot/probe_countdown_balanced.jsonl || exit 1
echo "BALANCED COUNTDOWN GATE RUN DONE"
