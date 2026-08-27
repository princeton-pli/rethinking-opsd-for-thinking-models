#!/bin/bash
#SBATCH --job-name=cd_probe
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Does the verification-generation gap exist for Countdown?
#
# On competition math the frozen 4B teacher adjudicates its own correct vs wrong
# solutions at 54% without thinking / 63% with (clean pairs, 2026-08-27) -- barely
# above the coin flip, which is why Arm 1 failed: the teacher cannot extract the
# label the design assumes. Countdown should be the opposite regime. Verifying
# "(69-66)+(60/6) = 13 using exactly [6,66,69,60]" is arithmetic; FINDING the
# expression is combinatorial search. CountdownTask.grade is a deterministic
# checker (eval + multiset compare), so there is also no false-negative class of
# the kind that cost 11% of the math labels.
#
# Stage 1: sample k=8 per problem to harvest (x+, x-) pairs.
# Stage 2: run the same 3x2 adjudication probe on those pairs.
# Grading is inline-safe here: CountdownTask.grade is pure eval(), no sympy fork.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

CD=/scratch/gpfs/ARORA/arora/data/raw/countdown_15k.parquet
[ -f "$CD" ] || { echo "countdown parquet missing"; exit 1; }

python pilot/countdown_pairs.py --countdown "$CD" --n_problems 2000 \
    --out pilot/countdown_pairs_v2.jsonl || exit 1
python pilot/probe_adjudication.py --parquet pilot/countdown_pairs_v2.parquet \
    --n_problems 300 --out pilot/probe_countdown_v2.jsonl || exit 1
echo "COUNTDOWN PROBE DONE"
