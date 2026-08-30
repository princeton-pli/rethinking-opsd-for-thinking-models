#!/bin/bash
#SBATCH --job-name=grader_eval
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/rl/slurm-%j.out
#
# Post-training eval batch for grader-RL (SPEC §6), checkpoint step 40:
#   1. PRIMARY: forward-pass logprob margin, {pair,bare} x {base,adapter},
#      92 held-out pairs (rl/eval_logprob_margin.py).
#   2. Adjudication probe with the MERGED model on the same 300 problems as
#      the pre-RL balanced probe (same parquet, same seed) -- gives the
#      before/after adjudication cells AND the none/* transfer cells for
#      Sanjeev's grading->solving hypothesis.
# Grading of (2) happens on CPU afterwards (grade_countdown_probe.py
# --probe pilot/probe_countdown_balanced_post.jsonl).
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

CKPT=/scratch/gpfs/ARORA/arora/opsd/rl/checkpoints/grader_grpo_lora_v1
[ -d "$CKPT/eval_adapter" ] || { echo "eval_adapter missing"; exit 1; }
[ -d "$CKPT/eval_merged" ] || { echo "eval_merged missing"; exit 1; }

python rl/eval_logprob_margin.py \
    --pairs pilot/countdown_pairs_balanced.parquet \
    --test_parquet rl/episodes_balanced/test.parquet \
    --out rl/margin_base.jsonl || exit 1

python rl/eval_logprob_margin.py \
    --pairs pilot/countdown_pairs_balanced.parquet \
    --test_parquet rl/episodes_balanced/test.parquet \
    --adapter "$CKPT/eval_adapter" \
    --out rl/margin_adapter.jsonl || exit 1

python pilot/probe_adjudication.py \
    --parquet pilot/countdown_pairs_balanced.parquet \
    --adapter "$CKPT/eval_adapter" \
    --n_problems 300 \
    --out pilot/probe_countdown_balanced_post.jsonl || exit 1

echo "GRADER EVAL BATCH DONE"
