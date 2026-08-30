#!/bin/bash
#SBATCH --job-name=eval_v4
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/rl/slurm-%j.out
#
# Eval battery for the v4-template GRPO run (grader_grpo_lora_v4tpl).
# All contexts are v4 to match training. Baselines for comparison:
#   base v4 probe: acc 0.732, adj-when-copying 86.6 (think), refs 26.9%
#   base v4 margins/slot: produced by the base-side runs below.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1

CKPT=/scratch/gpfs/ARORA/arora/opsd/rl/checkpoints/grader_grpo_lora_v4tpl
LAST=$(ls -d "$CKPT"/global_step_* | sort -t_ -k3 -n | tail -1)
[ -d "$LAST/actor/lora_adapter" ] || { echo "no lora_adapter in $LAST"; exit 1; }
cp -r "$LAST/actor/lora_adapter" "$CKPT/eval_adapter"
echo "adapter from $LAST"

for A in "" "--adapter $CKPT/eval_adapter"; do
    TAG=$([ -z "$A" ] && echo base || echo adapter)
    python rl/eval_logprob_margin.py --template v4 \
        --pairs pilot/countdown_pairs_balanced.parquet \
        --test_parquet rl/episodes_balanced_v4tpl/test.parquet \
        $A --out rl/margin_${TAG}_v4tpl.jsonl || exit 1
    python rl/eval_sparse_signal.py --template v4 --n_kl 0 \
        --pairs pilot/countdown_pairs_balanced.parquet \
        $A --out rl/sparse_${TAG}_v4tpl.jsonl || exit 1
done

python pilot/probe_adjudication.py --template v4 \
    --parquet pilot/countdown_pairs_balanced.parquet \
    --adapter "$CKPT/eval_adapter" --n_problems 300 \
    --out pilot/probe_v4tpl_post.jsonl || exit 1
echo "V4 EVAL BATTERY DONE"
