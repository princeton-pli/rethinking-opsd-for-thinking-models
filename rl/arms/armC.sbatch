#!/usr/bin/env bash
#SBATCH --job-name=opsd_armC_uniform_rlT
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=80G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%A.out
# =============================================================================
# ARM C — UNIFORM (all-ones) token weighting + RL'd (v4tpl grader) teacher.
# Identical to armA except the weight profile: isolates the tail-window
# effect (A vs C) at fixed teacher. 'uniform' runs the weighted code path at
# the FULL loss window -- note this differs from Arm 1 in BOTH teacher and
# loss window (Arm 1: self-teacher, loss on first 4096 tokens only), so C-vs-
# Arm-1 is not a single-variable comparison; A-vs-C is. Full rationale,
# sizing, and smoke protocol: rl/arms/armA.sbatch + rl/arms/REVIEW.md.
#
#   sbatch rl/arms/armC.sbatch
#   sbatch --time=01:00:00 --job-name=opsd_armC_smoke \
#          --export=ALL,SMOKE=1 rl/arms/armC.sbatch     # mandatory 1h smoke first
# =============================================================================
cd /scratch/gpfs/ARORA/arora/opsd || exit 1   # absolute: do not trust SLURM_SUBMIT_DIR

export MODELS_DIR=${MODELS_DIR:-/scratch/gpfs/ARORA/skaur/models}
export BASE_MODEL=${MODELS_DIR}/Qwen3-4B
export BASE_TOKENIZER=${BASE_MODEL}
export TRAIN_PATH=data/raw/countdown_opsd_train.parquet

# --- Arm identity ------------------------------------------------------------
export TOKEN_WEIGHT_MODE=uniform
export TEACHER_MODEL=rl/checkpoints/grader_grpo_lora_v4tpl/eval_adapter
export DATASET_NAME=cd_pairs_v4tpl

# --- Shared with Arm 1 / the paper -------------------------------------------
export MODE=qwen3_opsd
export OPSD_ALPHA=0.5 OPSD_LR=5e-6 OPSD_EPOCHS=1
export OPSD_MICRO_BATCH_SIZE=1 OPSD_GRADIENT_ACCUM=8
export OPSD_MAX_PROMPT_LENGTH=4096
export OPSD_MAX_COMPLETION_LENGTH=16384
export LOSS_MAX_COMPLETION_TOKENS=0
export JSD_CHUNK_SIZE=1024
export OPSD_SYNC_REF_MODEL=False
export PROMPT_KEY=problem
export GOLD_ANSWER_KEY=teacher_context
export TEACHER_PROMPT_TEMPLATE='{gold_answer}'
export GATE_MODE=wrong_only
export GATE_GRADER=countdown         # deterministic task grader for the gate
export GATE_REQUIRE_DIFF_ANSWER=False
export GATE_GOLD_ANSWER_KEY=answer
export GATE_WRONG_ANSWER_KEY=wrong_answer
export GATE_MAX_REGEN_ROUNDS=3
export WANDB_MODE=disabled WANDB_DISABLED=true

# --- Smoke variant -----------------------------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
    export MAX_STEPS=3
    export DATASET_NAME=${DATASET_NAME}_smoke
    mkdir -p logs
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 60 \
        > "logs/smoke_mem-${SLURM_JOB_ID}.csv" 2>&1 &
    SMOKE_MEM_PID=$!
    echo "[smoke] 3 steps, full-trace loss; GPU memory -> logs/smoke_mem-${SLURM_JOB_ID}.csv"
fi

bash slurm/train.sh
EXIT_CODE=$?
[ -n "${SMOKE_MEM_PID:-}" ] && kill "${SMOKE_MEM_PID}" 2>/dev/null
if [ "${SMOKE:-0}" = "1" ]; then
    echo "[smoke] peak memory.used per GPU (MiB):"
    awk -F', ' 'NR>1 && $2+0 > max[$1]+0 { max[$1]=$2+0 } END { for (g in max) print "  GPU " g ": " max[g] }' \
        "logs/smoke_mem-${SLURM_JOB_ID}.csv"
fi
exit $EXIT_CODE
