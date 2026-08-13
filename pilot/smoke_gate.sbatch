#!/bin/bash
#SBATCH --job-name=opsd_smoke
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Gated-trainer smoke test at the REAL Arm-1 shape: 8 GPUs.
#
# History: 1 GPU broke the stock FSDP->vLLM weight sync (single-GPU FSDP is
# degenerate; "Non-root FSDP instance's _is_root" assertion, job 12339899). At
# 2 GPUs every rung OOM'd -- but 2-rank FSDP carries ~15-18 GB/GPU MORE model
# state than 8-rank, which is exactly the headroom the loss region needs, so
# 2 GPUs was too strict a bar rather than a conservative one. 8 GPUs is both the
# real configuration and the only one whose verdict transfers directly.
#
# accum=2 (not 64) makes each step ~32x cheaper without changing peak memory:
# gradients accumulate into the same buffers and per-micro-batch tensors are
# freed. Effective batch here is 8x1x2=16, so 3 steps needs >=48 problems.
#
# Because della jobs queue sequentially, this walks a LADDER inside one
# allocation instead of failing and re-queuing:
# GENERATION stays at 16384 on every rung -- the gate needs a final answer, and
# at a 4096 cap 95% of rollouts on this data never reach one. What varies is the
# LOSS length, which is what actually drives memory:
#   A  loss 4096, jsd_chunk 1024  <- paper's exact loss-token budget; want this
#   B  loss 6144, jsd_chunk 1024  <- more supervised tokens if A has headroom
#   C  loss 4096, jsd_chunk  512  <- halve the in-checkpoint backward peak
#   D  loss 2048, jsd_chunk 1024  <- fallback
# First rung that survives 3 optimizer steps wins; its name lands in SMOKE_RESULT.
# What we read off a surviving rung: peak GPU memory, gate/live_fraction,
# gate/regen_rounds, and step wall-time (for sizing the Arm-1 job).

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh
source envs/train/.venv/bin/activate

export WANDB_MODE=disabled WANDB_DISABLED=true
RESULT=pilot/SMOKE_RESULT
: > "${RESULT}"

run_rung() {  # run_rung <name> <gen_cap> <vllm_mem_util> <sleep_mode> <jsd_chunk> <loss_tokens>
    local name=$1 cap=$2 mem=$3 sleep_mode=$4 chunk=$5 loss_tokens=$6
    local log=pilot/smoke_${name}.log
    echo "=== RUNG ${name}: gen_cap=${cap} loss_tokens=${loss_tokens} vllm_mem=${mem} sleep=${sleep_mode} jsd_chunk=${chunk} ==="
    # Sample GPU memory in the background so a surviving rung reports its peak.
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 10; done \
        > "pilot/smoke_${name}.mem" 2>/dev/null ) &
    local memsampler=$!
    # A fresh checkpoint dir per rung: train.sh skips training if one already exists.
    rm -rf "checkpoints/smoke_${name}"
    MODE=qwen3_opsd \
    BASE_MODEL=/scratch/gpfs/ARORA/skaur/models/Qwen3-4B \
    TRAIN_PATH=data/raw/contrastive_smoke.parquet \
    DATASET_NAME="smoke_${name}" \
    PROMPT_KEY=problem \
    GOLD_ANSWER_KEY=teacher_context \
    TEACHER_PROMPT_TEMPLATE='{gold_answer}' \
    GATE_MODE=wrong_only \
    GATE_REQUIRE_DIFF_ANSWER=False \
    GATE_GOLD_ANSWER_KEY=answer \
    GATE_WRONG_ANSWER_KEY=wrong_answer \
    GATE_MAX_REGEN_ROUNDS=2 \
    OPSD_MAX_PROMPT_LENGTH=4096 \
    OPSD_MAX_COMPLETION_LENGTH="${cap}" \
    OPSD_MICRO_BATCH_SIZE=1 \
    OPSD_GRADIENT_ACCUM=2 \
    OPSD_LR=5e-6 \
    OPSD_EPOCHS=1 \
    VLLM_GPU_MEM_UTIL="${mem}" \
    VLLM_SLEEP_MODE="${sleep_mode}" \
    JSD_CHUNK_SIZE="${chunk}" \
    LOSS_MAX_COMPLETION_TOKENS="${loss_tokens}" \
    OPSD_REPORT_TO=none \
    MAX_STEPS=3 \
        bash slurm/train.sh > "${log}" 2>&1
    local rc=$?
    kill "${memsampler}" 2>/dev/null
    local peak
    peak=$(sort -n "pilot/smoke_${name}.mem" 2>/dev/null | tail -1)
    echo "RUNG ${name} peak GPU memory: ${peak:-?} MiB / 81559 MiB" | tee -a "${RESULT}"
    grep -hoE "gate/(live_fraction|initial_live_fraction|regen_rounds)[^,}]*" "${log}" 2>/dev/null | tail -6 | tee -a "${RESULT}"
    if [ $rc -eq 0 ]; then
        echo "RUNG ${name} PASSED" | tee -a "${RESULT}"
        grep -iE "torch.OutOfMemory|CUDA out of memory" "${log}" >/dev/null && \
            { echo "  (but log mentions OOM - inspect)" | tee -a "${RESULT}"; }
        return 0
    fi
    if grep -iqE "torch.OutOfMemory|CUDA out of memory|NCCL.*out of memory" "${log}"; then
        echo "RUNG ${name} OOM" | tee -a "${RESULT}"
        return 1
    fi
    echo "RUNG ${name} FAILED (non-OOM, rc=${rc}) - stopping ladder" | tee -a "${RESULT}"
    tail -30 "${log}" | tee -a "${RESULT}"
    return 2
}

for rung in "A 16384 0.30 True 1024 4096" "B 16384 0.20 True 1024 6144" "C 16384 0.20 True 512 4096" "D 16384 0.20 True 1024 2048"; do
    # shellcheck disable=SC2086
    run_rung $rung
    rc=$?
    [ $rc -eq 0 ] && { echo "LADDER WINNER: ${rung}" | tee -a "${RESULT}"; break; }
    [ $rc -eq 2 ] && break   # non-OOM failure: a lower rung will not help
done

echo "SMOKE DONE" | tee -a "${RESULT}"
