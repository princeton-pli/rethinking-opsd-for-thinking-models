#!/bin/bash
#SBATCH --job-name=opsd_smoke
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Gated-trainer smoke test on TWO GPUs. Two, not one: single-GPU FSDP is a
# degenerate config the paper never ran, and it breaks the stock FSDP->vLLM weight
# sync (_sync_fsdp1_params_to_vllm summons submodules post-order before the root
# is lazily initialized -> "Non-root FSDP instance's _is_root" assertion, seen
# 2026-08-13 job 12339899). Two ranks exercise real sharding.
#
# Two GPUs is also the STRICTER memory bar: peak = per-micro-batch tensors (the
# full-vocab logits/JSD region, identical at any GPU count since micro-batch is 1)
# + FSDP-sharded model state, and that state shards by 2 here vs by 8 in the real
# Arm-1 job -- roughly +18 GB/GPU. So a pass here implies a pass at 8 GPUs.
# accum=4 (not 64) makes each step ~16x cheaper without changing peak memory:
# gradients accumulate into the same buffers and per-micro-batch tensors are freed.
#
# Because della jobs queue sequentially, this walks a LADDER inside one
# allocation instead of failing and re-queuing:
#   A  cap 16384, stock config          <- the configuration we actually want
#   B  cap 16384, vLLM sleep + 0.20 mem <- config levers only
#   C  cap 12288, vLLM sleep + 0.20 mem <- shorter rollouts, some truncation tax
#   D  cap  8192, vLLM sleep + 0.20 mem <- paper-adjacent fallback
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

run_rung() {  # run_rung <name> <completion_cap> <vllm_mem_util> <sleep_mode>
    local name=$1 cap=$2 mem=$3 sleep_mode=$4
    local log=pilot/smoke_${name}.log
    echo "=== RUNG ${name}: cap=${cap} vllm_mem=${mem} sleep=${sleep_mode} ==="
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
    OPSD_GRADIENT_ACCUM=4 \
    OPSD_LR=5e-6 \
    OPSD_EPOCHS=1 \
    VLLM_GPU_MEM_UTIL="${mem}" \
    VLLM_SLEEP_MODE="${sleep_mode}" \
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

for rung in "A 16384 0.30 True" "B 16384 0.20 True" "C 12288 0.20 True" "D 8192 0.20 True"; do
    # shellcheck disable=SC2086
    run_rung $rung
    rc=$?
    [ $rc -eq 0 ] && { echo "LADDER WINNER: ${rung}" | tee -a "${RESULT}"; break; }
    [ $rc -eq 2 ] && break   # non-OOM failure: a lower rung will not help
done

echo "SMOKE DONE" | tee -a "${RESULT}"
