#!/usr/bin/env bash
#SBATCH --job-name=generate_eval_retry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A.out

# =============================================================================
# Eval generate wrapper with retry.
#
# Lightweight CPU-only orchestrator that submits the GPU generate array job
# (generate.sh), waits for it, checks for the merged JSONL, and retries
# up to MAX_TRIES times on failure. On success it submits the eval metrics
# job (eval.sh). Generate workers resume from .partial shard files, so
# retries only redo unfinished work.
#
# Required env vars: CHECKPOINT_PATH, EVAL_SET_NAME, NUM_SHARDS, EVAL_MAX_TOKENS
# Optional: MAX_TRIES (default 3), SHARD_TIME (per-attempt --time for workers)
# =============================================================================

source slurm/config.sh
[ -f slurm/cluster_env.sh ] && source slurm/cluster_env.sh

MAX_TRIES="${MAX_TRIES:-3}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint directory '$CHECKPOINT_PATH' does not exist!"
  exit 1
fi

OUTPUT_DIR="${CHECKPOINT_PATH}/eval/${EVAL_SET_NAME}"
mkdir -p logs "$OUTPUT_DIR"

echo "[RETRY WRAPPER] CHECKPOINT_PATH : ${CHECKPOINT_PATH}"
echo "[RETRY WRAPPER] EVAL_SET_NAME   : ${EVAL_SET_NAME}"
echo "[RETRY WRAPPER] NUM_SHARDS      : ${NUM_SHARDS}"
echo "[RETRY WRAPPER] OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "[RETRY WRAPPER] MAX_TRIES       : ${MAX_TRIES}"

ATTEMPT=1
JSONL_COMPLETED="false"

while [ "$ATTEMPT" -le "$MAX_TRIES" ]; do
    echo "[RETRY WRAPPER] Submitting generate array job (Attempt ${ATTEMPT}/${MAX_TRIES})..."

    SHARD_TIME_ARG=""
    [ -n "${SHARD_TIME:-}" ] && SHARD_TIME_ARG="--time=${SHARD_TIME}"
    JOB_ID=$(sbatch --wait --parsable \
        --array=0-$((NUM_SHARDS-1)) \
        ${SHARD_TIME_ARG} \
        --export=ALL,SKIP_EVAL_CHAIN=true \
        slurm/generate.sh)

    echo "[RETRY WRAPPER] Array job ${JOB_ID} finished."

    # Locate the merged JSONL the same way eval.sh does (the filename
    # encodes temperature/top_p, so don't reconstruct it).
    FINAL_JSONL=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.jsonl" -not -name "*.shard_*" -not -name "*.partial" | head -n 1)

    if [ -f "${FINAL_JSONL}" ]; then
        echo "[RETRY WRAPPER] Merged JSONL found: ${FINAL_JSONL}"
        JSONL_COMPLETED="true"
        break
    else
        echo "[RETRY WRAPPER] Merged JSONL not found in ${OUTPUT_DIR}. Retrying..."
        ATTEMPT=$((ATTEMPT + 1))
        sleep 10
    fi
done

if [ "${JSONL_COMPLETED}" != "true" ]; then
    echo "[RETRY WRAPPER] FATAL: Generation failed after ${MAX_TRIES} attempts."
    exit 1
fi

echo "[RETRY WRAPPER] Submitting eval metrics job..."
sbatch --export=ALL slurm/eval.sh
echo "[RETRY WRAPPER] Done."
