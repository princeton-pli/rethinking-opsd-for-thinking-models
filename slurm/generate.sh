#!/usr/bin/env bash
#SBATCH --job-name=generate_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-7    # <<< must match NUM_SHARDS (default 8)
#SBATCH --output=logs/%x-%A-%a.out

# Sharded vLLM generation worker. Each array task handles the round-robin
# shard `question_id % NUM_SHARDS == SLURM_ARRAY_TASK_ID`, writes
# <output>.shard_<id> (resuming from a .partial file if present), and array
# task 0 merges all shards into the final JSONL.

# --- Configuration & environment setup --------------------------------------
source slurm/config.sh
[ -f slurm/cluster_env.sh ] && source slurm/cluster_env.sh
source envs/train/.venv/bin/activate
export PYTHONUNBUFFERED=1

# --- Execution ---------------------------------------------------------------
if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint directory '$CHECKPOINT_PATH' does not exist!"
  exit 1
fi
OUTPUT_DIR="${CHECKPOINT_PATH}/eval/${EVAL_SET_NAME}${EVAL_DIR_SUFFIX:-}"
mkdir -p logs "$OUTPUT_DIR"

if [ "$NUM_SHARDS" -ne "$SLURM_ARRAY_TASK_COUNT" ] && [ "${ALLOW_SHARD_SUBSET:-false}" != "true" ]; then
  echo "Error: NUM_SHARDS ($NUM_SHARDS) does not match SBATCH --array count ($SLURM_ARRAY_TASK_COUNT)!"
  echo "Set ALLOW_SHARD_SUBSET=true to bypass (for partial-shard reruns)."
  exit 1
fi

echo "[EVAL] Worker: $SLURM_ARRAY_TASK_ID / $NUM_SHARDS. Running inference on $EVAL_SET_NAME."

R1_FLAG=""
if [ "$USE_R1_STYLE_PROMPT" = "true" ]; then
    R1_FLAG="--use_r1_style_prompt"
fi

THINKING_FLAG=""
if [ -n "${ENABLE_THINKING:-}" ] && [ "${ENABLE_THINKING}" != "default" ]; then
    THINKING_FLAG="--enable_thinking ${ENABLE_THINKING}"
fi

MERGE_FLAG=""
if [ "${SKIP_GENERATE_MERGE:-false}" = "true" ]; then
    MERGE_FLAG="--skip_merge"
fi

python run_generate.py \
  --model_name "${CHECKPOINT_PATH}" \
  --dataset "${EVAL_SET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_generation "${N_SAMPLES}" \
  --temperature "${TEMP}" \
  --top_p "${TOP_P}" \
  --max_tokens "${EVAL_MAX_TOKENS}" \
  --num_gpus "${TENSOR_PARALLEL}" \
  --shard_id "${SLURM_ARRAY_TASK_ID}" \
  --num_shards "${NUM_SHARDS}" \
  $R1_FLAG \
  $THINKING_FLAG \
  $MERGE_FLAG

# --- Pipeline chaining (array worker 0 only) ---------------------------------
# When launched from generate_with_retry.sh, SKIP_EVAL_CHAIN is set and
# the wrapper submits the metrics job itself (with retry awareness).
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ "${SKIP_EVAL_CHAIN:-false}" != "true" ]; then
  GENERATE_JOB_ID="${SLURM_ARRAY_JOB_ID}"
  echo "[EVAL] Submitting metrics job (dependent on all array tasks in $GENERATE_JOB_ID)."
  sbatch \
      --dependency=afterok:"$GENERATE_JOB_ID" \
      --export=ALL \
      slurm/eval.sh
fi
