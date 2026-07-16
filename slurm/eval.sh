#!/usr/bin/env bash
#SBATCH --job-name=eval_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A.out

# Compute accuracy / unbiased pass@k over a merged generations JSONL and write
# metrics.json next to it.

source slurm/config.sh
[ -f slurm/cluster_env.sh ] && source slurm/cluster_env.sh
source envs/train/.venv/bin/activate
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${CHECKPOINT_PATH}/eval/${EVAL_SET_NAME}"
FINAL_EVAL_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.jsonl" -not -name "*.shard_*" | head -n 1)

if [ ! -f "$FINAL_EVAL_FILE" ]; then
  echo "Error: Final merged evaluation file not found in '$OUTPUT_DIR'!"
  exit 1
fi

echo "[EVAL] Calculating metrics on: $FINAL_EVAL_FILE"
DO_REGRADE_FLAG=""
if [ "$DO_REGRADE" = "true" ]; then
    DO_REGRADE_FLAG="--regrade"
fi

python run_eval.py \
  --file_path "${FINAL_EVAL_FILE}" \
  $DO_REGRADE_FLAG

echo "Pipeline complete!"
