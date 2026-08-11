#!/bin/bash
#SBATCH --job-name=opsd_harvest
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Harvest pass: k=8 student samples per problem on the 512-problem pilot
# subset, Qwen3-4B, training-time sampling params (T=1.0, top-p 1.0, top-k 50).
# Cap 16384 so thinking traces can complete (training cap is 4096; here we
# want finished, gradeable solutions to serve as the x+/x- demo pool).
# Single shard, single GPU — della jobs run in sequence.

set -euo pipefail
cd /scratch/gpfs/ARORA/arora/opsd
source slurm/cluster_env.sh
source envs/train/.venv/bin/activate

# Model and dataset are overridable so the same script serves every pilot:
#   sbatch --export=ALL,PILOT_MODEL=Qwen3-1.7B pilot/pilot_gen.sbatch
PILOT_MODEL=${PILOT_MODEL:-Qwen3-4B}
PILOT_DATASET=${PILOT_DATASET:-data/raw/ot15k_pilot512.parquet}
OUT_DIR=pilot/${PILOT_MODEL}

# run_generate.py embeds the --dataset path in its output filename, so the
# nested dir must exist under output_dir before the shard writer opens it.
mkdir -p "${OUT_DIR}/$(dirname "${PILOT_DATASET}")"

python run_generate.py \
    --model_name /scratch/gpfs/ARORA/skaur/models/${PILOT_MODEL} \
    --dataset "${PILOT_DATASET}" \
    --output_dir "${OUT_DIR}" \
    --batch_size 64 \
    --max_tokens 16384 \
    --temperature 1.0 --top_p 1.0 --top_k 50 \
    --num_generation 8 \
    --num_shards 1 --shard_id 0

echo "HARVEST DONE"
