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

# run_generate.py embeds the --dataset path in its output filename, so the
# nested dir must exist under output_dir before the shard writer opens it.
mkdir -p pilot/data/raw

python run_generate.py \
    --model_name /scratch/gpfs/ARORA/skaur/models/Qwen3-4B \
    --dataset data/raw/ot15k_pilot512.parquet \
    --output_dir pilot \
    --batch_size 64 \
    --max_tokens 16384 \
    --temperature 1.0 --top_p 1.0 --top_k 50 \
    --num_generation 8 \
    --num_shards 1 --shard_id 0

echo "HARVEST DONE"
