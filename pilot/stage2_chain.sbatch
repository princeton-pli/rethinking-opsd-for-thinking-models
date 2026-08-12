#!/bin/bash
#SBATCH --job-name=opsd_stage2
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Stage-2 chain, submitted with --dependency=afterok:<stage1 job> so the whole
# harvest completes unattended:
#   1. screen stage-1 (drop all-correct-at-k=3 problems)
#   2. 8-shard generation: +5 samples on survivors (k=3 banked + 5 = 8 total)
#   3. build the relaxed x+/x- pool from stage-1 + stage-2 JSONLs
#   4. render the Arm-1 training parquet (unlabeled two-response contexts)
# Sentinel: harvest/PIPELINE_DONE (or PIPELINE_FAILED with the failing step).

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd
source slurm/cluster_env.sh
source envs/train/.venv/bin/activate

BAND=data/raw/deepmath_band_7p5_9.parquet
STAGE1_JSONL=harvest/stage1/${BAND}-temp_1.0-top_p_1.0.jsonl
SURV=data/raw/deepmath_stage2_survivors.parquet
STAGE2_OUT=harvest/stage2
STAGE2_JSONL=${STAGE2_OUT}/${SURV}-temp_1.0-top_p_1.0.jsonl
H_MODEL=Qwen3-4B

fail() { echo "PIPELINE FAILED: $1" | tee harvest/PIPELINE_FAILED; exit 1; }

python pilot/screen_stage1.py "${STAGE1_JSONL}" "${BAND}" "${SURV}" || fail screen

mkdir -p "${STAGE2_OUT}/$(dirname "${SURV}")"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${shard} python run_generate.py \
        --model_name /scratch/gpfs/ARORA/skaur/models/${H_MODEL} \
        --dataset "${SURV}" \
        --output_dir "${STAGE2_OUT}" \
        --batch_size 64 \
        --max_tokens 16384 \
        --temperature 1.0 --top_p 1.0 --top_k 50 \
        --num_generation 5 \
        --num_shards 8 --shard_id ${shard} \
        > "${STAGE2_OUT}/shard_${shard}.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ ! -f "${STAGE2_JSONL}" ]; then
    python run_generate.py --model_name /scratch/gpfs/ARORA/skaur/models/${H_MODEL} \
        --dataset "${SURV}" --output_dir "${STAGE2_OUT}" \
        --temperature 1.0 --top_p 1.0 --num_shards 8 --shard_id 0 --merge_only || rc=1
fi
[ $rc -eq 0 ] && [ -f "${STAGE2_JSONL}" ] || fail stage2_generation

python data_tools/build_pool.py --out harvest/pool_deepmath_7p5_9.jsonl \
    "${STAGE1_JSONL}" "${STAGE2_JSONL}" || fail build_pool

python data_tools/build_contrastive_dataset.py \
    --pool harvest/pool_deepmath_7p5_9.jsonl \
    --output data/raw/contrastive_arm1_deepmath.parquet || fail build_dataset

echo "PIPELINE DONE" | tee harvest/PIPELINE_DONE
