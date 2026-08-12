#!/bin/bash
#SBATCH --job-name=opsd_harvest8
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Full-node harvest: 8 data shards, one per GPU, all inside one job (jobs run
# in sequence on this account, so a single 8-GPU job beats an 8-job array).
# Parameterized:
#   sbatch --export=ALL,H_DATASET=data/raw/deepmath_band.parquet,H_NSAMP=3,H_OUTDIR=harvest/stage1 \
#       pilot/harvest_stage.sbatch
# Shard 0's built-in supervisor merges all shard files at the end.

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd
source slurm/cluster_env.sh
source envs/train/.venv/bin/activate

H_DATASET=${H_DATASET:?H_DATASET required}
H_NSAMP=${H_NSAMP:?H_NSAMP required}
H_OUTDIR=${H_OUTDIR:?H_OUTDIR required}
H_MODEL=${H_MODEL:-Qwen3-4B}

mkdir -p "${H_OUTDIR}/$(dirname "${H_DATASET}")"

pids=()
for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${shard} python run_generate.py \
        --model_name /scratch/gpfs/ARORA/skaur/models/${H_MODEL} \
        --dataset "${H_DATASET}" \
        --output_dir "${H_OUTDIR}" \
        --batch_size 64 \
        --max_tokens 16384 \
        --temperature 1.0 --top_p 1.0 --top_k 50 \
        --num_generation "${H_NSAMP}" \
        --num_shards 8 --shard_id ${shard} \
        > "${H_OUTDIR}/shard_${shard}.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done

# Shard 0's supervisor gives up polling after ~5.5h of shard skew; guarantee the
# merge here so "DONE" always implies a merged JSONL exists. (H_DATASET must be
# a relative path — an absolute one would make run_generate write next to the
# parquet instead of into H_OUTDIR.)
MERGED="${H_OUTDIR}/${H_DATASET}-temp_1.0-top_p_1.0.jsonl"
if [ ! -f "${MERGED}" ]; then
    python run_generate.py --model_name /scratch/gpfs/ARORA/skaur/models/${H_MODEL} \
        --dataset "${H_DATASET}" --output_dir "${H_OUTDIR}" \
        --temperature 1.0 --top_p 1.0 \
        --num_shards 8 --shard_id 0 --merge_only || rc=1
fi
if [ $rc -eq 0 ] && [ -f "${MERGED}" ]; then
    echo "HARVEST STAGE DONE"
else
    echo "HARVEST STAGE FAILED (see shard logs)"
    rc=1
fi
exit $rc
