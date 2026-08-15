#!/bin/bash
#SBATCH --job-name=numina_stage2
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Stage 2 of the NuminaMath (aops_forum + cn_contest) harvest, chained off stage 1
# with --dependency=afterok. Screens out problems already all-correct at k=3, then
# tops survivors up from k=3 to k=8 at generation cap 32768.
#
# The pool build is deliberately NOT here: it is a ~1h CPU job (math_equal over
# ~150k samples) and della AUTO-CANCELS any job whose GPUs sit at 0% for 90 min.
# A login-node tmux watcher does that step instead.

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh
source envs/train/.venv/bin/activate

BAND=data/raw/numina_band.parquet
STAGE1_JSONL=harvest/numina_s1/${BAND}-temp_1.0-top_p_1.0.jsonl
SURV=data/raw/numina_survivors.parquet
STAGE2_OUT=harvest/numina_s2
STAGE2_JSONL=${STAGE2_OUT}/${SURV}-temp_1.0-top_p_1.0.jsonl
H_MODEL=Qwen3-4B
H_MAXTOK=32768

fail() { echo "STAGE2 FAILED: $1" | tee harvest/NUMINA_STAGE2_FAILED; exit 1; }

python pilot/screen_stage1.py "${STAGE1_JSONL}" "${BAND}" "${SURV}" || fail screen

mkdir -p "${STAGE2_OUT}/$(dirname "${SURV}")"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${shard} python run_generate.py \
        --model_name /scratch/gpfs/ARORA/skaur/models/${H_MODEL} \
        --dataset "${SURV}" \
        --output_dir "${STAGE2_OUT}" \
        --batch_size 64 \
        --max_tokens "${H_MAXTOK}" \
        --temperature 1.0 --top_p 1.0 --top_k 50 \
        --num_generation 5 \
        --num_shards 8 --shard_id ${shard} --skip_merge \
        > "${STAGE2_OUT}/shard_${shard}.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

python run_generate.py --model_name /scratch/gpfs/ARORA/skaur/models/${H_MODEL} \
    --dataset "${SURV}" --output_dir "${STAGE2_OUT}" \
    --temperature 1.0 --top_p 1.0 --num_shards 8 --shard_id 0 --merge_only || rc=1

[ $rc -eq 0 ] && [ -f "${STAGE2_JSONL}" ] || fail stage2_generation
echo "STAGE2 DONE" | tee harvest/NUMINA_STAGE2_DONE
