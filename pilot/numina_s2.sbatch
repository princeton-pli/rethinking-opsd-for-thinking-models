#!/bin/bash
#SBATCH --job-name=numina_s2
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# NuminaMath harvest, stage 2: top survivors from k=3 to k=8, grade, build the
# x+/x- pool and the Arm-1 training parquet. Chained off stage 1 with
# --dependency=afterok. Same constraints as numina_s1.sbatch: generation is
# --no_grade --skip_merge, all CPU stages run under pilot/gpu_keepalive.py, and
# every stage is skipped if its output already exists.

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || { echo "S2 FAILED: cluster_env"; exit 1; }
source envs/train/.venv/bin/activate || { echo "S2 FAILED: venv"; exit 1; }

SURV=data/raw/numina_survivors.parquet
OUT=harvest/numina_s2
RAW=${OUT}/${SURV}-temp_1.0-top_p_1.0.jsonl
GRADED=${OUT}/graded.jsonl
S1_GRADED=harvest/numina_s1/graded.jsonl
MODEL=/scratch/gpfs/ARORA/skaur/models/Qwen3-4B

die() { echo "S2 FAILED: $1"; echo "$1" > harvest/NUMINA_S2_FAILED; exit 1; }

[ -f "${SURV}" ] || die "no survivors parquet -- stage 1 did not finish"

# ---- generate (GPU) ----------------------------------------------------------
if [ ! -f "${RAW}" ]; then
    mkdir -p "${OUT}/$(dirname "${SURV}")"
    for shard in 0 1 2 3 4 5 6 7; do
        # See numina_s1.sbatch: each shard hands its GPU to a keepalive on exit so
        # a dead or early shard cannot idle a GPU past the 90-min watchdog.
        ( CUDA_VISIBLE_DEVICES=${shard} python run_generate.py \
              --model_name "${MODEL}" --dataset "${SURV}" --output_dir "${OUT}" \
              --batch_size 64 --max_tokens 32768 \
              --temperature 1.0 --top_p 1.0 --top_k 50 \
              --num_generation 5 --num_shards 8 --shard_id ${shard} \
              --no_grade --skip_merge \
              > "${OUT}/shard_${shard}.log" 2>&1
          echo $? > "${OUT}/shard_${shard}.rc"
          CUDA_VISIBLE_DEVICES=${shard} python pilot/gpu_keepalive.py \
              >> "${OUT}/shard_${shard}.log" 2>&1 ) &
    done
    while :; do
        n=$(ls "${OUT}"/shard_*.rc 2>/dev/null | wc -l)
        [ "$n" -eq 8 ] && break
        sleep 60
    done
    pkill -f gpu_keepalive.py
    rc=0
    for shard in 0 1 2 3 4 5 6 7; do
        [ "$(cat "${OUT}/shard_${shard}.rc")" -eq 0 ] || rc=1
    done
    [ $rc -eq 0 ] || die generation
    python run_generate.py --model_name "${MODEL}" --dataset "${SURV}" \
        --output_dir "${OUT}" --temperature 1.0 --top_p 1.0 \
        --num_shards 8 --shard_id 0 --merge_only || die merge
    [ -f "${RAW}" ] || die "merge produced no file"
    NSHARD=$(ls "${RAW}".shard_[0-7] 2>/dev/null | wc -l)
    [ "${NSHARD}" -eq 8 ] || die "merge saw ${NSHARD}/8 shard files"
fi
echo "stage 2 generations: $(wc -l < "${RAW}")"

# ---- grade + pool + dataset (CPU, GPUs held up by the keepalive) -------------
python pilot/gpu_keepalive.py > "${OUT}/keepalive.log" 2>&1 &
KEEP=$!
trap 'kill ${KEEP} 2>/dev/null' EXIT

python data_tools/grade_jsonl.py --in "${RAW}" --out "${GRADED}" --workers 32 || die grading
python data_tools/build_pool.py --out harvest/pool_numina.jsonl \
    "${S1_GRADED}" "${GRADED}" || die build_pool
python data_tools/build_contrastive_dataset.py --pool harvest/pool_numina.jsonl \
    --output data/raw/contrastive_arm1_numina.parquet || die build_dataset

kill ${KEEP} 2>/dev/null
echo "S2 DONE"; echo DONE > harvest/NUMINA_S2_DONE
