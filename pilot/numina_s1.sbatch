#!/bin/bash
#SBATCH --job-name=numina_s1
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# NuminaMath harvest, stage 1: generate k=3 -> merge -> grade -> screen.
#
# DESIGN CONSTRAINTS, both learned the hard way:
#  * della cancels a job whose GPUs idle 90 min, and it kills long-lived idle
#    processes generally -- so a login-node "conductor" that sleeps while waiting
#    is not viable either. Everything therefore lives inside SLURM jobs, and the
#    CPU stages run with pilot/gpu_keepalive.py holding GPU utilization up.
#  * grading must NOT run inside a vLLM worker: math_equal forks to implement its
#    timeout, and forking a CUDA-context-holding process deadlocked shard 0 of job
#    12413497 (5h of an 8-GPU node lost). Generation uses --no_grade; grading is a
#    separate CPU pass here, after all GPU work has exited.
# Every stage is skipped if its output exists, and generation resumes from
# .partial files, so resubmitting after any failure continues rather than restarts.

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || { echo "S1 FAILED: cluster_env"; exit 1; }
source envs/train/.venv/bin/activate || { echo "S1 FAILED: venv"; exit 1; }

BAND=data/raw/numina_band.parquet
OUT=harvest/numina_s1
RAW=${OUT}/${BAND}-temp_1.0-top_p_1.0.jsonl
GRADED=${OUT}/graded.jsonl
SURV=data/raw/numina_survivors.parquet
MODEL=/scratch/gpfs/ARORA/skaur/models/Qwen3-4B

die() { echo "S1 FAILED: $1"; echo "$1" > harvest/NUMINA_S1_FAILED; exit 1; }

# ---- generate (GPU) ----------------------------------------------------------
if [ ! -f "${RAW}" ]; then
    mkdir -p "${OUT}/$(dirname "${BAND}")"
    pids=()
    for shard in 0 1 2 3 4 5 6 7; do
        # Each shard hands its GPU straight to a keepalive when it exits. Without
        # this, one crashed or early-finishing shard leaves a GPU at 0% while the
        # others run for hours -- the 90-min watchdog then kills all eight. That is
        # what happened to job 12413497 and it is independent of the fork deadlock
        # that triggered it.
        ( CUDA_VISIBLE_DEVICES=${shard} python run_generate.py \
              --model_name "${MODEL}" --dataset "${BAND}" --output_dir "${OUT}" \
              --batch_size 64 --max_tokens 32768 \
              --temperature 1.0 --top_p 1.0 --top_k 50 \
              --num_generation 3 --num_shards 8 --shard_id ${shard} \
              --no_grade --skip_merge \
              > "${OUT}/shard_${shard}.log" 2>&1
          echo $? > "${OUT}/shard_${shard}.rc"
          CUDA_VISIBLE_DEVICES=${shard} python pilot/gpu_keepalive.py \
              >> "${OUT}/shard_${shard}.log" 2>&1 ) &
        pids+=($!)
    done
    # Wait on the generation processes by their sentinel files, not on the
    # subshells (which never exit -- they end in a keepalive loop).
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
    python run_generate.py --model_name "${MODEL}" --dataset "${BAND}" \
        --output_dir "${OUT}" --temperature 1.0 --top_p 1.0 \
        --num_shards 8 --shard_id 0 --merge_only || die merge
    [ -f "${RAW}" ] || die "merge produced no file"
    # A shard that loaded no data exits 0 and writes no shard file; run_generate's
    # merge globs whatever exists, so the pool would silently shrink with no error.
    NSHARD=$(ls "${RAW}".shard_[0-7] 2>/dev/null | wc -l)
    [ "${NSHARD}" -eq 8 ] || die "merge saw ${NSHARD}/8 shard files"
fi
echo "stage 1 generations: $(wc -l < "${RAW}")"

# ---- grade + screen (CPU, GPUs held up by the keepalive) ---------------------
python pilot/gpu_keepalive.py > "${OUT}/keepalive.log" 2>&1 &
KEEP=$!
trap 'kill ${KEEP} 2>/dev/null' EXIT

python data_tools/grade_jsonl.py --in "${RAW}" --out "${GRADED}" --workers 32 || die grading
[ -f "${SURV}" ] || python pilot/screen_stage1.py "${GRADED}" "${BAND}" "${SURV}" || die screen

kill ${KEEP} 2>/dev/null
echo "S1 DONE"; echo DONE > harvest/NUMINA_S1_DONE
