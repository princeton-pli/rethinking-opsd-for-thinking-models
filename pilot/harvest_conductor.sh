#!/usr/bin/env bash
# =============================================================================
# End-to-end NuminaMath harvest, run from a login-node tmux session.
#
#   tmux new-session -d -s conductor 'bash pilot/harvest_conductor.sh'
#
# WHY A CONDUCTOR RATHER THAN SLURM DEPENDENCIES. Two failure modes have now
# killed multi-hour runs, both from mixing CPU work into GPU jobs:
#   1. a CPU tail (pool build) inside a GPU job -> GPUs idle -> 90-min watchdog
#   2. inline grading inside a vLLM worker -> fork deadlock -> one shard stalls,
#      its GPU idles, the watchdog kills all 8 (job 12413497, 5h lost)
# So GPU jobs here do generation ONLY (--no_grade --skip_merge), and every CPU
# stage -- grading, screening, pool build -- runs on the login node between them.
# This script is the thing that sequences them, and it is idempotent: every stage
# is skipped if its output already exists, and generation resumes from .partial
# files, so it can be killed and restarted at any point.
# =============================================================================
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source envs/train/.venv/bin/activate

BAND=data/raw/numina_band.parquet
S1_DIR=harvest/numina_s1
S2_DIR=harvest/numina_s2
SURV=data/raw/numina_survivors.parquet
S1_RAW=${S1_DIR}/${BAND}-temp_1.0-top_p_1.0.jsonl
S1_GRADED=${S1_DIR}/graded.jsonl
S2_RAW=${S2_DIR}/${SURV}-temp_1.0-top_p_1.0.jsonl
S2_GRADED=${S2_DIR}/graded.jsonl
STATUS=harvest/CONDUCTOR_STATUS

say() { echo "[$(date +%H:%M)] $*" | tee -a "${STATUS}"; }
die() { say "FAILED: $*"; echo FAILED > harvest/CONDUCTOR_FAILED; exit 1; }

# wait_job <jobid>: block until the job leaves the queue, reporting periodically.
wait_job() {
    local jid=$1 n=0
    while squeue -j "${jid}" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|COMPLETING'; do
        sleep 300
        n=$((n+1))
        [ $((n % 12)) -eq 0 ] && say "  ... job ${jid} still going ($((n*5)) min)"
    done
    local state
    state=$(sacct -j "${jid}" -o State -n 2>/dev/null | head -1 | tr -d ' ')
    say "  job ${jid} finished: ${state}"
    case "${state}" in COMPLETED*) return 0 ;; *) return 1 ;; esac
}

# ---- Stage 1: generate (GPU) ------------------------------------------------
if [ ! -f "${S1_RAW}" ]; then
    say "stage 1: submitting generation (resumes from existing .partial files)"
    JID=$(sbatch --parsable \
        --export=ALL,H_DATASET=${BAND},H_NSAMP=3,H_OUTDIR=${S1_DIR},H_MAXTOK=32768 \
        pilot/harvest_stage.sbatch) || die "stage1 sbatch"
    say "stage 1 job ${JID}"
    wait_job "${JID}" || die "stage1 generation"
    [ -f "${S1_RAW}" ] || die "stage1 produced no merged jsonl"
else
    say "stage 1: merged jsonl already present, skipping generation"
fi

# ---- Stage 1: grade (CPU, login node) ---------------------------------------
say "stage 1: grading on CPU"
python data_tools/grade_jsonl.py --in "${S1_RAW}" --out "${S1_GRADED}" --workers 16 \
    >> harvest/grade_s1.log 2>&1 || die "stage1 grading"

# ---- Screen -----------------------------------------------------------------
if [ ! -f "${SURV}" ]; then
    say "screening out problems already all-correct at k=3"
    python pilot/screen_stage1.py "${S1_GRADED}" "${BAND}" "${SURV}" \
        >> harvest/screen.log 2>&1 || die "screen"
fi
say "survivors: $(python -c "import pandas as pd;print(len(pd.read_parquet('${SURV}')))")"

# ---- Stage 2: generate (GPU) ------------------------------------------------
if [ ! -f "${S2_RAW}" ]; then
    say "stage 2: submitting top-up generation (k=3 -> k=8)"
    JID=$(sbatch --parsable \
        --export=ALL,H_DATASET=${SURV},H_NSAMP=5,H_OUTDIR=${S2_DIR},H_MAXTOK=32768 \
        pilot/harvest_stage.sbatch) || die "stage2 sbatch"
    say "stage 2 job ${JID}"
    wait_job "${JID}" || die "stage2 generation"
    [ -f "${S2_RAW}" ] || die "stage2 produced no merged jsonl"
else
    say "stage 2: merged jsonl already present, skipping generation"
fi

# ---- Stage 2: grade (CPU) ---------------------------------------------------
say "stage 2: grading on CPU"
python data_tools/grade_jsonl.py --in "${S2_RAW}" --out "${S2_GRADED}" --workers 16 \
    >> harvest/grade_s2.log 2>&1 || die "stage2 grading"

# ---- Pool + Arm-1 dataset (CPU) ---------------------------------------------
say "building x+/x- pool"
python data_tools/build_pool.py --out harvest/pool_numina.jsonl "${S1_GRADED}" "${S2_GRADED}" \
    >> harvest/pool.log 2>&1 || die "build_pool"
say "rendering Arm-1 dataset"
python data_tools/build_contrastive_dataset.py --pool harvest/pool_numina.jsonl \
    --output data/raw/contrastive_arm1_numina.parquet >> harvest/pool.log 2>&1 || die "build_dataset"

say "CONDUCTOR DONE"
echo DONE > harvest/CONDUCTOR_DONE
