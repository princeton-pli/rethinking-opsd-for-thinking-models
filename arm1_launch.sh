#!/usr/bin/env bash
# =============================================================================
# Launch contrastive-OPSD Arm 1 (and its dense-gold control) + the paper's eval
# battery. Mirrors submit_experiment.sh's structure so results drop into the same
# Table-1 / Figure-1 coordinates.
#
#   ./arm1_launch.sh arm1        # contrastive teacher + wrong-only gate
#   ./arm1_launch.sh baseline    # dense gold teacher, no gate (control)
#   DRY_RUN=1 ./arm1_launch.sh arm1
#
# Config comes from the 2026-08-14 smoke ladder, which passed at rung A with a
# 38.9/81.6 GB peak: generation 16384 (the gate needs a gradeable final answer --
# at a 4096 generation cap 95% of rollouts on this data never reach \boxed{}),
# loss on the first 4096 tokens (the paper's exact loss-token budget), JSD chunked
# at 1024.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

NAME=${1:-}
DRY_RUN=${DRY_RUN:-0}
MODELS_DIR=${MODELS_DIR:-/scratch/gpfs/ARORA/skaur/models}
MODEL=${MODEL:-Qwen3-4B}
DATA=${DATA:-data/raw/contrastive_arm1_numina.parquet}
MATH_EVALS="aime24_38912 aime25_38912 hmmt25_38912"

# Shared with the paper: alpha 0.5 JSD, frozen self-teacher, 1 epoch, effective
# batch 64 (8 GPUs x micro 1 x accum 8), cosine schedule, bf16, FSDP, seed 42.
COMMON="MODE=qwen3_opsd,BASE_MODEL=${MODELS_DIR}/${MODEL},TRAIN_PATH=${DATA}"
COMMON+=",OPSD_ALPHA=0.5,OPSD_LR=5e-6,OPSD_EPOCHS=1"
COMMON+=",OPSD_MICRO_BATCH_SIZE=1,OPSD_GRADIENT_ACCUM=8"
COMMON+=",OPSD_MAX_PROMPT_LENGTH=4096,OPSD_MAX_COMPLETION_LENGTH=16384"
COMMON+=",LOSS_MAX_COMPLETION_TOKENS=4096,JSD_CHUNK_SIZE=1024"
COMMON+=",OPSD_SYNC_REF_MODEL=False,PROMPT_KEY=problem"

case "$NAME" in
    arm1)
        # Teacher context is pre-rendered into the teacher_context column by
        # build_contrastive_dataset.py (problem + two UNLABELED student responses,
        # one correct one wrong, order randomized), so the template is a passthrough.
        EXPORTS="${COMMON},GOLD_ANSWER_KEY=teacher_context,DATASET_NAME=numina_contrastive"
        EXPORTS+=",GATE_MODE=wrong_only,GATE_REQUIRE_DIFF_ANSWER=False"
        EXPORTS+=",GATE_GOLD_ANSWER_KEY=answer,GATE_WRONG_ANSWER_KEY=wrong_answer"
        EXPORTS+=",GATE_MAX_REGEN_ROUNDS=3"
        export TEACHER_PROMPT_TEMPLATE='{gold_answer}'
        EXPORTS+=",TEACHER_PROMPT_TEMPLATE"
        CKPT="checkpoints/opsd__m_${MODEL}__d_numina_contrastive__alpha0.5__bs64__lr5e-6__ep1__gate_wrong_only"
        ;;
    baseline)
        # Paper's dense arm on the SAME problems: gold solution in the teacher
        # context, default template, loss on every rollout. This is the control the
        # Arm-1 delta is measured against; running it ourselves is what replaces the
        # Table-1 comparability we gave up by leaving OpenThoughts.
        EXPORTS="${COMMON},GOLD_ANSWER_KEY=gold_solution,DATASET_NAME=numina_densegold"
        CKPT="checkpoints/opsd__m_${MODEL}__d_numina_densegold__alpha0.5__bs64__lr5e-6__ep1"
        ;;
    *)
        echo "Usage: $0 {arm1|baseline}"; exit 1 ;;
esac

echo "== ${NAME}"
echo "   data=${DATA}  ckpt=${CKPT}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] sbatch --gres=gpu:h100:8 --export=${EXPORTS} slurm/train.sh"
    echo "[dry-run] evals: ${MATH_EVALS} @ 38912, 16 samples, 8 shards, dep afterok"
    exit 0
fi

TRAIN_ID=$(sbatch --parsable --gres=gpu:h100:8 --mem-per-gpu=80G --cpus-per-gpu=12 \
    --time=20:00:00 --export="${EXPORTS}" slurm/train.sh)
echo "   training job: ${TRAIN_ID}"

for eval_name in $MATH_EVALS; do
    sbatch --dependency=afterok:"${TRAIN_ID}" \
        --export=ALL,EVAL_SET_NAME=${eval_name},EVAL_MAX_TOKENS=38912,NUM_SHARDS=8,CHECKPOINT_PATH="${CKPT}",TEMP=0.6,N_SAMPLES=16,TOP_P=0.95,EVAL_BATCH_SIZE=8 \
        slurm/generate_with_retry.sh > /dev/null
done
echo "   evals submitted (dep afterok): ${MATH_EVALS} @ T=0.6, top_p=0.95, n=16, 8 shards"
unset TEACHER_PROMPT_TEMPLATE
