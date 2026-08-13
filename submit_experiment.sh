#!/usr/bin/env bash
# =============================================================================
# Submit a paper OPSD experiment by name: training + chained 38,912-token evals.
#
# Every run from the paper is one row in the case table below, with all
# hyperparameters explicit.
#
# Usage:
#   ./submit_experiment.sh list              # show all experiment names
#   ./submit_experiment.sh <name>            # submit one experiment
#   ./submit_experiment.sh openthoughts      # groups: openthoughts, sparse, countdown,
#                                            #   countdown_sparse, nothink, concise, all
#   DRY_RUN=1 ./submit_experiment.sh all     # print sbatch commands only
#
# Environment overrides: MODELS_DIR (default: models), GPU_TYPE (default:
# h100), QOS_ARGS (e.g. "--qos=pli-cp"). Mail/partition come from SBATCH_*
# vars (see slurm/cluster_env.sh.example).
# =============================================================================

cd "$(dirname "$0")" || exit 1

MODELS_DIR=${MODELS_DIR:-models}
GPU_TYPE=${GPU_TYPE:-h100}
QOS_ARGS=${QOS_ARGS:-}
DRY_RUN=${DRY_RUN:-0}

OT15K="data/raw/openthoughts_math_filtered_15k.parquet"
OT30K="data/raw/openthoughts_math_filtered_30k.parquet"
CD15K="data/raw/countdown_15k.parquet"
MATH_EVALS="aime24_38912 aime25_38912 hmmt25_38912"
CD_EVALS="countdown_38912 aime24_38912 aime25_38912 hmmt25_38912"
CONCISE_TEMPLATE='Solve the following math problem concisely and correctly. Be direct -- avoid unnecessary elaboration, redundant steps, or restating the problem. Focus only on the key reasoning steps needed to reach the answer.\n\n{prompt}'

# Dense OpenThoughts runs for the thinking models (+ sparse counterparts and
# the instruct-2507 / OLMo-instruct comparison runs trained the same way).
OT15K_MODELS="qwen3_1p7b qwen3_4b qwen3_8b qwen3_4b_thinking_2507 qwen3_4b_instruct_2507 olmo3_7b_think olmo3_7b_instruct"
OT_DENSE="qwen3_1p7b_ot15k_dense qwen3_4b_ot15k_dense qwen3_8b_ot15k_dense qwen3_4b_thinking_2507_ot15k_dense olmo3_7b_think_ot15k_dense"
CD_DENSE="qwen3_4b_thinking_2507_countdown_dense qwen3_4b_instruct_2507_countdown_dense olmo3_7b_think_countdown_dense olmo3_7b_instruct_countdown_dense"
OT30K_PAIR="qwen3_4b_ot30k_thinking qwen3_4b_ot30k_nothink"

all_names() {
    for m in $OT15K_MODELS; do echo "${m}_ot15k_dense"; echo "${m}_ot15k_sparse"; done
    echo "$CD_DENSE" | tr ' ' '\n'
    echo "$CD_DENSE" | tr ' ' '\n' | sed 's/_dense$/_sparse/'
    echo "$OT30K_PAIR" | tr ' ' '\n'
    echo "qwen3_8b_ot15k_concise"
}

run_sbatch() {  # run_sbatch <fake-id-label> <sbatch args...>
    local label="$1"; shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] sbatch $*" >&2
        echo "DRYRUN_${label}"
    else
        sbatch "$@"
    fi
}

submit_one() {
    local NAME="$1"

    # ---- defaults shared by every paper run --------------------------------
    local MODE=qwen3_opsd PROMPT_KEY=problem GOLD_KEY=solution
    local MPL=25000 GPUS=8 ACCUM=16 LR=5e-6 TIME=6:00:00
    local DATA_PARQUET=$OT15K DATASET_NAME=""
    local EVAL_LIST=$MATH_EVALS EVAL_TEMP=0.6 EVAL_TOP_P=0.95 EVAL_BS=8 SHARD_TIME=""
    local WANDB_OFF=0 NOTHINK=0 CONCISE=0

    # ---- model family ------------------------------------------------------
    local MODEL=""
    case "$NAME" in
        qwen3_1p7b_*)              MODEL=Qwen3-1.7B ;;
        qwen3_4b_thinking_2507_*)  MODEL=Qwen3-4B-Thinking-2507 ;;
        qwen3_4b_instruct_2507_*)  MODEL=Qwen3-4B-Instruct-2507; EVAL_TEMP=0.7; EVAL_TOP_P=0.8 ;;
        qwen3_4b_*)                MODEL=Qwen3-4B ;;
        qwen3_8b_*)                MODEL=Qwen3-8B; LR=2e-6; EVAL_BS=4; TIME=8:00:00 ;;
        olmo3_7b_think_*)          MODEL=Olmo-3-7B-Think; MODE=olmo3_opsd; TIME=8:00:00; SHARD_TIME=4:00:00 ;;
        olmo3_7b_instruct_*)       MODEL=Olmo-3-7B-Instruct; MODE=olmo3_opsd; TIME=8:00:00; SHARD_TIME=4:00:00 ;;
        *) echo "Unknown experiment: $NAME (try 'list')"; return 1 ;;
    esac

    # ---- dataset x condition -------------------------------------------------
    # dense = full gold demonstration in the teacher context (mpl 25000, 8x1x8)
    # sparse = final answer only (mpl 8192, 4x1x16; Qwen3-8B: 8x1x8) — bs 64 always
    case "$NAME" in
        *_ot15k_dense)
            DATASET_NAME="openthoughts_filtered_15k"; ACCUM=8 ;;
        *_ot15k_sparse)
            DATASET_NAME="openthoughts_filtered_15k__answer_only"; GOLD_KEY=Answer; MPL=8192
            GPUS=4; ACCUM=16; [ "$MODEL" = Qwen3-8B ] && { GPUS=8; ACCUM=8; TIME=6:00:00; } || TIME=5:00:00
            [ "$MODE" = olmo3_opsd ] && TIME=8:00:00 ;;
        *_ot15k_concise)
            # Conciseness-instruction control (Qwen3-8B): teacher gets no gold,
            # only a "be concise" instruction. Same data/schedule as dense.
            DATASET_NAME="openthoughts_filtered_15k__concise"; ACCUM=8; CONCISE=1 ;;
        *_countdown_dense)
            DATA_PARQUET=$CD15K; DATASET_NAME="countdown_15k"
            PROMPT_KEY=datapoint_input_text; GOLD_KEY=response_suffix; ACCUM=8
            EVAL_LIST=$CD_EVALS; WANDB_OFF=1
            [ "$MODE" = olmo3_opsd ] && { EVAL_BS=4; TIME=8:00:00; } || TIME=6:00:00 ;;
        *_countdown_sparse)
            DATA_PARQUET=$CD15K; DATASET_NAME="countdown_15k__answer_only"
            PROMPT_KEY=datapoint_input_text; GOLD_KEY=response_answer_only; MPL=8192
            GPUS=4; ACCUM=16; EVAL_LIST=$CD_EVALS; WANDB_OFF=1
            [ "$MODE" = olmo3_opsd ] && { EVAL_BS=4; TIME=8:00:00; } || TIME=5:00:00 ;;
        *_ot30k_thinking|*_ot30k_nothink)
            # Thinking vs. thinking-disabled training pair: Qwen3-4B on the
            # full 30k set at lr 1e-6 (evaluation always uses thinking mode).
            DATA_PARQUET=$OT30K; DATASET_NAME="openthoughts_filtered_30k"
            ACCUM=8; LR=1e-6; TIME=12:00:00
            [ "${NAME##*_}" = "nothink" ] && NOTHINK=1 ;;
        *) echo "Unknown experiment: $NAME (try 'list')"; return 1 ;;
    esac

    # ---- assemble ------------------------------------------------------------
    local BS=$(( GPUS * 1 * ACCUM ))
    local COMMON_EXPORTS="MODE=${MODE},BASE_MODEL=${MODELS_DIR}/${MODEL},TRAIN_PATH=${DATA_PARQUET},PROMPT_KEY=${PROMPT_KEY},GOLD_ANSWER_KEY=${GOLD_KEY},OPSD_ALPHA=0.5,OPSD_LR=${LR},OPSD_EPOCHS=1,OPSD_MICRO_BATCH_SIZE=1,OPSD_GRADIENT_ACCUM=${ACCUM},OPSD_MAX_PROMPT_LENGTH=${MPL},OPSD_MAX_COMPLETION_LENGTH=4096,OPSD_SYNC_REF_MODEL=False,DATASET_NAME=${DATASET_NAME}"
    COMMON_EXPORTS+=",JSD_CHUNK_SIZE=${JSD_CHUNK_SIZE:-0},LOSS_MAX_COMPLETION_TOKENS=${LOSS_MAX_COMPLETION_TOKENS:-0}"
    [ "$WANDB_OFF" = 1 ] && COMMON_EXPORTS+=",WANDB_MODE=disabled,WANDB_DISABLED=true"
    [ "$NOTHINK" = 1 ] && COMMON_EXPORTS+=",ENABLE_THINKING=false"
    if [ "$CONCISE" = 1 ]; then
        # Template contains commas/spaces: export by NAME so sbatch reads the
        # value from the environment instead of parsing it out of the list.
        export TEACHER_PROMPT_TEMPLATE="$CONCISE_TEMPLATE"
        COMMON_EXPORTS+=",TEACHER_PROMPT_TEMPLATE"
    fi

    local SUFFIX=""
    [ "$NOTHINK" = 1 ] && SUFFIX="__nothink"
    local CKPT="checkpoints/opsd__m_${MODEL}__d_${DATASET_NAME}__alpha0.5__bs${BS}__lr${LR}__ep1__mpl${MPL}${SUFFIX}"

    echo "== ${NAME}"
    echo "   model=${MODEL} data=${DATASET_NAME} gold=${GOLD_KEY} mpl=${MPL} bs=${BS} (${GPUS}x1x${ACCUM}) lr=${LR}"
    echo "   ckpt=${CKPT}"

    local TRAIN_JOB_ID
    TRAIN_JOB_ID=$(run_sbatch "train_${NAME}" --parsable ${QOS_ARGS} \
        --gres=gpu:${GPU_TYPE}:${GPUS} --mem-per-gpu=80G --cpus-per-gpu=12 --time=${TIME} \
        --export="${COMMON_EXPORTS}" \
        slurm/train.sh)
    echo "   training job: ${TRAIN_JOB_ID}"

    local eval_name
    for eval_name in $EVAL_LIST; do
        run_sbatch "eval_${NAME}_${eval_name}" \
            --dependency=afterok:"${TRAIN_JOB_ID}" \
            --export=ALL,EVAL_SET_NAME=${eval_name},EVAL_MAX_TOKENS=38912,NUM_SHARDS=8,CHECKPOINT_PATH="${CKPT}",TEMP=${EVAL_TEMP},N_SAMPLES=16,TOP_P=${EVAL_TOP_P},EVAL_BATCH_SIZE=${EVAL_BS}${SHARD_TIME:+,SHARD_TIME=${SHARD_TIME}} \
            slurm/generate_with_retry.sh > /dev/null
    done
    echo "   evals submitted (dep afterok): ${EVAL_LIST} @ T=${EVAL_TEMP}, top_p=${EVAL_TOP_P}, n=16, 8 shards"
    unset TEACHER_PROMPT_TEMPLATE
}

case "${1:-}" in
    list)             all_names ;;
    all)              for n in $(all_names); do submit_one "$n" || exit 1; done ;;
    openthoughts)     for n in $OT_DENSE; do submit_one "$n" || exit 1; done ;;
    sparse)           for n in $(echo "$OT_DENSE" | tr ' ' '\n' | sed 's/_dense$/_sparse/'); do submit_one "$n" || exit 1; done ;;
    countdown)        for n in $CD_DENSE; do submit_one "$n" || exit 1; done ;;
    countdown_sparse) for n in $(echo "$CD_DENSE" | tr ' ' '\n' | sed 's/_dense$/_sparse/'); do submit_one "$n" || exit 1; done ;;
    nothink)          for n in $OT30K_PAIR; do submit_one "$n" || exit 1; done ;;
    concise)          submit_one qwen3_8b_ot15k_concise ;;
    "")               echo "Usage: $0 {list|all|openthoughts|sparse|countdown|countdown_sparse|nothink|concise|<name>}"; exit 1 ;;
    *)                submit_one "$1" ;;
esac
