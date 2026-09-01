#!/usr/bin/env bash
#SBATCH --job-name=opsd_armA_tailwin_rlT
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=80G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%A.out
# =============================================================================
# ARM A — tail-window token weighting + RL'd (v4tpl grader) frozen teacher.
# Countdown contrastive-OPSD, token-weighted line (rl/proposals/, rev 3).
#
#   cd /scratch/gpfs/ARORA/arora/opsd
#   sbatch rl/arms/armA.sbatch                                   # real run
#   sbatch --time=01:00:00 --job-name=opsd_armA_smoke \
#          --export=ALL,SMOKE=1 rl/arms/armA.sbatch              # MANDATORY 1h
#                                        # memory smoke FIRST: 3 optimizer
#                                        # steps, full-trace loss, logs GPU
#                                        # memory every 60s. Full-window JSD
#                                        # peak is an ESTIMATE (~53 GB vs Arm
#                                        # 1's measured 38.9 GB at loss=4096);
#                                        # nothing launches before this passes.
#
# SIZING (from Arm-1's measured throughput; see rl/arms/REVIEW.md):
#   Arm-1 gated OPSD, 8xH100, gen 16384, gate regen 3: job 12777077 = 91
#   steps in 13:54:50 -> 1.22 H100-h/step; job 12836814 = 57 steps in
#   8:56:51 -> 1.26 H100-h/step. This arm: 3353 rows / batch 64 = 53 steps
#   x ~1.25 = ~66 H100-h, + full-window loss overhead (unmeasured, JSD
#   forwards at up to 16k instead of 4k loss tokens; generation still
#   dominates) -> estimate 66-75 H100-h == 8.3-9.4h wall on 8 GPUs.
#   Time limit 14h = 49% margin over the 9.4h high estimate (the standing
#   >=30% rule after the 13202596-98 harvest TIMEOUTs).
#
# Teacher: PEFT ADAPTER, NOT a merged dir. Measured 2026-08-31: global
# max|dW| = 3.0e-4 over all 252 LoRA modules, and a bf16 merge destroys the
# delta (norm survival 0.52-0.80, cos(effective, intended) 0.39-0.62) -- the
# HANDOFF_2026-08-30 bf16-merge concern APPLIES to this adapter after all.
# The unmerged adapter computes its delta in activation space, which is how
# every v4tpl eval was run (vLLM enable_lora / fp32 forward), so the probe
# gains (0.732 -> 0.893) certify exactly this load path. train.py loads it
# via transformers' peft integration (adapter_config.json in the dir).
# =============================================================================
cd /scratch/gpfs/ARORA/arora/opsd || exit 1   # absolute: do not trust SLURM_SUBMIT_DIR

export MODELS_DIR=${MODELS_DIR:-/scratch/gpfs/ARORA/skaur/models}
export BASE_MODEL=${MODELS_DIR}/Qwen3-4B
export BASE_TOKENIZER=${BASE_MODEL}
export TRAIN_PATH=data/raw/countdown_opsd_train.parquet   # rl/build_opsd_dataset.py
                                                          # (held-out 373 qids excluded)

# --- Arm identity ------------------------------------------------------------
export TOKEN_WEIGHT_MODE=tail-window
export TOKEN_WEIGHT_TAIL_TOKENS=4096
export TOKEN_WEIGHT_EPSILON=0.001
export TEACHER_MODEL=rl/checkpoints/grader_grpo_lora_v4tpl/eval_adapter
export DATASET_NAME=cd_pairs_v4tpl
# slurm/train.sh appends __tw_tail-window__tchr_grader_grpo_lora_v4tpl_eval_adapter
# to the checkpoint name, so arms can never silently collide.

# --- Shared with Arm 1 / the paper (arm1_launch.sh lineage) ------------------
export MODE=qwen3_opsd
export OPSD_ALPHA=0.5 OPSD_LR=5e-6 OPSD_EPOCHS=1
export OPSD_MICRO_BATCH_SIZE=1 OPSD_GRADIENT_ACCUM=8       # 8x1x8 = batch 64
export OPSD_MAX_PROMPT_LENGTH=4096      # measured teacher_context max: 3,344 tokens
export OPSD_MAX_COMPLETION_LENGTH=16384 # gate needs a gradeable \boxed{} answer
export LOSS_MAX_COMPLETION_TOKENS=0     # REQUIRED by token weighting (hard assert):
                                        # the weighted tail sits at the trace END
export JSD_CHUNK_SIZE=1024
export OPSD_SYNC_REF_MODEL=False        # frozen teacher (also asserted by the swap)
export PROMPT_KEY=problem
export GOLD_ANSWER_KEY=teacher_context  # pre-rendered v4 pair context
export TEACHER_PROMPT_TEMPLATE='{gold_answer}'
# (expandable_segments removed: incompatible with vLLM sleep-mode memory pool)
export SAVE_STRATEGY=steps SAVE_STEPS=15  # late-OOM costs a resume, not the run
export GATE_MODE=wrong_only
export GATE_GRADER=countdown         # deterministic task grader for the gate
export GATE_REQUIRE_DIFF_ANSWER=False   # as in Arm 1
export GATE_GOLD_ANSWER_KEY=answer      # Countdown TARGET integer (value-equality
                                        # gate; constraint caveat in REVIEW.md)
export GATE_WRONG_ANSWER_KEY=wrong_answer
export GATE_MAX_REGEN_ROUNDS=3
export WANDB_MODE=disabled WANDB_DISABLED=true             # no outbound on pli-c

# --- Smoke variant -----------------------------------------------------------
if [ "${SMOKE:-0}" = "1" ]; then
    export MAX_STEPS=3
    export DATASET_NAME=${DATASET_NAME}_smoke   # never collide with the real
                                                # checkpoint: train.py SAVES at
                                                # exit even after 3 steps, and
                                                # slurm/train.sh skips training
                                                # when config.json exists
    mkdir -p logs
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 60 \
        > "logs/smoke_mem-${SLURM_JOB_ID}.csv" 2>&1 &
    SMOKE_MEM_PID=$!
    echo "[smoke] 3 steps, full-trace loss; GPU memory -> logs/smoke_mem-${SLURM_JOB_ID}.csv"
fi

bash slurm/train.sh
EXIT_CODE=$?
[ -n "${SMOKE_MEM_PID:-}" ] && kill "${SMOKE_MEM_PID}" 2>/dev/null
if [ "${SMOKE:-0}" = "1" ]; then
    echo "[smoke] peak memory.used per GPU (MiB):"
    awk -F', ' 'NR>1 && $2+0 > max[$1]+0 { max[$1]=$2+0 } END { for (g in max) print "  GPU " g ": " max[g] }' \
        "logs/smoke_mem-${SLURM_JOB_ID}.csv"
fi
exit $EXIT_CODE
