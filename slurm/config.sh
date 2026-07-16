#!/usr/bin/env bash
# Centralized configuration, sourced by every slurm/ template. Every variable
# can be overridden by exporting it before submission (the scripts/submit_*.sh
# wrappers do exactly that). Cluster-specific setup (module loads, NCCL
# workarounds, cache locations) belongs in slurm/cluster_env.sh — see
# slurm/cluster_env.sh.example.

# === 1. Core paths ===
export MODELS_DIR=${MODELS_DIR:-"models"}
export BASE_MODEL=${BASE_MODEL:-"${MODELS_DIR}/Qwen3-1.7B"}
export BASE_TOKENIZER=${BASE_TOKENIZER:-"$BASE_MODEL"}
export MODE=${MODE:-"qwen3_opsd"}   # selects configs/fsdp_config_${MODE}.json
model_basename=$(basename "$BASE_MODEL")
export model_basename

export DATASET_NAME=${DATASET_NAME:-"openthoughts_filtered_15k"}
export DATA_ROOT=${DATA_ROOT:-"data"}
export TRAIN_PATH=${TRAIN_PATH:-"${DATA_ROOT}/raw/openthoughts_math_filtered_15k.parquet"}

# Wandb
export WANDB_PROJECT=${WANDB_PROJECT:-"rethinking-opsd"}

# === 2. Shared training hyperparameters ===
export SEED=${SEED:-42}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
export FSDP_CONFIG="configs/fsdp_config_${MODE}.json"

# === 3. Evaluation defaults (paper settings) ===
export EVAL_SET_NAME=${EVAL_SET_NAME:-"aime24_38912"}
export NUM_SHARDS=${NUM_SHARDS:-8}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
export N_SAMPLES=${N_SAMPLES:-16}
export TEMP=${TEMP:-0.6}
export TOP_P=${TOP_P:-0.95}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-38912}
export TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
export USE_R1_STYLE_PROMPT=${USE_R1_STYLE_PROMPT:-false}
export DO_REGRADE=${DO_REGRADE:-false}
