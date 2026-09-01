#!/usr/bin/env bash
# =============================================================================
# ARM D* — STaR control: SFT Qwen3-4B on each problem's own correct solution
# (x_plus) from the same train split as arms A-C. Answers "does the OPSD
# machinery beat just fine-tuning on the correct answers you already have?"
#
# LOGIN-NODE LAUNCHER, not an sbatch file: the SFT machinery is
# della-post-training's (data/sft_tokenize.py 'prompt_response' adapter +
# code/slurm/slurm_sft.sh), which tokenizes on the login node before
# submitting -- same shape as its recipes/01_sft_1p7b.sh. Run:
#
#   cd /scratch/gpfs/ARORA/arora/opsd && bash rl/arms/armD_star_launch.sh
#   DRY_RUN=1 bash rl/arms/armD_star_launch.sh     # print the sbatch line only
#
# SIZING (deliberately NOT 60-80 H100-h): SFT on 3,353 (prompt ~120 tok,
# response ~350 tok) pairs is ~53 steps/epoch at batch 64 -- minutes per
# epoch on 4xH100. Matching the distillation arms' compute would mean
# hundreds of epochs of pure overfitting; STaR controls are DATA-matched
# (same problems, same correct solutions), not compute-matched. Default 3
# epochs (~160 steps); the 1h limit is >10x margin.
#
# CAVEATS (also in rl/arms/REVIEW.md):
#   - x_plus is think-stripped (see rl/build_star_sft_dataset.py header).
#   - Cross-repo: uses della-post-training's venv + slurm script; the
#     checkpoint is redirected into THIS repo's checkpoints/ so the eval
#     battery (slurm/generate.sh) can consume it unchanged.
#   - EXTRA_ARGS overrides della-post-training's default report_to=wandb.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1        # opsd repo root
OPSD_ROOT=$(pwd)
DPT=/scratch/gpfs/ARORA/arora/della-post-training
DRY_RUN=${DRY_RUN:-0}

MODELS_DIR=${MODELS_DIR:-/scratch/gpfs/ARORA/skaur/models}
BASE_MODEL=${MODELS_DIR}/Qwen3-4B
RAW=${OPSD_ROOT}/data/raw/countdown_star_fullthink_922.parquet
TOKENIZED=${OPSD_ROOT}/data/sft/countdown_star_fullthink922_Qwen3-4B
CKPT=${OPSD_ROOT}/checkpoints/sft_star__m_Qwen3-4B__d_cd_fullthink922__bs64__lr5e-6__ep3

# --- 1. Build the (prompt, response) parquet from the shared train split -----
if [ ! -f "$RAW" ]; then
    "$OPSD_ROOT"/envs/train/.venv/bin/python rl/build_star_sft_fullthink.py \
        --train data/raw/countdown_opsd_train.parquet --rollouts pilot/prefix_rollouts_balanced.jsonl --qid_offset 0 --out "$RAW" || exit 1
fi

# --- 2. Tokenize with della-post-training's adapter --------------------------
if [ ! -d "$TOKENIZED" ]; then
    "$DPT"/envs/train/.venv/bin/python "$DPT"/data/sft_tokenize.py \
        --dataset_path "$RAW" \
        --dataset_format prompt_response \
        --model_name "$BASE_MODEL" \
        --output_path "$TOKENIZED" \
        --max_tokens 8192 || exit 1
    # Eyeball check: decode example 0 so template problems (e.g. an unexpected
    # empty <think> block in the assistant turn) are visible before any GPU run.
    "$DPT"/envs/train/.venv/bin/python - "$TOKENIZED" "$BASE_MODEL" <<'EOF'
import sys
from datasets import load_from_disk
from transformers import AutoTokenizer
ds = load_from_disk(sys.argv[1])
ds = ds["train"] if hasattr(ds, "keys") and "train" in getattr(ds, "keys", lambda: [])() else ds
tok = AutoTokenizer.from_pretrained(sys.argv[2])
print("=== decoded tokenized example 0 (first 1500 chars) ===")
print(tok.decode(ds[0]["input_ids"])[:1500])
print("=== end ===")
EOF
fi

# --- 3. Submit through della-post-training's SFT slurm script ----------------
SUBMIT="sbatch --partition=pli-c --gres=gpu:h100:4 --time=03:00:00 \
    --job-name=opsd_armD2_fullthink \
    --output=${OPSD_ROOT}/logs/%x-%A.out \
    --export=ALL,PROJECT_ROOT=${DPT},BASE_MODEL=${BASE_MODEL},BASE_TOKENIZER=${BASE_MODEL},DATASET_NAME=cd_fullthink922,TRAIN_PATH=${TOKENIZED},CHECKPOINT_PATH=${CKPT},SFT_LR=5e-6,SFT_EPOCHS=3,SFT_MICRO_BATCH_SIZE=1,SFT_GRADIENT_ACCUMULATION_STEPS=16,SFT_MAX_LENGTH=8192,EXTRA_ARGS=--use_load_from_disk=True\ --report_to=none \
    ${DPT}/code/slurm/slurm_sft.sh"

echo "== armD* (STaR SFT control)"
echo "   data=${RAW}  tokenized=${TOKENIZED}"
echo "   ckpt=${CKPT}"
if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] ${SUBMIT}"
    exit 0
fi
eval "$SUBMIT"
