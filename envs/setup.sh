#!/usr/bin/env bash
# Build the single uv-locked environment (envs/train/.venv) used for OPSD
# training, sharded vLLM generation, and eval grading.
#
# Run from the repo root on a node with internet access:
#     bash envs/setup.sh
#
# The lock pins the package versions recorded in the paper's training runs
# (torch 2.9.0, transformers 4.57.1, trl 0.24.0, vllm 0.12.0, accelerate
# 1.11.0, ...). Python 3.10 is required (uv downloads one if needed).

set -e

cd "$(dirname "$0")/train"

# Keep uv's cache off small home quotas if UV_CACHE_DIR is preconfigured
# (see slurm/cluster_env.sh.example).
uv sync --frozen

echo ""
echo "Done. Activate with:  source envs/train/.venv/bin/activate"
