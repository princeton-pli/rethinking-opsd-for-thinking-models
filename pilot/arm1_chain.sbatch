#!/bin/bash
#SBATCH --job-name=arm1_chain
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
#
# Fires after the harvest (numina_s2) succeeds: pools the two sources, attaches
# gold solutions for the control arm, sanity-checks the result, and submits
# Arm 1 + the dense-gold baseline with their chained eval batteries.
#
# Takes 1 GPU only because pli-c rejects GPU-less jobs; the work is CPU and
# takes minutes, well inside the 90-min idle watchdog.
#
# The size assert is the point of the whole script: if the harvest yields far
# fewer usable problems than the pilots projected (~4400 numina + 2177 deepmath),
# training on it would burn a node to produce a result too underpowered to
# interpret. Better to stop and leave a sentinel for a human.

set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || { echo "chain FAILED: cd"; exit 1; }
source slurm/cluster_env.sh || { echo "chain FAILED: cluster_env"; exit 1; }
source envs/train/.venv/bin/activate || { echo "chain FAILED: venv"; exit 1; }

MIN_POOLED=${MIN_POOLED:-3000}
POOLED=data/raw/contrastive_arm1_pooled.parquet

die() { echo "CHAIN FAILED: $1"; echo "$1" > harvest/ARM1_CHAIN_FAILED; exit 1; }

python data_tools/pool_arm1_sources.py \
    --inputs data/raw/contrastive_arm1_numina.parquet:numina \
             data/raw/contrastive_arm1_deepmath_clean.parquet:deepmath \
    --output "${POOLED}" || die pooling

N=$(python -c "import pandas as pd;print(len(pd.read_parquet('${POOLED}')))")
echo "pooled training problems: ${N}"
[ "${N}" -ge "${MIN_POOLED}" ] || die "pool too small: ${N} < ${MIN_POOLED} -- inspect before training"

# Gold solutions for the dense-gold control (Arm 1 itself does not need them).
python data_tools/add_gold_solutions.py --parquet "${POOLED}" || die gold_solutions

./arm1_launch.sh arm1 || die submit_arm1
./arm1_launch.sh baseline || die submit_baseline

echo "CHAIN DONE"; echo DONE > harvest/ARM1_CHAIN_DONE
