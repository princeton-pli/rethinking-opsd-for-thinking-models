#!/bin/bash
#SBATCH --job-name=opsd_probe
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/gpfs/ARORA/arora/opsd/pilot/slurm-%j.out
# Adjudication probe: generation only (see probe_adjudication.py header).
# Single process, exits on completion -> GPU released; no keepalive needed.
set -uo pipefail
cd /scratch/gpfs/ARORA/arora/opsd || exit 1
source slurm/cluster_env.sh || exit 1
source envs/train/.venv/bin/activate || exit 1
[ -f data/raw/contrastive_arm1_pooled_v2.parquet ] || { echo "v2 parquet missing"; exit 1; }
python pilot/probe_adjudication.py
echo "PROBE GENERATION DONE"
