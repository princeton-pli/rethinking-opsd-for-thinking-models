#!/usr/bin/env bash
# Merge sharded eval generations for a checkpoint, and report any missing shards.
#
#   ./merge_evals.sh checkpoints/<run>            # all three math evals
#   ./merge_evals.sh checkpoints/<run> hmmt25     # one of them
#
# Runs on the login node: merging is pure file concatenation, seconds of CPU.
#
# This exists so eval shards can be submitted with SKIP_GENERATE_MERGE=true.
# Otherwise run_generate.py makes array worker 0 a merge supervisor that polls
# for its siblings for up to ~5.5h while holding a GPU at 0% utilization -- which
# is what set off della's idle-GPU warning on 2026-08-23 (job 12828716 shard 0 sat
# 2h13m waiting on two shards that had died). Merging here also makes a missing
# shard a loud error instead of a silently short merged file.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
source envs/train/.venv/bin/activate

CKPT=${1:?usage: merge_evals.sh <checkpoint_dir> [eval_name ...]}
shift
EVALS=${*:-"aime24 aime25 hmmt25"}

rc=0
for name in $EVALS; do
    dir="${CKPT}/eval/${name}_38912"
    merged="${dir}/${name}_38912-temp_0.6-top_p_0.95.jsonl"
    n=$(ls "${merged}".shard_[0-7] 2>/dev/null | wc -l)
    if [ "$n" -ne 8 ]; then
        echo "${name}: only ${n}/8 shards present -- NOT merging (resubmit the array; finished shards skip instantly)"
        ls "${merged}".shard_[0-7] 2>/dev/null | sed 's/.*shard_/  have shard /'
        rc=1
        continue
    fi
    python run_generate.py --model_name "${CKPT}" --dataset "${name}_38912" \
        --output_dir "${dir}" --temperature 0.6 --top_p 0.95 \
        --num_shards 8 --shard_id 0 --merge_only > /dev/null 2>&1
    if [ -f "${merged}" ]; then
        echo "${name}: merged $(wc -l < "${merged}") rows"
        python run_eval.py --file_path "${merged}" 2>&1 | tail -1
    else
        echo "${name}: merge produced no file"; rc=1
    fi
done
exit $rc
