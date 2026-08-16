# Pool the NuminaMath and DeepMath Arm-1 parquets into one training set.
#
#   python data_tools/pool_arm1_sources.py \
#       --inputs data/raw/contrastive_arm1_numina.parquet:numina \
#                data/raw/contrastive_arm1_deepmath_clean.parquet:deepmath \
#       --output data/raw/contrastive_arm1_pooled.parquet
#
# Pooling was approved on the criterion "unless the difficulty levels are
# distinct" -- they are not: mean solve rate 0.51 (numina middle-band, k=3) vs
# 0.48 (deepmath usable, k=8). The screen (>=1 correct AND >=1 gradeable wrong)
# is itself a difficulty filter, which compresses whatever difference the sources
# had. A `source` column is carried so the eval can be split post-hoc: the two
# do differ in TOPIC (deepmath 7.5-9 skews graduate-abstract, numina
# aops_forum/cn_contest is competition-style and closer to the AIME/HMMT eval),
# so a topic-dependent effect stays detectable.
#
# question_id is re-issued globally: the per-source ids overlap, and the trainer
# only needs them to be unique.
import argparse

import pandas as pd

SHARED = ["problem", "teacher_context", "answer", "wrong_answer",
          "x_plus", "x_minus", "pos_first"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="each as path:source_label")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = []
    for spec in args.inputs:
        path, _, label = spec.rpartition(":")
        df = pd.read_parquet(path)
        missing = [c for c in SHARED if c not in df.columns]
        assert not missing, f"{path} missing columns {missing}"
        sub = df[SHARED].copy()
        sub["source"] = label
        sub["orig_question_id"] = df["question_id"].values
        print(f"{label:12s} {len(sub):6d} rows  <- {path}")
        frames.append(sub)

    pooled = pd.concat(frames, ignore_index=True)
    # Shuffle so the two sources interleave: unshuffled, one source would occupy
    # whole contiguous batches and any per-step metric would swing by source.
    pooled = pooled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    pooled["question_id"] = range(len(pooled))
    pooled.to_parquet(args.output, index=False)

    print(f"\nwrote {len(pooled)} rows -> {args.output}")
    print(pooled["source"].value_counts().to_string())
    print(f"steps at effective batch 64: {len(pooled) // 64}")
    ctx = pooled["teacher_context"].str.len()
    print(f"teacher_context chars: mean {ctx.mean():.0f} p99 {ctx.quantile(.99):.0f} "
          f"max {ctx.max()} (~{int(ctx.max() // 4)} tokens)")


if __name__ == "__main__":
    main()
