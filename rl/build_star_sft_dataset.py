# Build the STaR-control SFT parquet (arm D*) from the OPSD train split.
#
#   envs/train/.venv/bin/python rl/build_star_sft_dataset.py \
#       --train data/raw/countdown_opsd_train.parquet \
#       --out data/raw/countdown_star_sft.parquet
#
# prompt   = problem (bare harvest prompt -- same student prompt as arms A-C)
# response = x_plus  (the problem's own sampled CORRECT solution)
#
# Reads the TRAIN parquet (not the pool) so the 373 held-out question_ids are
# excluded by construction -- all four arms train on the same problem set.
#
# CAVEAT (flagged in rl/arms/REVIEW.md): x_plus is the THINK-STRIPPED visible
# solution (pilot/countdown_pairs.py removed <think>...</think> at harvest).
# SFT on it therefore teaches answer-without-thinking behaviour, while arms
# A-C distill over full thinking rollouts. This is the STaR control as
# specified (SFT on the pool's x_plus), but the comparison is
# outcome-supervision-matched, not channel-matched; a full-trace variant
# would need a re-harvest that keeps <think> (GPU).
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="countdown_opsd_train.parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    out = pd.DataFrame({
        "prompt": df["problem"],
        "response": df["x_plus"],
        "question_id": df["question_id"],
    })
    assert out.prompt.str.len().min() > 0 and out.response.str.len().min() > 0
    out.to_parquet(args.out, index=False)
    rc = out.response.str.len()
    print(f"wrote {len(out)} rows -> {args.out}")
    print(f"response chars: mean {rc.mean():.0f} p99 {rc.quantile(.99):.0f} max {rc.max()}")
    print("sanity: no <think> in responses:", not out.response.str.contains("<think>").any())


if __name__ == "__main__":
    main()
