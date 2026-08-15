# Build the full Arm-1 harvest band from NuminaMath-1.5 competition pools.
#
#   python data_tools/build_numina_band.py --pools aops_forum,cn_contest
#
# Pool choice comes from the 2026-08-14 stratified pilot at generation cap 32768,
# where usable rates were aops_forum 18%, cn_contest 17%, olympiads 12%
# ("usable" = >=1 correct AND >=1 completed gradeable wrong rollout). Same answer
# sanity filter as the pilot, and the same decontamination against AIME/HMMT.
import argparse
import glob
import os
import re

import pandas as pd
from datasets import load_dataset

BOXED = "Return your final response within \\boxed{}."
BINARY = {"yes", "no", "true", "false"}


def answer_ok(a: pd.Series) -> pd.Series:
    a = a.astype(str).str.strip()
    bad = a.str.lower().isin({"proof", "", "nan", "none", "notfound"} | BINARY)
    return ~(bad | (a.str.len() > 60) | a.str.fullmatch(r"[a-zA-Z ]{6,}").fillna(False))


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="aops_forum,cn_contest")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole band")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/raw/numina_band.parquet")
    args = ap.parse_args()

    nm = load_dataset("AI-MO/NuminaMath-1.5", split="train").to_pandas()
    nm = nm[(nm.problem_is_valid == "Yes") & (nm.question_type == "math-word-problem")]
    pools = [p.strip() for p in args.pools.split(",")]
    sub = nm[nm.source.isin(pools)]
    sub = sub[answer_ok(sub["answer"])]
    print(f"pools {pools}: {len(sub)} clean problems")

    evalkeys = set()
    for p in glob.glob("data/eval/*.parquet"):
        ev = pd.read_parquet(p)
        col = "problem" if "problem" in ev.columns else ev.columns[0]
        evalkeys |= set(ev[col].map(key))
    before = len(sub)
    sub = sub[~sub["problem"].map(key).isin(evalkeys)]
    print(f"decontaminated against {len(evalkeys)} eval problems: dropped {before - len(sub)}")

    sub = sub.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        sub = sub.head(args.limit)

    out = pd.DataFrame({
        "problem": sub["problem"].astype(str).str.rstrip() + " " + BOXED,
        "answer": sub["answer"].astype(str),
        "pool": sub["source"].astype(str),
    })
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out)} rows -> {args.output}")
    print(out["pool"].value_counts().to_string())


if __name__ == "__main__":
    main()
