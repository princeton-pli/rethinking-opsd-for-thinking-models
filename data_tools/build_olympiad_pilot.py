# Stratified olympiad-prep pilot: sample equally from several candidate pools so
# ONE harvest job measures the middle-band width of each (della queues jobs
# sequentially, so a stratified pilot beats four separate ones).
#
#   python data_tools/build_olympiad_pilot.py --per_source 128
#
# Pools (all olympiad/competition-prep, per the 2026-08-13 substrate discussion:
# these are certainly in pretraining data, but small models still fail on them,
# which is what widens the middle band):
#   numina_aops_forum  - AoPS forum: the IMO/Putnam prep venue itself
#   numina_cn_contest  - Chinese contest problems
#   numina_olympiads   - NuminaMath's olympiad pool
#   omni_math          - Omni-MATH benchmark, olympiad-level, difficulty-labelled
#
# Emits data/raw/olympiad_pilot.parquet with problem/answer/pool columns, and
# DECONTAMINATES against our eval sets (AIME24/25, HMMT25) since some of these
# pools are benchmarks that may overlap them.
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
    ap.add_argument("--per_source", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/raw/olympiad_pilot.parquet")
    args = ap.parse_args()

    frames = []

    nm = load_dataset("AI-MO/NuminaMath-1.5", split="train").to_pandas()
    nm = nm[(nm.problem_is_valid == "Yes") & (nm.question_type == "math-word-problem")]
    for src, label in [("aops_forum", "numina_aops_forum"),
                       ("cn_contest", "numina_cn_contest"),
                       ("olympiads", "numina_olympiads")]:
        sub = nm[nm.source == src]
        sub = sub[answer_ok(sub["answer"])]
        print(f"{label}: {len(sub)} clean available")
        frames.append(pd.DataFrame({"problem": sub["problem"].astype(str),
                                    "answer": sub["answer"].astype(str),
                                    "pool": label}))

    om = load_dataset("KbsdJames/Omni-MATH", split="test").to_pandas()
    om = om[answer_ok(om["answer"])]
    print(f"omni_math: {len(om)} clean available")
    frames.append(pd.DataFrame({"problem": om["problem"].astype(str),
                                "answer": om["answer"].astype(str),
                                "pool": "omni_math"}))

    # Decontaminate against the eval benchmarks before sampling.
    evalkeys = set()
    for p in glob.glob("data/eval/*.parquet"):
        ev = pd.read_parquet(p)
        col = "problem" if "problem" in ev.columns else ev.columns[0]
        evalkeys |= set(ev[col].map(key))
    print(f"eval decontamination keys: {len(evalkeys)}")

    out = []
    for f in frames:
        f = f[~f["problem"].map(key).isin(evalkeys)]
        n = min(args.per_source, len(f))
        out.append(f.sample(n=n, random_state=args.seed))
    df = pd.concat(out).reset_index(drop=True)
    df["problem"] = df["problem"].str.rstrip() + " " + BOXED

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nwrote {len(df)} rows -> {args.output}")
    print(df["pool"].value_counts().to_string())


if __name__ == "__main__":
    main()
