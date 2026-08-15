# Attach NuminaMath gold solutions to an Arm-1 parquet, so the dense-gold control
# arm can be trained on exactly the same problems as Arm 1.
#
#   python data_tools/add_gold_solutions.py --parquet data/raw/contrastive_arm1_numina.parquet
#
# Without this the baseline arm has no privileged context to condition on, and the
# Arm-1 delta would have to be measured against the paper's OpenThoughts numbers --
# a different substrate, which is exactly the comparison we are trying to avoid.
# Rows whose problem text cannot be matched back are dropped from the CONTROL file
# only; Arm 1 keeps every row.
import argparse
import re

import pandas as pd
from datasets import load_dataset

BOXED = "Return your final response within \\boxed{}."


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", default=None, help="default: <parquet>.densegold.parquet")
    args = ap.parse_args()
    out_path = args.output or args.parquet.replace(".parquet", "_densegold.parquet")

    df = pd.read_parquet(args.parquet)
    print(f"{len(df)} Arm-1 rows")

    nm = load_dataset("AI-MO/NuminaMath-1.5", split="train").to_pandas()
    nm = nm[nm["solution"].notna() & (nm["solution"].astype(str).str.strip() != "")]
    # The Arm-1 problem text has the boxed instruction appended; strip it before matching.
    lookup = dict(zip(nm["problem"].map(key), nm["solution"].astype(str)))

    stripped = df["problem"].str.replace(re.escape(BOXED), "", regex=True).str.rstrip()
    sols = stripped.map(lambda p: lookup.get(key(p)))
    hit = sols.notna()
    print(f"gold solution matched for {hit.sum()}/{len(df)} ({100*hit.mean():.1f}%)")

    out = df[hit].copy()
    out["gold_solution"] = sols[hit].values
    out.to_parquet(out_path, index=False)
    print(f"wrote {len(out)} rows -> {out_path}")
    lens = out["gold_solution"].str.len()
    print(f"gold_solution chars: median {lens.median():.0f} p99 {lens.quantile(.99):.0f} max {lens.max()}")


if __name__ == "__main__":
    main()
