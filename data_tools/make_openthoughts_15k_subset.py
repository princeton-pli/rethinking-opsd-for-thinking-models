"""
Carve the 15k training subset out of the cleaned OpenThoughts 30k parquet.

The main OpenThoughts experiments train on the first 15,000 rows of the
cleaned 30k set. Run build_openthoughts_30k.py first.

Usage (from the repo root):
    python data_tools/make_openthoughts_15k_subset.py
"""

import argparse
import os

import pandas as pd

from data_utils import RAW_DATA_DIR

DEFAULT_INPUT = RAW_DATA_DIR / "openthoughts_math_filtered_30k.parquet"
DEFAULT_OUTPUT = RAW_DATA_DIR / "openthoughts_math_filtered_15k.parquet"
SUBSET_SIZE = 15_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--size", type=int, default=SUBSET_SIZE)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    if len(df) < args.size:
        raise ValueError(
            f"{args.input_path} has only {len(df)} rows; expected >= {args.size}. "
            "Did build_openthoughts_30k.py run correctly?"
        )

    subset = df.head(args.size).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    subset.to_parquet(args.output_path, index=False)
    print(f"Wrote {len(subset)} rows -> {args.output_path}")
    print(f"Columns: {list(subset.columns)}")


if __name__ == "__main__":
    main()
