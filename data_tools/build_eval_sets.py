"""
Build the AIME24 / AIME25 / HMMT25 evaluation parquets from public sources.

The paper evaluates on the 30-problem AIME 2024, AIME 2025, and HMMT February
2025 sets. These are public competition materials; this script fetches them
from Hugging Face mirrors and normalizes each to the two-column schema the
task handlers in tasks.py expect:

    problem (str)   fed to the model as the user turn, verbatim
    answer  (str)   gold answer (AIME: integer string; HMMT: raw LaTeX)

Problem statements from different mirrors can differ slightly in LaTeX
formatting, which perturbs prompts (not grading). If you maintain your own
curated parquets with this schema, drop them into data/eval/ and skip this
script.

Usage (from the repo root):
    python data_tools/build_eval_sets.py [--only aime24] [--force]
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset

from data_utils import EVAL_DATA_DIR

SPECS = {
    "aime24": {
        "source": "HuggingFaceH4/aime_2024",
        "splits": ["train", "test"],
        "expected_rows": 30,
    },
    "aime25": {
        "source": "math-ai/aime25",
        "splits": ["test", "train"],
        "expected_rows": 30,
    },
    "hmmt25": {
        "source": "MathArena/hmmt_feb_2025",
        "splits": ["train", "test"],
        "expected_rows": 30,
    },
}

PROBLEM_CANDIDATES = ["problem", "question", "prompt"]
ANSWER_CANDIDATES = ["answer", "final_answer", "expected_answer", "gold_answer"]


def pick_column(columns, candidates, what, source):
    for cand in candidates:
        if cand in columns:
            return cand
    raise KeyError(
        f"Could not find a {what} column in {source} (columns: {sorted(columns)}). "
        f"Tried {candidates}. Pass --source/--split overrides or adjust SPECS."
    )


def load_first_split(source, splits):
    last_err = None
    for split in splits:
        try:
            return load_dataset(source, split=split), split
        except Exception as e:  # split missing, config error, ...
            last_err = e
    raise RuntimeError(f"Could not load any split {splits} of {source}: {last_err}")


def build_one(name, spec, force=False):
    ds, split = load_first_split(spec["source"], spec["splits"])
    df = ds.to_pandas()

    problem_col = pick_column(df.columns, PROBLEM_CANDIDATES, "problem", spec["source"])
    answer_col = pick_column(df.columns, ANSWER_CANDIDATES, "answer", spec["source"])

    out = pd.DataFrame(
        {
            "problem": df[problem_col].astype(str),
            "answer": df[answer_col].astype(str),
        }
    )

    expected = spec["expected_rows"]
    if len(out) != expected:
        msg = f"{name}: expected {expected} rows, got {len(out)} from {spec['source']} [{split}]"
        if not force:
            raise ValueError(msg + " (pass --force to write anyway)")
        print(f"WARNING: {msg}")

    os.makedirs(EVAL_DATA_DIR, exist_ok=True)
    out_path = EVAL_DATA_DIR / f"{name}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"{name}: {len(out)} rows from {spec['source']} [{split}] -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None, choices=sorted(SPECS.keys()),
                        help="Build a single set instead of all three")
    parser.add_argument("--source", type=str, default=None,
                        help="Override the HF dataset id (requires --only)")
    parser.add_argument("--split", type=str, default=None,
                        help="Override the split (requires --only)")
    parser.add_argument("--force", action="store_true",
                        help="Write even if the row count is unexpected")
    args = parser.parse_args()

    names = [args.only] if args.only else sorted(SPECS.keys())
    for name in names:
        spec = dict(SPECS[name])
        if args.source:
            if not args.only:
                raise SystemExit("--source requires --only")
            spec["source"] = args.source
        if args.split:
            if not args.only:
                raise SystemExit("--split requires --only")
            spec["splits"] = [args.split]
        build_one(name, spec, force=args.force)


if __name__ == "__main__":
    main()
