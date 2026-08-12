# Build a 512-problem mid-difficulty pilot subset from DeepMath-103K.
#
#   python data_tools/build_deepmath_pilot.py [--lo 5 --hi 7] [--n 512]
#
# DeepMath ships per-problem difficulty labels and verified final answers —
# exactly what the contrastive-OPSD harvest needs (x+/x- come from the student,
# so no gold traces required; r1 solutions are kept anyway for a possible
# dense-gold baseline arm on the same data). Emits data/raw/deepmath_pilot512.parquet
# with 'problem'/'answer' columns compatible with evaluation.tasks.MathTask.
import argparse

import pandas as pd
from datasets import load_dataset

BOXED_INSTRUCTION = "Return your final response within \\boxed{}."


def pick_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"no {what} column found; have {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=5, help="difficulty band lower bound (inclusive)")
    ap.add_argument("--hi", type=float, default=7, help="difficulty band upper bound (inclusive)")
    ap.add_argument("--n", type=int, default=512, help="subset size; 0 = the whole band")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="data/raw/deepmath_pilot512.parquet")
    args = ap.parse_args()

    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    df = ds.to_pandas()
    q_col = pick_col(df, ["question", "problem", "prompt"], "question")
    a_col = pick_col(df, ["final_answer", "answer", "Answer"], "answer")
    d_col = pick_col(df, ["difficulty", "difficulty_level", "level"], "difficulty")
    print(f"{len(df)} rows; difficulty distribution:")
    print(df[d_col].value_counts().sort_index())

    band = df[(df[d_col] >= args.lo) & (df[d_col] <= args.hi)].copy()
    band = band[band[a_col].notna() & (band[a_col].astype(str).str.strip() != "")]
    print(f"band [{args.lo}, {args.hi}]: {len(band)} rows")
    if args.n > 0:
        sub = band.sample(n=min(args.n, len(band)), random_state=args.seed).reset_index(drop=True)
    else:
        sub = band.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = pd.DataFrame({
        "problem": sub[q_col].astype(str).str.rstrip() + " " + BOXED_INSTRUCTION,
        "answer": sub[a_col].astype(str),
        "difficulty": sub[d_col],
    })
    # Keep one gold solution for a potential dense-gold baseline arm.
    for sol_col in ("r1_solution_1", "solution"):
        if sol_col in sub.columns:
            out["gold_solution"] = sub[sol_col].astype(str)
            break
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out)} rows -> {args.output}")
    print(f"difficulty mix in subset:\n{out['difficulty'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
