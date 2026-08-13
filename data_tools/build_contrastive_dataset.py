# Build the contrastive-OPSD (Arm 1) training parquet from a harvest pool.
#
#   python data_tools/build_contrastive_dataset.py \
#       --pool pilot/pilot_pool.jsonl --output data/raw/contrastive_arm1.parquet
#
# Each pool row (see pilot/pilot_coverage.py) has: problem, gold_answer,
# x_plus / x_minus (think-stripped student solutions), x_minus_answer.
# The teacher context is fully rendered HERE (template decided 2026-08-10:
# two unlabeled student responses, order randomized per problem) and shipped
# as a plain column; training passes TEACHER_PROMPT_TEMPLATE='{gold_answer}'
# with GOLD_ANSWER_KEY=teacher_context so no train.py changes are needed.
# Gate columns: 'answer' (gold, for correctness grading) and 'wrong_answer'
# (x_minus's answer, for the different-wrong-answer check).
import argparse
import json
import random
import re

import pandas as pd

TEMPLATE = (
    "{problem}\n\n"
    "Here are two examples of student responses to the question:\n"
    "{resp_a}\n \n"
    "{resp_b}\n \n"
    "Now answer with a response of your own, including the thinking process."
)


BINARY = {"yes", "no", "true", "false"}
SYNONYM = {("yes", "true"), ("true", "yes"), ("no", "false"), ("false", "no")}
_WRAPPER = re.compile(r"\\(mathbb|mathrm|text|mathcal|mathbf|left|right|displaystyle)\s*")


def _norm(s):
    """Aggressive normalization used only to DETECT grader false negatives."""
    s = _WRAPPER.sub("", str(s))
    for ch in "{}$ ":
        s = s.replace(ch, "")
    return s.replace("\\", "").replace("dfrac", "frac").replace("tfrac", "frac").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool JSONL from pilot_coverage.py")
    ap.add_argument("--output", required=True, help="output parquet path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N rows (smoke runs)")
    ap.add_argument("--keep_binary", action="store_true",
                    help="keep Yes/No-answer problems (default: drop them)")
    args = ap.parse_args()

    rows = []
    with open(args.pool) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"{len(rows)} pool rows loaded")

    # Quality filters, both established from the 2026-08-13 DeepMath pool audit:
    #  (a) binary-answer problems give a degenerate contrastive signal (x- differs
    #      from x+ by one word) and a ~50% guess rate makes the gate's label noise;
    #  (b) x- whose answer normalizes to gold is a grader FALSE NEGATIVE — a correct
    #      response mislabeled wrong (math_equal misses \mathbb{Z} vs Z, Yes vs True).
    n_bin = n_fn = 0
    kept = []
    for r in rows:
        g, w = str(r["gold_answer"]).strip(), str(r["x_minus_answer"]).strip()
        if not args.keep_binary and g.strip("$").lower() in BINARY:
            n_bin += 1
            continue
        if _norm(g) == _norm(w) or (g.lower(), w.lower()) in SYNONYM:
            n_fn += 1
            continue
        kept.append(r)
    print(f"dropped {n_bin} binary-answer, {n_fn} grader-false-negative -> {len(kept)} rows")
    rows = kept
    if args.limit:
        rows = rows[: args.limit]
        print(f"limited to {len(rows)} rows")

    rng = random.Random(args.seed)
    out = []
    n_pos_first = 0
    for r in rows:
        assert r["problem"] and r["x_plus"] and r["x_minus"] and r["gold_answer"], \
            f"incomplete pool row qid={r.get('question_id')}"
        pos_first = rng.random() < 0.5
        n_pos_first += pos_first
        a, b = (r["x_plus"], r["x_minus"]) if pos_first else (r["x_minus"], r["x_plus"])
        out.append({
            "problem": r["problem"],
            "teacher_context": TEMPLATE.format(problem=r["problem"], resp_a=a, resp_b=b),
            "answer": str(r["gold_answer"]),
            "wrong_answer": str(r["x_minus_answer"]),
            "x_plus": r["x_plus"],
            "x_minus": r["x_minus"],
            "pos_first": pos_first,
            "question_id": r["question_id"],
        })

    df = pd.DataFrame(out)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df)} rows -> {args.output} (x+ first in {n_pos_first}, "
          f"x- first in {len(df) - n_pos_first})")
    ctx_chars = df["teacher_context"].str.len()
    print(f"teacher_context chars: mean {ctx_chars.mean():.0f} p99 {ctx_chars.quantile(.99):.0f} "
          f"max {ctx_chars.max()} (~{ctx_chars.max() // 4} tokens; set OPSD_MAX_PROMPT_LENGTH above this)")


if __name__ == "__main__":
    main()
