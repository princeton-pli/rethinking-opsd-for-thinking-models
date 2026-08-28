# Hard-Countdown generator: 5-6 numbers, solvable by construction.
#
# Motivation (SPEC_grader_rl.md §3): with the honest grader, 3-4 number
# Countdown is generation-too-easy for Qwen3-4B (75.2% of problems solved 8/8,
# usable-pair rate 7.9%), so it cannot supply (x+, x-) training pairs. More
# numbers grow the search space combinatorially while verification stays a
# one-line eval. We GENERATE hard instances rather than filter existing ones:
# filtering conditions on model failure and biases the sub-distribution.
#
# Construction: sample n numbers, then repeatedly combine two pooled values
# with a random op under classic-Countdown constraints (every intermediate a
# positive integer). The finished expression's value is the target, so every
# instance is solvable by construction. The witness expression is stored for
# audit but must NEVER enter a prompt (statement-only discipline).
#
# The prompt says "must be used exactly once", unlike the source data's "at
# most once" -- CountdownTask.grade has always demanded the full multiset, and
# the mismatch made 7.7% of harvested x- mere subset solutions.
import argparse
import random
import sys

PROMPT_TEMPLATE = (
    "You are given a list of numbers and a target number. Your goal is to use "
    "the given numbers with basic arithmetic operations (+, -, *, /) to reach "
    "the target number. Each number must be used exactly once.\n"
    "Numbers: [{nums}]\n"
    "Target: {target}\n"
    "Please provide an equation using these numbers to reach the target. For "
    "example, if the numbers are [2, 3, 5] and the target is 10, then the "
    "answer could be \\boxed{{2+3+5}}.\n"
    "Please reason step by step and put your final answer within \\boxed{{}}."
)

INTERMEDIATE_CAP = 9999  # keeps rejection sampling fast; classic shows allow ~this


def _combine(rng, pool):
    """Combine two random pool entries with a random legal op, in place.

    Pool entries are (value, expr_str). Constraints: results are positive
    integers <= INTERMEDIATE_CAP; x1/:1 multiplication-division and a-b=0 are
    rejected as degenerate (they reduce the effective number count).
    Returns False if the chosen pair admits no legal op.
    """
    i, j = rng.sample(range(len(pool)), 2)
    (va, ea), (vb, eb) = pool[i], pool[j]
    ops = []
    if va + vb <= INTERMEDIATE_CAP:
        ops.append(("+", va + vb, f"({ea}+{eb})"))
    if va != vb:
        hi, lo = ((va, ea), (vb, eb)) if va > vb else ((vb, eb), (va, ea))
        ops.append(("-", hi[0] - lo[0], f"({hi[1]}-{lo[1]})"))
    if va != 1 and vb != 1 and va * vb <= INTERMEDIATE_CAP:
        ops.append(("*", va * vb, f"({ea}*{eb})"))
    if vb > 1 and va % vb == 0:
        ops.append(("/", va // vb, f"({ea}/{eb})"))
    if va > 1 and vb % va == 0:
        ops.append(("/", vb // va, f"({eb}/{ea})"))
    if not ops:
        return False
    _, val, expr = rng.choice(ops)
    for k in sorted((i, j), reverse=True):
        pool.pop(k)
    pool.append((val, expr))
    return True


def gen_instance(rng, n, num_lo=1, num_hi=99, target_lo=10, target_hi=100,
                 max_tries=200):
    """One solvable instance: dict(nums, target, witness) or None."""
    for _ in range(max_tries):
        nums = [rng.randint(num_lo, num_hi) for _ in range(n)]
        pool = [(v, str(v)) for v in nums]
        ok = True
        while len(pool) > 1:
            if not _combine(rng, pool):
                ok = False
                break
        if not ok:
            continue
        target, witness = pool[0]
        if target_lo <= target <= target_hi:
            return {"nums": nums, "target": target, "witness": witness}
    return None


def build_rows(seed, n_instances, mix):
    """Deduped rows in the countdown_15k normalized schema (+ witness)."""
    rng = random.Random(seed)
    sizes = [n for n, _ in mix]
    weights = [w for _, w in mix]
    rows, seen, attempts = [], set(), 0
    while len(rows) < n_instances:
        attempts += 1
        if attempts > n_instances * 50:
            raise RuntimeError(
                f"only {len(rows)}/{n_instances} after {attempts} attempts; "
                "constraints too tight")
        n = rng.choices(sizes, weights=weights)[0]
        inst = gen_instance(rng, n)
        if inst is None:
            continue
        key = (tuple(sorted(inst["nums"])), inst["target"])
        if key in seen:
            continue
        seen.add(key)
        prompt = PROMPT_TEMPLATE.format(
            nums=", ".join(str(v) for v in inst["nums"]), target=inst["target"])
        rows.append({
            "source_index": len(rows),
            "datapoint_input_text": prompt,
            "datapoint_nums": inst["nums"],
            "datapoint_x": inst["nums"],
            "datapoint_y": inst["target"],
            "datapoint_target": inst["target"],
            "input_count": n,
            "witness_expr": inst["witness"],   # audit only -- never in prompts
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_instances", type=int, default=3000)
    ap.add_argument("--mix", default="5:0.5,6:0.5",
                    help="comma list of n:weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/countdown_hard_3k.parquet")
    args = ap.parse_args()

    mix = [(int(p.split(":")[0]), float(p.split(":")[1]))
           for p in args.mix.split(",")]
    rows = build_rows(args.seed, args.n_instances, mix)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    counts = df["input_count"].value_counts().sort_index()
    print(f"wrote {len(df)} instances -> {args.out}")
    print(f"  input_count breakdown: {counts.to_dict()}")
    print(f"  target range: {df['datapoint_target'].min()}"
          f"-{df['datapoint_target'].max()}")


if __name__ == "__main__":
    sys.exit(main())
