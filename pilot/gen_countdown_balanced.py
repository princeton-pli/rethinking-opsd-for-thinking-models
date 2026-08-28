# Pattern-balanced hard-Countdown generator (n=5,6), following the group's own
# protocol from Park/Kaur/Arora, "How does RL Post-training induce skill
# composition? A Case Study on Countdown" (arXiv 2512.01775).
#
# Provenance: pattern machinery from princeton-pli/RL-skill-comp @ c32724b
# (plug_in_nums / is_integer vendored verbatim below; canonical patterns read
# from annotated_expressions.json produced by that repo's
# annotate_expressions.py). Their protocol: pick a canonical pattern FIRST,
# then sample numbers uniformly from [1,99] until the pattern evaluates to an
# integer target in [1,99]. Balancing per pattern removes the structural and
# selection biases they measured (each x// ~10x under-represented under naive
# generate-then-filter). Non-integer intermediates are allowed, e.g.
# 8/(3-8/3)=24 -- unlike pilot/gen_hard_countdown.py's tree construction,
# which is kept as a secondary integer-intermediate slice.
#
# The instantiated pattern is the witness: solvable by construction, stored
# for audit, NEVER in prompts. Every witness is verified through
# CountdownTask.grade before the row is kept.
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pilot.gen_hard_countdown import PROMPT_TEMPLATE


# --- vendored verbatim from RL-skill-comp/generate_puzzles.py @ c32724b ---
def plug_in_nums(pattern, nums):
    for i in range(len(nums)):
        pattern = pattern.replace(chr(ord('A') + i), str(nums[i]))
    return pattern


def is_integer(result):
    if isinstance(result, int):
        return True
    if isinstance(result, float) and result.is_integer():
        return True
    return False
# --------------------------------------------------------------------------


def load_patterns(annotated_path, size):
    with open(annotated_path) as f:
        data = json.load(f)
    patterns, seen = [], set()
    for d in data[str(size)]:
        if d["canonical"] not in seen:
            patterns.append(d["canonical"])
            seen.add(d["canonical"])
    return patterns


def sample_for_pattern(rng, pattern, size, per_pattern, seen_keys,
                       max_tries=200_000):
    """Up to per_pattern instances of one pattern (paper's inner loop).

    One value is drawn per DISTINCT letter, and the instance's number list is
    read back off the instantiated witness's leaf occurrences: 812 of the
    4,328 n=6 canonical patterns repeat a letter (e.g. "...*E-D" with D twice
    and no F), so independent per-position draws would fail the exactly-once
    multiset essentially always and silently exclude the whole
    duplicated-number structural class (CR finding, 2026-08-27 late).
    """
    letters = sorted(set(re.findall(r"[A-F]", pattern)))
    out, tries = [], 0
    while len(out) < per_pattern and tries < max_tries:
        tries += 1
        values = [rng.randint(1, 99) for _ in letters]
        expr = pattern
        for letter, value in zip(letters, values):
            expr = expr.replace(letter, str(value))
        try:
            result = eval(expr)
        except Exception:        # division by zero
            continue
        if not (is_integer(result) and 1 <= result < 100):
            continue
        nums = [int(x) for x in re.findall(r"\d+", expr)]
        if len(nums) != size:    # every k-size shape has exactly k leaves
            raise RuntimeError(f"pattern {pattern!r} has {len(nums)} leaves, "
                               f"expected {size}")
        key = (tuple(sorted(nums)), int(result))
        if key in seen_keys:     # global dedup: one row per (multiset, target)
            continue
        seen_keys.add(key)
        out.append({"nums": nums, "target": int(result), "witness": expr,
                    "pattern": pattern})
    return out


def build_rows(annotated_path, seed, plan, grade_check=None):
    """plan: list of (size, n_instances). Balanced across canonical patterns:
    cycling passes take one instance per live pattern per pass (spread <= 1
    within a pass), and a pattern that yields nothing in a pass is dropped as
    unsatisfiable-in-budget (~11% of patterns at 50k tries, mostly
    division-heavy n=6 ones) so its slot redistributes to the others."""
    rng = random.Random(seed)
    rows, seen_keys, dropped, grade_dropped = [], set(), 0, 0
    for size, n_instances in plan:
        patterns = load_patterns(annotated_path, size)
        rng.shuffle(patterns)
        active, got = patterns, 0
        while got < n_instances and active:
            next_active = []
            for pattern in active:
                if got >= n_instances:
                    next_active.append(pattern)
                    continue
                insts = sample_for_pattern(rng, pattern, size, 1, seen_keys,
                                           max_tries=50_000)
                if not insts:
                    dropped += 1
                    continue
                inst = insts[0]
                prompt = PROMPT_TEMPLATE.format(
                    nums=", ".join(str(v) for v in inst["nums"]),
                    target=inst["target"])
                row = {
                    "source_index": len(rows),
                    "datapoint_input_text": prompt,
                    "datapoint_nums": inst["nums"],
                    "datapoint_x": inst["nums"],
                    "datapoint_y": inst["target"],
                    "datapoint_target": inst["target"],
                    "input_count": size,
                    "witness_expr": inst["witness"],  # audit only
                    "canonical_pattern": inst["pattern"],
                }
                if grade_check is not None and not grade_check(row):
                    # Pattern is DROPPED (not kept live): a pattern whose
                    # instances keep failing the grader would otherwise stall
                    # the pass loop forever. Counted so exclusions are visible.
                    grade_dropped += 1
                    continue
                rows.append(row)
                got += 1
                next_active.append(pattern)
            active = next_active
        print(f"n={size}: {got} instances over {len(patterns)} patterns "
              f"(target {n_instances})")
        if got < n_instances:
            print(f"  WARNING: patterns exhausted at {got}")
    if dropped:
        print(f"dropped {dropped} pattern-passes as unsatisfiable in budget")
    if grade_dropped:
        print(f"DROPPED {grade_dropped} patterns on witness grade failure -- "
              f"investigate, this should be ~0 after the distinct-letter fix")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", default="data/raw/annotated_expressions.json")
    ap.add_argument("--plan", default="5:1500,6:1500",
                    help="comma list of size:count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/countdown_balanced_3k.parquet")
    args = ap.parse_args()

    from evaluation.tasks import CountdownTask
    task = CountdownTask({})

    def grade_check(row):
        return bool(task.grade("\\boxed{" + row["witness_expr"] + "}",
                               row["datapoint_y"], row)["is_correct"])

    plan = [(int(p.split(":")[0]), int(p.split(":")[1]))
            for p in args.plan.split(",")]
    rows = build_rows(args.annotated, args.seed, plan, grade_check=grade_check)

    import pandas as pd
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} instances -> {args.out}")
    print(f"  input_count breakdown: "
          f"{df['input_count'].value_counts().sort_index().to_dict()}")
    print(f"  distinct patterns used: {df['canonical_pattern'].nunique()}")
    print(f"  target range: {df['datapoint_y'].min()}-{df['datapoint_y'].max()}")


if __name__ == "__main__":
    sys.exit(main())
