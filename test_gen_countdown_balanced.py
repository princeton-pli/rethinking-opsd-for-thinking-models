# Tests for pilot/gen_countdown_balanced.py. Run from repo root in the train
# venv: python test_gen_countdown_balanced.py
# Needs data/raw/annotated_expressions.json (from RL-skill-comp @ c32724b).
import random
import re
import sys

from pilot.gen_countdown_balanced import (build_rows, load_patterns,
                                          sample_for_pattern)
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE
from evaluation.tasks import CountdownTask

ANNOTATED = "data/raw/annotated_expressions.json"
PLAN = [(5, 150), (6, 150)]


def main():
    task = CountdownTask({})

    def grade_check(row):
        return bool(task.grade("\\boxed{" + row["witness_expr"] + "}",
                               row["datapoint_y"], row)["is_correct"])

    rows = build_rows(ANNOTATED, seed=0, plan=PLAN, grade_check=grade_check)
    rows_again = build_rows(ANNOTATED, seed=0, plan=PLAN,
                            grade_check=grade_check)
    failures = []

    def check(name, cond):
        print(("PASS" if cond else "FAIL"), name)
        if not cond:
            failures.append(name)

    check("deterministic under fixed seed", rows == rows_again)
    check("row count meets target exactly (redistribution)",
          len(rows) == sum(n for _, n in PLAN))
    check("sizes correct", all(r["input_count"] in (5, 6)
                               and len(r["datapoint_nums"]) == r["input_count"]
                               for r in rows))
    check("numbers in [1,99]",
          all(1 <= v <= 99 for r in rows for v in r["datapoint_nums"]))
    check("target integer in [1,99]",
          all(isinstance(r["datapoint_y"], int) and 1 <= r["datapoint_y"] <= 99
              for r in rows))
    check("no duplicate (multiset, target)",
          len({(tuple(sorted(r["datapoint_nums"])), r["datapoint_y"])
               for r in rows}) == len(rows))

    # Balance: with 150 instances over 558 (n=5) / 4328 (n=6) patterns, no
    # pattern should contribute more than 1.
    from collections import Counter
    per_pat = Counter(r["canonical_pattern"] for r in rows)
    check("pattern-balanced (max 1 per pattern at this plan size)",
          max(per_pat.values()) == 1)

    check("every witness passes CountdownTask.grade",
          all(task.grade("\\boxed{" + r["witness_expr"] + "}",
                         r["datapoint_y"], r)["is_correct"] for r in rows))
    check("witness never in prompt",
          all(r["witness_expr"] not in r["datapoint_input_text"] for r in rows))
    check("prompt says exactly once",
          all("must be used exactly once" in r["datapoint_input_text"]
              for r in rows))
    check("prompt round-trips through probe regexes",
          all(NUMS_RE.search(r["datapoint_input_text"]) is not None
              and [int(x) for x in NUMS_RE.search(
                      r["datapoint_input_text"]).group(1).split(",")]
                  == r["datapoint_nums"]
              and int(TARGET_RE.search(
                      r["datapoint_input_text"]).group(1)) == r["datapoint_y"]
              for r in rows))

    # The balanced set must actually contain non-integer-intermediate
    # instances (the slice our tree generator cannot produce) -- checked via
    # witnesses whose plain eval is fine but whose every subexpression being
    # integer would be required by the tree generator. Cheap proxy: at least
    # one witness contains a division whose immediate left operand is not a
    # multiple of its right operand when both are literals... too fiddly;
    # instead just require some witness with a '/' NOT at an integer split:
    def has_nonint_intermediate(expr):
        # evaluate every parenthesized subexpression; True if any is non-integer
        spans = []
        for i, c in enumerate(expr):
            if c == "(":
                spans.append(i)
            elif c == ")":
                j = spans.pop()
                try:
                    v = eval(expr[j:i + 1])
                except Exception:
                    return False
                if isinstance(v, float) and not v.is_integer():
                    return True
        return False
    check("set includes non-integer-intermediate instances",
          any(has_nonint_intermediate(r["witness_expr"]) for r in rows))

    # Repeated-letter patterns (812 of 4328 at n=6) must be instantiable:
    # they need the SAME value at the repeated leaf, and their exclusion was
    # the CR-caught silent bias. Also: every leaf count matches input_count.
    rep = [p for p in load_patterns(ANNOTATED, 6)
           if max(map(len, [re.findall(l, p)
                            for l in set(re.findall(r"[A-F]", p))])) > 1]
    check("repeated-letter n=6 patterns exist in the source file",
          len(rep) > 0)
    rep_rng = random.Random(0)
    rep_insts = []
    for p in rep[:5]:
        rep_insts.extend(sample_for_pattern(rep_rng, p, 6, 1, set()))
    check("repeated-letter patterns instantiate and pass the grader",
          len(rep_insts) == 5 and all(
              task.grade("\\boxed{" + i["witness"] + "}", i["target"],
                         {"datapoint_nums": i["nums"]})["is_correct"]
              for i in rep_insts))
    check("repeated-letter instances duplicate a number in nums",
          all(len(set(i["nums"])) < len(i["nums"]) for i in rep_insts))
    check("leaf count == input_count on all rows",
          all(len(r["datapoint_nums"]) == r["input_count"] for r in rows))

    # Parquet round-trip through the harvest seam.
    import tempfile
    import pandas as pd
    from pilot.countdown_pairs import row_to_item
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        pd.DataFrame(rows[:50]).to_parquet(tmp.name, index=False)
        back = pd.read_parquet(tmp.name)
    rt_ok = []
    for _, row in back.iterrows():
        item = row_to_item(row)
        msgs = task.format_prompt(item)
        gold = task.get_gold(item)
        rt_ok.append(isinstance(msgs, list)
                     and msgs[0]["content"] == item["datapoint_input_text"]
                     and gold == item["datapoint_y"]
                     and task.grade("\\boxed{" + item["witness_expr"] + "}",
                                    gold, item)["is_correct"])
    check("parquet round-trip: format_prompt/get_gold/grade consistent",
          all(rt_ok))

    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        return 1
    print(f"\nall checks pass on {len(rows)} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
