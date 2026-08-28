# Tests for pilot/gen_hard_countdown.py. Run from repo root in the train venv
# (needs evaluation.tasks): python test_gen_hard_countdown.py
#
# The witness check uses CountdownTask.grade itself as the oracle -- the same
# grader the harvest and RL reward will use -- rather than a reimplementation
# (an auditor sharing the auditee's assumptions is not an audit; see
# HANDOFF_2026-08-27 §7).
import re
import sys

from pilot.gen_hard_countdown import (build_rows, gen_instance,
                                      signed_sum_solvable, PROMPT_TEMPLATE)
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE
from evaluation.tasks import CountdownTask

MIX = [(5, 0.5), (6, 0.5)]
N = 200


def main():
    task = CountdownTask({})
    rows = build_rows(seed=0, n_instances=N, mix=MIX)
    rows_again = build_rows(seed=0, n_instances=N, mix=MIX)
    failures = []

    def check(name, cond):
        print(("PASS" if cond else "FAIL"), name)
        if not cond:
            failures.append(name)

    check("deterministic under fixed seed", rows == rows_again)
    check("row count", len(rows) == N)
    check("n in {5,6}", all(r["input_count"] in (5, 6) for r in rows))
    check("both sizes present",
          {r["input_count"] for r in rows} == {5, 6})
    check("numbers in [1,99]",
          all(1 <= v <= 99 for r in rows for v in r["datapoint_nums"]))
    check("target in [10,100]",
          all(10 <= r["datapoint_y"] <= 100 for r in rows))
    check("no duplicate (multiset, target)",
          len({(tuple(sorted(r["datapoint_nums"])), r["datapoint_y"])
               for r in rows}) == N)

    graded = [task.grade("\\boxed{" + r["witness_expr"] + "}",
                         r["datapoint_y"], r)["is_correct"] for r in rows]
    check("every witness passes CountdownTask.grade (arith + exactly-once)",
          all(graded))

    check("witness never appears in the prompt",
          all(r["witness_expr"] not in r["datapoint_input_text"] for r in rows))
    check("prompt says exactly once",
          all("must be used exactly once" in r["datapoint_input_text"]
              for r in rows))

    parsed = []
    for r in rows:
        m_nums = NUMS_RE.search(r["datapoint_input_text"])
        m_tgt = TARGET_RE.search(r["datapoint_input_text"])
        parsed.append(
            m_nums is not None and m_tgt is not None
            and [int(x) for x in m_nums.group(1).split(",")] == r["datapoint_nums"]
            and int(m_tgt.group(1)) == r["datapoint_y"])
    check("prompt round-trips through the probe regexes", all(parsed))

    # eval() of the witness with the grader's own normalization must equal the
    # target exactly (positive-integer intermediates make this exact division).
    vals_ok = []
    for r in rows:
        e = r["witness_expr"]
        vals_ok.append(re.fullmatch(r"[\d+\-*/()]+", e) is not None
                       and eval(e) == r["datapoint_y"])
    check("witness is plain arithmetic and evals to target", all(vals_ok))

    check("no instance is signed-sum solvable (default filter)",
          not any(signed_sum_solvable(r["datapoint_nums"], r["datapoint_y"])
                  for r in rows))

    # Parquet round-trip through the exact seam the GPU harvest uses:
    # list column -> ndarray -> row_to_item -> format_prompt/get_gold/grade
    # (the landmine documented in countdown_pairs.py's row_to_item docstring).
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
        rt_ok.append(
            isinstance(msgs, list)
            and msgs[0]["content"] == item["datapoint_input_text"]
            and gold == item["datapoint_y"]
            and task.grade("\\boxed{" + item["witness_expr"] + "}",
                           gold, item)["is_correct"])
    check("parquet round-trip: format_prompt/get_gold/grade all consistent",
          all(rt_ok))

    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        return 1
    print(f"\nall checks pass on {N} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
