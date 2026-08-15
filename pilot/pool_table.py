# Per-pool coverage table for a stratified pilot harvest.
#
#   python pilot/pool_table.py <harvest.jsonl> <pilot.parquet>
#
# "usable" is the relaxed Arm-1 screen: >=1 correct rollout AND >=1 COMPLETED
# wrong rollout with an extractable answer (the gate needs a gradeable failure).
# The gap between middle-band and usable is the truncation tax -- rollouts that
# ran out of generation budget mid-<think> and so cannot be graded.
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from evaluation.utils import extract_answer_math

jsonl, parquet = sys.argv[1], sys.argv[2]
pilot = pd.read_parquet(parquet)
rows = [json.loads(l) for l in open(jsonl)]
by_q = collections.defaultdict(list)
for r in rows:
    by_q[r["question_id"]].append(r)

stats = collections.defaultdict(collections.Counter)
for qid, gens in by_q.items():
    pool = pilot.iloc[qid]["pool"] if "pool" in pilot.columns else "all"
    s = stats[pool]
    n_ok = sum(1 for r in gens if r.get("is_correct"))
    completed_wrong = sum(
        1 for r in gens
        if not r.get("is_correct")
        and ("</think>" in r["response"] or "<think>" not in r["response"])
        and extract_answer_math(r["response"])
    )
    truncated = sum(1 for r in gens if "<think>" in r["response"] and "</think>" not in r["response"])
    s["n"] += 1
    s["correct"] += n_ok
    s["gens"] += len(gens)
    s["trunc"] += truncated
    if 1 <= n_ok <= len(gens) - 1:
        s["middle"] += 1
    if n_ok >= 1 and completed_wrong >= 1:
        s["usable"] += 1

print("%-20s %6s %8s %10s %10s %10s" % ("pool", "n", "solve", "truncated", "middle", "USABLE"))
for pool, s in sorted(stats.items()):
    print("%-20s %6d %7.0f%% %9.0f%% %9.0f%% %9.0f%%" % (
        pool, s["n"], 100 * s["correct"] / s["gens"], 100 * s["trunc"] / s["gens"],
        100 * s["middle"] / s["n"], 100 * s["usable"] / s["n"]))
