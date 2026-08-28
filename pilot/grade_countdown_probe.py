# Grade the Countdown adjudication probe, and audit the grader itself.
#
# Two questions:
#  1) Can Qwen3-4B adjudicate its own correct-vs-wrong Countdown solutions?
#  2) Is our LABEL reliable here, or do we repeat the math false-negative problem?
#
# For (2) Countdown should be structurally safe: correctness is "does this
# expression evaluate to the target using exactly the allowed numbers", which is
# eval() + a multiset compare -- no notation ambiguity, so no equivalent-but-
# differently-written class. This script checks that claim rather than assuming
# it, by re-deriving each verdict independently from the problem text and
# reporting any disagreement with CountdownTask.
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}")
NUMS_RE = re.compile(r"Numbers:\s*\[([0-9,\s]+)\]")
TARGET_RE = re.compile(r"Target:\s*(-?\d+)")


def independent_grade(expr_text, nums, target):
    """Re-derive correctness from scratch: evaluate, compare, check multiset."""
    # Keep only the expression side of "expr = result" (same fix as
    # CountdownTask.grade in b0de1b4): the guard below forbids "=", so
    # "92+5-66=31" would score False regardless of the arithmetic.
    expr_text = expr_text.split("=")[0]
    e = (expr_text.replace("$", "").replace(" ", "")
         .replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
         .replace("\\left", "").replace("\\right", "")
         .replace("{", "(").replace("}", ")"))
    e = re.sub(r"\\[dt]?frac\(([^()]+)\)\(([^()]+)\)", r"((\1)/(\2))", e)
    if not re.fullmatch(r"[\d\+\-\*/\(\)\.]+", e):
        return False
    try:
        val = eval(e)
    except Exception:
        return False
    if abs(val - float(target)) > 1e-5:
        return False
    used = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", e)]
    return collections.Counter(used) == collections.Counter(float(n) for n in nums)


def main():
    ap = argparse.ArgumentParser()
    # v2 defaults: the v1 files were harvested through the pre-b0de1b4 grader
    # (~92% of x- actually correct) and every number derived from them is void.
    ap.add_argument("--probe", default="pilot/probe_countdown_v2.jsonl")
    ap.add_argument("--pairs", default="pilot/countdown_pairs_v2.parquet")
    args = ap.parse_args()

    pairs = pd.read_parquet(args.pairs).set_index("question_id")
    rows = [json.loads(l) for l in open(args.probe)]
    print(f"{len(rows)} probe generations over {len({r['question_id'] for r in rows})} problems")

    cells = collections.defaultdict(lambda: [0, 0])
    copies = collections.defaultdict(collections.Counter)
    audit = collections.Counter()

    for r in rows:
        q = pairs.loc[r["question_id"]]
        nums = [int(x) for x in NUMS_RE.search(q["problem"]).group(1).split(",")]
        target = int(TARGET_RE.search(q["problem"]).group(1))

        boxes = BOX_RE.findall(r["response"])
        pred = boxes[-1].strip() if boxes else ""
        ok = independent_grade(pred, nums, target) if pred else False

        c = cells[r["cell"]]
        c[0] += ok
        c[1] += 1

        # Which candidate did it reproduce?
        norm = lambda s: re.sub(r"\s", "", str(s))
        if pred:
            if norm(pred) == norm(q["answer"]) or ok:
                copies[r["cell"]]["x+"] += 1
            elif norm(pred) == norm(q["wrong_answer"]):
                copies[r["cell"]]["x-"] += 1
            else:
                copies[r["cell"]]["other"] += 1

        # Audit: does the STORED x-/x+ label survive an independent re-grade?
        audit["x+ regraded correct"] += independent_grade(
            (BOX_RE.findall(q["x_plus"]) or [""])[-1], nums, target)
        audit["x- regraded correct (FALSE NEGATIVE)"] += independent_grade(
            (BOX_RE.findall(q["x_minus"]) or [""])[-1], nums, target)
        audit["n"] += 1

    print("\n%-22s %8s %7s   %8s %8s" % ("cell", "acc", "n", "picks x+", "picks x-"))
    for cell in ["none/nothink", "unlabeled/nothink", "neglabel/nothink",
                 "none/think", "unlabeled/think", "neglabel/think"]:
        ok, n = cells[cell]
        cp = copies[cell]
        tot = sum(cp.values()) or 1
        print("%-22s %8.3f %7d   %7.1f%% %7.1f%%" %
              (cell, ok / max(n, 1), n, 100 * cp["x+"] / tot, 100 * cp["x-"] / tot))

    print()
    for cell in ["unlabeled/nothink", "unlabeled/think"]:
        cp = copies[cell]
        d = cp["x+"] + cp["x-"]
        print("%-20s when it copies a candidate, picks the RIGHT one: %.1f%% (n=%d)"
              % (cell, 100 * cp["x+"] / max(d, 1), d))

    n = max(audit["n"], 1)
    print(f"\n=== label audit (independent re-grade of the stored exemplars) ===")
    print(f"  x+ confirmed correct : {100*audit['x+ regraded correct']/n:.1f}%")
    print(f"  x- actually CORRECT  : {100*audit['x- regraded correct (FALSE NEGATIVE)']/n:.1f}%"
          f"   <- math pool was 11.1%")


if __name__ == "__main__":
    main()
