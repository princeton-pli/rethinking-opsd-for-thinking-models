# Build the x+/x- training pool from harvest JSONLs (stage 1 + optional stage 2).
#
#   python data_tools/build_pool.py --out pool.jsonl stage1.jsonl [stage2.jsonl ...]
#
# Relaxed screen (decided 2026-08-12 after 4 pilots showed modal errors on every
# substrate): usable(q) = >=1 correct completed response AND >=1 wrong completed
# response. No distinct-wrong requirement. x+ = shortest correct visible solution;
# x- = shortest visible solution ending in the MODAL wrong answer (math_equal
# bucketing). Stage-2 rows are remapped to stage-1 question ids via the
# 'orig_question_id' column that survivors parquets carry.
import argparse
import json
import os
import re
import signal
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.utils import extract_answer_math, strip_string
from evaluation.grader import math_equal

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


def _alarm(signum, frame):
    raise TimeoutError()


def safe_math_equal(a, b, seconds=20):
    """math_equal with a hard wall-clock ceiling.

    math_equal's own timeout only guards its top-level call; its recursive
    branches re-enter with timeout=False and run sympy inline and unbounded. A
    wedge here would hang the pool build inside a GPU allocation, and with the
    keepalive running nothing would kill it until the walltime.
    """
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return math_equal(a, b, timeout=True)
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonls", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_q = collections.defaultdict(list)
    for path in args.jsonls:
        n = 0
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = r.get("orig_question_id", r["question_id"])
                by_q[qid].append(r)
                n += 1
        print(f"{path}: {n} generations")

    pool, stats = [], collections.Counter()
    for qid, gens in sorted(by_q.items()):
        correct_pool = []
        wrong_by_answer = collections.defaultdict(list)
        n_correct = 0
        for r in gens:
            resp = r["response"]
            visible = THINK_RE.sub("", resp).strip()
            completed = len(visible) > 0 and ("<think>" not in resp or "</think>" in resp)
            pred = extract_answer_math(resp)
            if r.get("is_correct"):
                n_correct += 1
                if completed:
                    correct_pool.append((len(visible), visible))
            elif pred and completed:
                norm = strip_string(str(pred))
                for rep in wrong_by_answer:
                    if safe_math_equal(norm, rep):
                        norm = rep
                        break
                wrong_by_answer[norm].append((len(visible), visible))

        if not correct_pool or not wrong_by_answer:
            stats["skipped"] += 1
            continue
        stats["usable"] += 1
        x_plus = min(correct_pool)[1]
        modal_answer, modal_traces = max(
            wrong_by_answer.items(), key=lambda kv: (len(kv[1]), -min(t[0] for t in kv[1]))
        )
        raw = gens[0]
        pool.append({
            "question_id": qid,
            "problem": raw.get("problem", ""),
            "gold_answer": raw.get("gold_answer", ""),
            "n_correct": n_correct,
            "n_gens": len(gens),
            "wrong_answer_counts": {k: len(v) for k, v in wrong_by_answer.items()},
            "x_plus": x_plus,
            "x_minus": min(modal_traces)[1],
            "x_minus_answer": modal_answer,
        })

    with open(args.out, "w") as f:
        for e in pool:
            f.write(json.dumps(e) + "\n")
    nq = len(by_q)
    print(f"questions: {nq}; usable: {stats['usable']} ({stats['usable'] / max(nq, 1):.2f}); "
          f"skipped: {stats['skipped']}")
    print(f"wrote {len(pool)} -> {args.out}")


if __name__ == "__main__":
    main()
