# Offline correctness grading for harvest JSONLs produced with --no_grade.
#
#   python data_tools/grade_jsonl.py --in raw.jsonl --out graded.jsonl [--workers 16]
#
# WHY THIS EXISTS. math_equal implements its 1s timeout by forking the calling
# process. When that caller is a vLLM worker holding a CUDA context, the fork can
# deadlock: on 2026-08-15 shard 0 of job 12413497 froze mid-batch and its idle GPU
# tripped della's 90-min watchdog, killing all 8 shards after 5h. Grading is pure
# CPU work, so it belongs in a plain CPU process where a fork is cheap and a hang
# costs minutes instead of GPU-hours.
#
# Defence in depth, since a sympy hang is still possible even on CPU:
#   * work is spread over a multiprocessing Pool, so one bad row stalls one worker
#   * each row is graded through the pool's own timeout and falls back to False
#   * output is written incrementally and the run is resumable (already-graded
#     question/generation pairs are skipped), so a restart never redoes work
import argparse
import json
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.tasks import get_task

_HANDLER = None
_TASK_NAME = None


def _init(task_name):
    global _HANDLER, _TASK_NAME
    _TASK_NAME = task_name
    _HANDLER = get_task(task_name)


def _grade_one(payload):
    """Grade a single row. Returns (key, is_correct). Never raises."""
    key, response, gold = payload

    def _alarm(signum, frame):
        raise TimeoutError()

    # Hard per-row ceiling on top of math_equal's own timeout: sympy can wedge
    # inside a single call, and a wedged pool worker would stall the whole run.
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        return key, bool(_HANDLER.grade(response, gold, {})["is_correct"])
    except Exception:
        return key, False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--task", default="math", help="task name for get_task (default: math)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["question_id"], r["generation_id"]))
                except Exception:
                    continue
        print(f"resuming: {len(done)} rows already graded")

    rows = []
    with open(args.inp) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r["question_id"], r["generation_id"]) in done:
                continue
            rows.append(r)
    print(f"{len(rows)} rows to grade from {args.inp}")
    if not rows:
        print("nothing to do")
        return

    import multiprocessing as mp

    payloads = [((r["question_id"], r["generation_id"]), r["response"], r.get("gold_answer", ""))
                for r in rows]
    verdicts = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.task,)) as pool:
        for i, (key, ok) in enumerate(pool.imap_unordered(_grade_one, payloads, chunksize=8), 1):
            verdicts[key] = ok
            if i % 5000 == 0:
                print(f"  graded {i}/{len(rows)}", flush=True)

    n_ok = 0
    with open(args.out, "a") as f:
        for r in rows:
            r["is_correct"] = verdicts.get((r["question_id"], r["generation_id"]), False)
            n_ok += bool(r["is_correct"])
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} graded rows -> {args.out}  ({100*n_ok/len(rows):.1f}% correct)")


if __name__ == "__main__":
    main()
