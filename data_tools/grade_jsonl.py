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
    """Grade one row. Returns (key, True | False | 'timeout' | 'error'). Never raises.

    The three-state verdict matters: an earlier version collapsed timeouts and
    crashes into False, which is how a total grading failure could have shipped
    looking like a plausible set of wrong answers.
    """
    key, response, gold = payload

    def _alarm(signum, frame):
        raise TimeoutError()

    # Hard per-row ceiling on top of math_equal's own timeout: sympy can wedge
    # inside a single call, and a wedged worker would stall the whole run.
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(20)
    try:
        return key, bool(_HANDLER.grade(response, gold, {})["is_correct"])
    except TimeoutError:
        return key, "timeout"
    except Exception:
        return key, "error"
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
    from concurrent.futures import ProcessPoolExecutor

    for r in rows:
        assert "gold_answer" in r, f"row {r['question_id']}/{r['generation_id']} has no gold_answer"
    payloads = [((r["question_id"], r["generation_id"]), r["response"], r["gold_answer"])
                for r in rows]
    by_key = {(r["question_id"], r["generation_id"]): r for r in rows}

    # ProcessPoolExecutor, NOT multiprocessing.Pool: Pool workers are DAEMONIC, and
    # a daemonic process cannot spawn children. math_equal implements its timeout by
    # spawning one, so under Pool every sympy-requiring comparison raised
    # AssertionError -> caught below -> silently graded False. That would have
    # mislabelled every non-trivially-equal answer (\frac{7}{2} vs 3.5, 2\sqrt{3} vs
    # 3.464, ...) across the whole training set. ProcessPoolExecutor workers are
    # non-daemonic, so the nested spawn works.
    ctx = mp.get_context("fork")
    n_ok = n_to = n_err = 0
    written = 0
    # Ensure we never append onto a half-written line from an interrupted run.
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(args.out, "a") as fa:
                    fa.write("\n")

    # Write incrementally: a 40min-2h pass killed near the end must not lose everything.
    with open(args.out, "a") as fout, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx, initializer=_init, initargs=(args.task,)
    ) as pool:
        for i, (key, verdict) in enumerate(pool.map(_grade_one, payloads, chunksize=8), 1):
            r = by_key[key]
            r["is_correct"] = verdict is True
            n_ok += r["is_correct"]
            n_to += verdict == "timeout"
            n_err += verdict == "error"
            fout.write(json.dumps(r) + "\n")
            written += 1
            if i % 1000 == 0:
                fout.flush()
                print(f"  graded {i}/{len(rows)}  ok={n_ok} timeout={n_to} err={n_err}", flush=True)

    rate = 100 * n_ok / max(written, 1)
    print(f"wrote {written} graded rows -> {args.out}  ({rate:.1f}% correct, "
          f"{n_to} timeouts, {n_err} errors)")
    # A grading pass that silently degenerates (e.g. every comparison failing) is the
    # exact failure this file exists to prevent -- fail loudly instead of shipping it.
    assert 2.0 <= rate <= 90.0, f"implausible correct rate {rate:.1f}% -- grading is broken"
    assert n_err <= 0.01 * written, f"{n_err} grading errors ({100*n_err/written:.1f}%) -- investigate"


if __name__ == "__main__":
    main()
