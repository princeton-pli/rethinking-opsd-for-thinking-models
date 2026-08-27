# Harvest (x+, x-) pairs for Countdown, in the same shape the math pool uses, so
# the identical adjudication probe can run on both and the numbers are comparable.
#
# Grading is done INLINE here, which is safe for Countdown specifically:
# CountdownTask.grade is pure eval() plus a multiset compare -- no sympy, no fork,
# so none of the deadlock hazard that forced math grading out of GPU processes.
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluation.tasks import CountdownTask

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\})*[^{}]*)\}")


def row_to_item(row):
    """pandas row -> plain dict, with numpy arrays as lists.

    CountdownTask does `item.get("datapoint_x") or item.get("datapoint_nums")`,
    which raises on a numpy array ("truth value of an array is ambiguous"). The
    repo's own path feeds it HF Dataset rows, where these fields are lists, so
    the idiom is safe there and only breaks for a pandas caller like this one.
    """
    item = {}
    for k, v in row.to_dict().items():
        item[k] = v.tolist() if hasattr(v, "tolist") else v
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countdown", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--n_problems", type=int, default=600)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", default="pilot/countdown_pairs.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(args.countdown).sample(n=args.n_problems, random_state=3).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    task = CountdownTask({})

    prompts = []
    for _, r in df.iterrows():
        msgs = task.format_prompt(row_to_item(r))
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9)
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, top_k=50, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)

    pool, n_solved_all, n_solved_none = [], 0, 0
    for (_, row), out in zip(df.iterrows(), outs):
        item = row_to_item(row)
        gold = task.get_gold(item)
        correct, wrong_by_expr = [], {}
        for o in out.outputs:
            resp = o.text
            visible = THINK_RE.sub("", resp).strip()
            completed = len(visible) > 0 and ("<think>" not in resp or "</think>" in resp)
            if not completed:
                continue
            boxes = {b.strip() for b in BOX_RE.findall(visible)}
            if len(boxes) != 1:           # same consistency rule as the math pool
                continue
            ok = bool(task.grade(resp, gold, item)["is_correct"])
            if ok:
                correct.append((len(visible), visible))
            else:
                wrong_by_expr.setdefault(next(iter(boxes)), []).append((len(visible), visible))
        n_all = sum(1 for o in out.outputs if task.grade(o.text, gold, item)["is_correct"])
        n_solved_all += (n_all == len(out.outputs))
        n_solved_none += (n_all == 0)
        if not correct or not wrong_by_expr:
            continue
        modal_expr, traces = max(wrong_by_expr.items(), key=lambda kv: len(kv[1]))
        pool.append({
            "question_id": int(row.name),
            "problem": item["datapoint_input_text"],
            "answer": str(gold),
            "wrong_answer": modal_expr,
            "x_plus": min(correct)[1],
            "x_minus": min(traces)[1],
            "pos_first": bool(int(row.name) % 2 == 0),
            "source": "countdown",
            "n_correct": n_all,
            "n_gens": len(out.outputs),
        })

    with open(args.out, "w") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    pq = args.out.replace(".jsonl", ".parquet")
    pd.DataFrame(pool).to_parquet(pq, index=False)

    n = len(df)
    print(f"\n=== Countdown pair harvest: {n} problems, k={args.k} ===")
    print(f"  solved ALL {args.k}/{args.k} (too easy) : {n_solved_all} ({100*n_solved_all/n:.1f}%)")
    print(f"  solved NONE (too hard)                  : {n_solved_none} ({100*n_solved_none/n:.1f}%)")
    print(f"  USABLE pairs (>=1 correct, >=1 wrong)   : {len(pool)} ({100*len(pool)/n:.1f}%)")
    print(f"wrote {pq}")


if __name__ == "__main__":
    main()
