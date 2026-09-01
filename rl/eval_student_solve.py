# PRIMARY outcome for the OPSD arms: held-out hard-Countdown solve rate of the
# STUDENT, from the bare question (no pair, no privileged context) -- the
# distribution the student will actually face.
#
# Reports avg@k (mean per-sample correctness) and pass@k (any-of-k), graded by
# the deterministic task grader (value + exactly-once multiset, strip-at-'='),
# plus mean/median completion length -- the Arm-1 damage canary.
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

from rl.reward_countdown import grade_countdown
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="student checkpoint dir")
    ap.add_argument("--tokenizer", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_parquet(args.heldout)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = [tok.apply_chat_template(
                   [{"role": "user", "content": r["problem"]}],
                   tokenize=False, add_generation_prompt=True,
                   enable_thinking=True)
               for _, r in d.iterrows()]

    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16",
              gpu_memory_utilization=0.9)
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, top_k=50,
                        max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)

    n_correct = n_gen = n_solved = 0
    lens, rows = [], []
    for (_, r), out in zip(d.iterrows(), outs):
        nums = [int(x) for x in NUMS_RE.search(r["problem"]).group(1).split(",")]
        target = int(TARGET_RE.search(r["problem"]).group(1))
        oks = []
        for o in out.outputs:
            visible = THINK_RE.sub("", o.text).strip()
            ok = grade_countdown(visible, str(target), nums) == 1.0
            oks.append(ok)
            lens.append(len(o.token_ids))
        n_correct += sum(oks)
        n_gen += len(oks)
        n_solved += any(oks)
        rows.append({"question_id": int(r["question_id"]),
                     "n_correct": sum(oks), "k": len(oks)})

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    lens.sort()
    print(f"[{args.tag}] problems={len(d)} k={args.k} | "
          f"avg@{args.k}={n_correct / max(n_gen, 1):.4f} | "
          f"pass@{args.k}={n_solved / max(len(d), 1):.4f} | "
          f"mean_len={sum(lens) / max(len(lens), 1):.0f} | "
          f"median_len={lens[len(lens) // 2] if lens else 0}")


if __name__ == "__main__":
    main()
