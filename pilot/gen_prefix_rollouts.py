# Thinking-retained rollouts for grader-RL episode prefixes (SPEC §5).
#
# The pair harvest (countdown_pairs.py) stores only post-</think> visible text,
# but RL episodes need prefixes of THINKING traces. Rather than re-harvest,
# this generates fresh rollouts (raw text, <think> retained) for exactly the
# problems that yielded a usable pair. Prompts are the BARE problem -- no pair
# context -- matching the distribution of OPSD student rollouts the teacher
# will eventually score. Because these are a separate batch from the pair
# exemplars, the spec's H1 pair/prefix disjointness holds by construction.
#
# GENERATION ONLY, no grading here (grading is cheap eval() but keep stages
# separable); the episode exporter grades and slices.
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="usable-pairs parquet from countdown_pairs.py")
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_parquet(args.pairs)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(
                   [{"role": "user", "content": r["problem"]}],
                   tokenize=False, add_generation_prompt=True,
                   enable_thinking=True)
               for _, r in d.iterrows()]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9)
    sp = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, top_k=50,
                        max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp, use_tqdm=True)

    n_written = 0
    with open(args.out, "w") as f:
        for (_, r), out in zip(d.iterrows(), outs):
            for gid, o in enumerate(out.outputs):
                f.write(json.dumps({
                    "question_id": int(r["question_id"]),
                    "generation_id": gid,
                    "response": o.text,          # RAW, <think> retained
                    "finish_reason": o.finish_reason,
                }) + "\n")
                n_written += 1
    print(f"wrote {n_written} raw rollouts over {len(d)} problems -> {args.out}")


if __name__ == "__main__":
    main()
