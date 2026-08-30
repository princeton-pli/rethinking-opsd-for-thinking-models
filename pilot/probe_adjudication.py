# Adjudication probe: can the frozen teacher tell which of two unlabeled
# solutions is correct — and does that ability require thinking?
#
# Motivated by the Arm-1 negative result (avg 0.465 vs control 0.542, matched):
# the OPSD teacher never generates, so adjudicating the (x+, x-) pair must happen
# implicitly in a single forward pass. This probe measures the frozen model's
# final-answer accuracy on the SAME problems under a 3x2 grid:
#
#   context:  none            (anchor: the model alone on these problems)
#             unlabeled       (the exact Arm-1 teacher context)
#             neglabel        (Arm 1c: the wrong answer is identified)
#   thinking: off             (proxy for scoring-time forward-pass conditions)
#             on              (what a precomputed-deliberation context could buy)
#
# Readout (per cell, accuracy vs gold):
#   unlabeled/off  ~= none/off        -> pair is invisible to the forward pass
#   unlabeled/off  <  none/off        -> pair actively CONFUSES it (Arm-1 mechanism)
#   unlabeled/on   >> unlabeled/off   -> adjudication needs deliberation -> F3/1d
#   neglabel cells >> unlabeled cells -> the label is load-bearing -> Arm 1c
#
# GENERATION ONLY (no grading in-process: math_equal forks, and forking a
# CUDA-context holder deadlocked job 12413497). Grade afterwards with
# data_tools/grade_jsonl.py, then summarize with pilot/probe_report.py.
import argparse
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

UNLABELED_TAIL = ("Here are two examples of student responses to the question:\n"
                  "{a}\n \n{b}\n \n"
                  "Now answer with a response of your own, including the thinking process.")
NEGLABEL_TAIL = ("Here are two examples of student responses to the question. "
                 "One of them ends in the answer {wrong}, which is known to be incorrect:\n"
                 "{a}\n \n{b}\n \n"
                 "Now answer with a response of your own, including the thinking process.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/contrastive_arm1_pooled_v2.parquet")
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--n_problems", type=int, default=300)
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--out", default="pilot/probe_adjudication.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=None,
                    help="peft LoRA adapter dir; applied via vLLM enable_lora "
                         "(a bf16-merged model would bury a small adapter "
                         "below quantization noise)")
    ap.add_argument("--template", default="v1", choices=["v1", "v2", "v3"],
                    help="v2 = Sanjeev's 2026-08-30 restructured phrasing; "
                         "v3 = v2 + no-reference instruction "
                         "(rl/templates.py); v2/v3 run none+unlabeled cells "
                         "only (neglabel is a dead design)")
    args = ap.parse_args()

    d = pd.read_parquet(args.parquet)
    if len(d) == 0:
        raise SystemExit(f"no pairs in {args.parquet}; nothing to probe")
    # Honest-grader harvests can yield fewer usable pairs than requested
    # (countdown v2: 158 of 2000) — probe whatever exists rather than crash.
    n = min(args.n_problems, len(d))
    if n < args.n_problems:
        print(f"only {len(d)} pairs available (< {args.n_problems}); probing all of them")
    d = d.sample(n=n, random_state=args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)

    prompts, meta = [], []
    for qid, r in d.iterrows():
        # Reconstruct the exemplar order the Arm-1 context used (pos_first).
        a, b = (r["x_plus"], r["x_minus"]) if r["pos_first"] else (r["x_minus"], r["x_plus"])
        if args.template in ("v2", "v3"):
            from rl.templates import (v2_pair_content, v2_bare_content,
                                      v3_pair_content)
            from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE
            nums = [int(x) for x in
                    NUMS_RE.search(r["problem"]).group(1).split(",")]
            target = int(TARGET_RE.search(r["problem"]).group(1))
            build = (v3_pair_content if args.template == "v3"
                     else v2_pair_content)
            contexts = {
                "none": v2_bare_content(nums, target),
                "unlabeled": build(nums, target, a, b),
            }
        else:
            contexts = {
                "none": r["problem"],
                "unlabeled": r["problem"] + "\n\n" + UNLABELED_TAIL.format(a=a, b=b),
                "neglabel": r["problem"] + "\n\n" + NEGLABEL_TAIL.format(
                    a=a, b=b, wrong="\\boxed{" + str(r["wrong_answer"]) + "}"),
            }
        for ctx_name, content in contexts.items():
            for think in (False, True):
                text = tok.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=think)
                prompts.append((text, think))
                meta.append({"question_id": int(r["question_id"]), "cell": f"{ctx_name}/{'think' if think else 'nothink'}",
                             "gold_answer": str(r["answer"]), "source": r.get("source", "?"),
                             "answer_len": len(str(r["answer"]))})

    lora_kwargs, lora_request = {}, None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora_kwargs = {"enable_lora": True, "max_lora_rank": 32}
        lora_request = LoRARequest("adapter", 1, args.adapter)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9,
              **lora_kwargs)
    # Thinking-off responses are short; thinking-on needs full deliberation room.
    sp_think = SamplingParams(n=args.n_samples, temperature=0.6, top_p=0.95, max_tokens=16384)
    sp_nothink = SamplingParams(n=args.n_samples, temperature=0.6, top_p=0.95, max_tokens=2048)

    idx_think = [i for i, (_, t) in enumerate(prompts) if t]
    idx_nothink = [i for i, (_, t) in enumerate(prompts) if not t]

    results = [None] * len(prompts)
    for idxs, sp in ((idx_nothink, sp_nothink), (idx_think, sp_think)):
        outs = llm.generate([prompts[i][0] for i in idxs], sp, use_tqdm=True,
                            lora_request=lora_request)
        for i, out in zip(idxs, outs):
            results[i] = [o.text for o in out.outputs]

    with open(args.out, "w") as f:
        for m, texts in zip(meta, results):
            for k, t in enumerate(texts):
                f.write(json.dumps({**m, "generation_id": k, "response": t}) + "\n")
    print(f"wrote {sum(len(t) for t in results)} generations -> {args.out}")


if __name__ == "__main__":
    main()
