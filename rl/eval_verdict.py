# Vanilla single-answer grading probe (Sanjeev, 2026-08-31): one question,
# ONE candidate attempt, verdict correct-or-not. All prior grading evals are
# pair-based (a contrast crutch); this measures absolute verification, and --
# since the RL never trained verdict emission -- transfer from
# completion-training to explicit verification.
#
# Cells: {x+, x-} x {think, nothink}, temperature 0, n=1. For nothink we also
# record first-token logprobs, giving a calibrated P(Yes)-style classifier
# score. Ground truth is the exemplar label (audited: 0.3% noise).
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

from rl.templates import V2_HEAD
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

VERDICT_TAIL = (
    "\nBelow is a student attempt at this instance.\n\n"
    "ATTEMPT:\n{x}\n\n"
    "Is this attempt a correct solution? Answer yes or no."
)

# Sanjeev 2026-08-31: rule-by-rule decision checklist.
VERDICT_CHECK_TAIL = (
    "\nBelow is a student attempt at this instance.\n\n"
    "ATTEMPT:\n{x}\n\n"
    "Check the attempt against each rule, one at a time:\n"
    "1. Write out the numbers used in its final expression, with "
    "multiplicity. Is this exactly the multiset LIST?\n"
    "2. Evaluate the final expression step by step. Does it equal TARGET?\n"
    "3. Does it use only +, -, *, / and parentheses?\n"
    "Then answer yes if the attempt is a correct solution, or no if it "
    "is not."
)
YES_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n_problems", type=int, default=300)
    ap.add_argument("--checklist", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tail = VERDICT_CHECK_TAIL if args.checklist else VERDICT_TAIL

    df = pd.read_parquet(args.pairs).sample(
        n=args.n_problems, random_state=0)
    tok = AutoTokenizer.from_pretrained(args.model)

    prompts, meta = [], []
    for _, r in df.iterrows():
        nums = [int(x) for x in NUMS_RE.search(r["problem"]).group(1).split(",")]
        target = int(TARGET_RE.search(r["problem"]).group(1))
        head = V2_HEAD.format(nums=", ".join(str(n) for n in nums),
                              target=target)
        for cand, text in (("plus", r["x_plus"]), ("minus", r["x_minus"])):
            content = head + tail.format(x=text)
            for think in (False, True):
                prompts.append((tok.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=think), think))
                meta.append({"question_id": int(r["question_id"]),
                             "candidate": cand,
                             "cell": "think" if think else "nothink"})

    lora_kwargs, lora_request = {}, None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lora_kwargs = {"enable_lora": True, "max_lora_rank": 32}
        lora_request = LoRARequest("adapter", 1, args.adapter)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.9,
              **lora_kwargs)
    sp_no = SamplingParams(temperature=0.0, max_tokens=512, logprobs=20)
    sp_th = SamplingParams(temperature=0.0, max_tokens=8192)

    idx = {False: [i for i, (_, t) in enumerate(prompts) if not t],
           True: [i for i, (_, t) in enumerate(prompts) if t]}
    results = [None] * len(prompts)
    for think, sp in ((False, sp_no), (True, sp_th)):
        outs = llm.generate([prompts[i][0] for i in idx[think]], sp,
                            use_tqdm=True, lora_request=lora_request)
        for i, out in zip(idx[think], outs):
            o = out.outputs[0]
            row = {"text": o.text}
            if not think and o.logprobs:
                first = o.logprobs[0]
                row["first_token_lps"] = {
                    tok.decode([tid]): lp.logprob
                    for tid, lp in list(first.items())[:20]}
            results[i] = row

    n = {"plus": [0, 0, 0, 0], "minus": [0, 0, 0, 0]}  # per cand: [ok_no, tot_no, ok_th, tot_th]
    with open(args.out, "w") as f:
        for m, res in zip(meta, results):
            vis = res["text"].split("</think>")[-1]
            # LAST yes/no: under a checklist the early ones answer per-rule
            # sub-questions, the final one is the verdict.
            matches = YES_RE.findall(vis)
            verdict = matches[-1].lower() if matches else None
            want = "yes" if m["candidate"] == "plus" else "no"
            ok = verdict == want
            j = 0 if m["cell"] == "nothink" else 2
            n[m["candidate"]][j] += ok
            n[m["candidate"]][j + 1] += 1
            f.write(json.dumps({**m, "verdict": verdict, "ok": ok,
                                "text": res["text"],
                                **({"first_token_lps": res["first_token_lps"]}
                                   if "first_token_lps" in res else {})}) + "\n")

    tag = "adapter" if args.adapter else "base"
    for cell, j in (("nothink", 0), ("think", 2)):
        acc_p = n["plus"][j] / max(n["plus"][j + 1], 1)
        acc_m = n["minus"][j] / max(n["minus"][j + 1], 1)
        print(f"[{tag}] {cell:8s}: x+ correctly accepted {acc_p:.3f} | "
              f"x- correctly rejected {acc_m:.3f} | balanced "
              f"{(acc_p + acc_m) / 2:.3f}")


if __name__ == "__main__":
    main()
