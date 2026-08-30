# Mid-trace steering measurement (Sanjeev, 2026-08-30): does the pair-informed
# teacher push back at the point where a wrong THINKING trace makes its first
# arithmetic slip?
#
# This is the load-bearing question for token-weighted OPSD vs a plain
# STaR/RFT baseline: slot-only signal ~= "fix the final answer" (SFT on x+
# does that); MID-TRACE signal is what only distillation can transmit.
#
# Method: over raw wrong thinking rollouts (gen_prefix_rollouts output),
# find the first arithmetic statement "a op b = c" whose claim is false;
# score, given the identical context + trace-up-to-the-result, the stated
# FALSE result vs the arithmetically TRUE result, under pair vs bare context
# (v2 template). Report the pair-specific effects:
#   d_false = lp_pair(false) - lp_bare(false)   (negative = pair suppresses slip)
#   d_true  = lp_pair(true)  - lp_bare(true)    (positive = pair steers to truth)
#   margin  = lp(true) - lp(false) under each context.
# Also reports slip prevalence -- if slips are rare, wrongness on this task is
# constraint accounting, not arithmetic, which itself shapes the weighting.
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.eval_logprob_margin import mean_logprob
from rl.templates import v2_pair_content, V2_HEAD, V2_BARE
from rl.reward_countdown import grade_countdown
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
EQ_RE = re.compile(r"(\d+)\s*([-+*/×÷x])\s*(\d+)\s*=\s*(-?\d+)")
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "x": lambda a, b: a * b,
       "×": lambda a, b: a * b,
       "/": lambda a, b: a / b if b else None,
       "÷": lambda a, b: a / b if b else None}


def first_slip(think):
    """(char_pos_of_result, false_result_str, true_result_str) or None."""
    for m in EQ_RE.finditer(think):
        a, op, b, stated = (int(m.group(1)), m.group(2), int(m.group(3)),
                            int(m.group(4)))
        true = OPS[op](a, b)
        if true is None or not float(true).is_integer():
            continue
        true = int(true)
        if true != stated:
            return m.start(4), str(stated), str(true)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--max_n", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=device)
    model.eval()

    pairs = pd.read_parquet(args.pairs).set_index("question_id")

    def render(content):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)

    n_wrong, n_slip, rows = 0, 0, []
    with open(args.rollouts) as f:
        for line in f:
            if n_wrong >= args.max_n:
                break
            r = json.loads(line)
            if r["question_id"] not in pairs.index:
                continue
            p = pairs.loc[r["question_id"]]
            m = THINK_RE.search(r["response"])
            if not m:
                continue
            nums = [int(x) for x in
                    NUMS_RE.search(p["problem"]).group(1).split(",")]
            target = int(TARGET_RE.search(p["problem"]).group(1))
            visible = THINK_RE.sub("", r["response"]).strip()
            if grade_countdown(visible, str(target), nums) == 1.0:
                continue                    # only wrong rollouts
            n_wrong += 1
            think = m.group(1)
            slip = first_slip(think)
            if slip is None:
                continue
            n_slip += 1
            pos, false_s, true_s = slip
            # trace prefix up to (not including) the stated result
            head = r["response"][:r["response"].find(think) + pos]
            a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                    else (p["x_minus"], p["x_plus"]))
            pair_ctx = render(v2_pair_content(nums, target, a, b)) + head
            bare_ctx = render(
                V2_HEAD.format(nums=", ".join(str(n) for n in nums),
                               target=target) + V2_BARE) + head
            row = {"question_id": int(r["question_id"]),
                   "generation_id": int(r["generation_id"])}
            for ctx_name, ctx in (("pair", pair_ctx), ("bare", bare_ctx)):
                row[f"lp_false_{ctx_name}"] = mean_logprob(
                    model, tok, device, ctx, false_s)
                row[f"lp_true_{ctx_name}"] = mean_logprob(
                    model, tok, device, ctx, true_s)
            row["d_false"] = row["lp_false_pair"] - row["lp_false_bare"]
            row["d_true"] = row["lp_true_pair"] - row["lp_true_bare"]
            rows.append(row)
            if n_slip % 50 == 0:
                print(f"{n_slip} slips / {n_wrong} wrong rollouts", flush=True)

    df = pd.DataFrame(rows)
    df.to_json(args.out, orient="records", lines=True)
    print(f"\n=== first arithmetic slip in wrong thinking traces ===")
    print(f"  slip prevalence: {n_slip}/{n_wrong} wrong rollouts "
          f"({n_slip/max(n_wrong,1):.1%})")
    if len(df):
        for col, desc in (("d_false", "pair effect on FALSE result lp"),
                          ("d_true", "pair effect on TRUE result lp")):
            v = df[col]
            sem = v.std() / (len(v) ** 0.5)
            print(f"  {desc}: {v.mean():+.4f} (sem {sem:.4f}, "
                  f"{v.mean()/max(sem,1e-9):+.1f} sigma)")
        for ctx in ("pair", "bare"):
            mg = df[f"lp_true_{ctx}"] - df[f"lp_false_{ctx}"]
            print(f"  margin(true-false) | {ctx}: {mg.mean():+.4f}  "
                  f"frac positive {(mg > 0).mean():.3f}")


if __name__ == "__main__":
    main()
