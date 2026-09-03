# Does the OPSD teacher PREFER the student's wrong traces? (Sanjeev, 2026-09-03)
#
# The arms minimize JSD(student || teacher) on the student's WRONG rollouts,
# pulling the student toward the teacher there. The teacher's context contains
# x-, an example of exactly that kind of wrong reasoning. If pair-conditioning
# RAISES the likelihood of the wrong trace relative to the unconditioned model,
# then the distillation target is more wrong than the student -- correct loss
# sign, harmful effect, and a clean explanation for the -9/-10 pt damage.
#
# Measures mean per-token logprob of WRONG rollout tokens under:
#   bare      : base model, no pair            (the student's own prior)
#   pair_base : base model + v4 pair context   (arm B's teacher)
#   pair_rl   : RL'd teacher + v4 pair context (arm A/C's teacher)
# Reports paired deltas vs bare. Positive delta = teacher likes the wrong
# trace MORE than the unconditioned model does.
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.templates import v2_bare_content, v4_pair_content
from rl.reward_countdown import grade_countdown
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


@torch.no_grad()
def mean_logp(model, tok, device, ctx, cont, max_cont_tokens=3000):
    ctx_ids = tok(ctx, return_tensors="pt", add_special_tokens=False)["input_ids"]
    cont_ids = tok(cont, return_tensors="pt",
                   add_special_tokens=False)["input_ids"][:, :max_cont_tokens]
    n_ctx = ctx_ids.shape[1]
    ids = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    logits = model(ids).logits.float()
    lp = torch.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    sel = lp[torch.arange(n_ctx - 1, ids.shape[1] - 1), tgt[n_ctx - 1:]]
    return sel.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--adapter", required=True, help="RL'd teacher adapter")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    pool = pd.read_parquet(args.pool).set_index("question_id")

    # Collect WRONG rollouts with complete traces.
    items = []
    for f in args.rollouts:
        for line in open(f):
            if len(items) >= args.n:
                break
            r = json.loads(line)
            qid = int(r["question_id"])
            if qid not in pool.index:
                continue
            p = pool.loc[qid]
            nums = [int(x) for x in NUMS_RE.search(p["problem"]).group(1).split(",")]
            target = int(TARGET_RE.search(p["problem"]).group(1))
            vis = THINK_RE.sub("", r["response"]).strip()
            if "</think>" not in r["response"]:
                continue
            if grade_countdown(vis, str(target), nums) == 1.0:
                continue                       # want WRONG only
            items.append((qid, p, nums, target, r["response"]))

    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=device).eval()
    from peft import PeftModel
    rl = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map=device),
        args.adapter).eval()

    def render(content):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)

    rows = []
    for qid, p, nums, target, resp in items:
        a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                else (p["x_minus"], p["x_plus"]))
        bare_ctx = render(v2_bare_content(nums, target))
        pair_ctx = render(v4_pair_content(nums, target, a, b))
        rows.append({
            "question_id": qid,
            "bare": mean_logp(base, tok, device, bare_ctx, resp),
            "pair_base": mean_logp(base, tok, device, pair_ctx, resp),
            "pair_rl": mean_logp(rl, tok, device, pair_ctx, resp),
        })
        if len(rows) % 25 == 0:
            print(f"{len(rows)}/{len(items)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_json(args.out, orient="records", lines=True)
    print(f"\n=== teacher likelihood of the student's WRONG traces (n={len(df)}) ===")
    print(f"  bare (no pair, base model):  {df['bare'].mean():+.4f}")
    for col, name in (("pair_base", "arm B teacher (base+pair)"),
                      ("pair_rl", "arm A/C teacher (RL'd+pair)")):
        d = df[col] - df["bare"]
        sem = d.std() / (len(d) ** 0.5)
        print(f"  {name}: {df[col].mean():+.4f} | delta {d.mean():+.4f} "
              f"(sem {sem:.4f}, {d.mean()/max(sem,1e-9):+.1f} sigma) "
              f"| frac positive {(d > 0).mean():.3f}")
    print("\n  POSITIVE delta = teacher finds the WRONG trace MORE likely than")
    print("  the unconditioned model -> distilling toward it teaches wrongness.")


if __name__ == "__main__":
    main()
