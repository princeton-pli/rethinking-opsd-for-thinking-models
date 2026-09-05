# Does the teacher ENCOURAGE the self-cancelling cheat? (Sanjeev, 2026-09-04)
#
# Sanjeev's objection: if the student writes "+(83-83)", the teacher should
# give it LOW probability, so JSD should push the student away -- yet OPSD
# training raises the cheat rate 30% -> 43%.
#
# The earlier whole-trace measurement (teacher -0.086 vs base) cannot answer
# this: it averages ~10k tokens, and the cheat is ~10 of them. Same trap as
# the whole-trace margin (~0) vs the answer slot (+0.84).
#
# Hypothesis: x- sits IN the teacher's context and is ITSELF a
# self-cancelling-cheat solution 65% of the time (194/300 constraint
# violations vs 32 arithmetic errors). An in-context demonstration RAISES the
# probability of the pattern it exemplifies, so the teacher may score the
# cheat span HIGHER than the unconditioned model even while disliking the
# whole wrong trace.
#
# Measures mean per-token logprob of the CHEAT SPAN ONLY (the boxed
# expression of a wrong trace that reuses numbers), under:
#   base_bare : base model, no pair
#   pair_base : base model + v4 pair context     (arm B's teacher)
#   pair_rl   : RL'd teacher + v4 pair context   (arm A/C's teacher)
# Split by whether the pair's OWN x- contains a cheat -- if the effect is
# driven by contaminated exemplars, the gap should be far larger there.
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.templates import v2_bare_content, v4_pair_content, v5_pair_content
from rl.reward_countdown import _last_boxed_content
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


def reuses_numbers(boxed, nums):
    """True if the boxed expression uses a number more often than allowed."""
    if not boxed:
        return False
    e = boxed.split("=")[0]
    used = Counter(float(x) for x in re.findall(r"\d+(?:\.\d+)?", e))
    avail = Counter(float(n) for n in nums)
    return any(used[k] > avail.get(k, 0) for k in used)


@torch.no_grad()
def span_logp(model, tok, device, ctx, prefix, span, max_prefix_tokens=3000):
    """Mean per-token logprob of `span`, given ctx + (truncated) prefix.

    The prefix is capped to its LAST max_prefix_tokens tokens: full traces run
    10k+ tokens and two fp32 4B models OOM'd on the raw version. Both models
    see the identical truncated context, so the comparison is unaffected.
    """
    ctx_ids = tok(ctx, return_tensors="pt", add_special_tokens=False)["input_ids"]
    pre_ids = tok(prefix, return_tensors="pt",
                  add_special_tokens=False)["input_ids"][:, -max_prefix_tokens:]
    head = torch.cat([ctx_ids, pre_ids], dim=1)
    tail = tok(span, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if tail.shape[1] == 0:
        return None
    n = head.shape[1]
    ids = torch.cat([head, tail], dim=1).to(device)
    lp = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
    tgt = ids[0, 1:]
    return lp[torch.arange(n - 1, ids.shape[1] - 1), tgt[n - 1:]].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    pool = pd.read_parquet(args.pool).set_index("question_id")

    items = []
    for f in args.rollouts:
        for line in open(f):
            if len(items) >= args.n:
                break
            r = json.loads(line)
            qid = int(r["question_id"])
            if qid not in pool.index or "</think>" not in r["response"]:
                continue
            p = pool.loc[qid]
            nums = [int(x) for x in NUMS_RE.search(p["problem"]).group(1).split(",")]
            target = int(TARGET_RE.search(p["problem"]).group(1))
            vis = THINK_RE.sub("", r["response"]).strip()
            boxed = _last_boxed_content(vis.replace("$", "").replace(" ", ""))
            # want WRONG rollouts whose failure IS the reuse cheat
            if not reuses_numbers(boxed, nums):
                continue
            i = vis.rfind("\\boxed{")
            if i == -1:
                continue
            items.append((qid, p, nums, target, r["response"], vis, i, boxed))

    device = "cuda"

    def render(c):
        return tok.apply_chat_template([{"role": "user", "content": c}],
                                       tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True)

    # Precompute contexts/prefixes/spans once.
    work = []
    for qid, p, nums, target, resp, vis, i, boxed in items:
        a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                else (p["x_minus"], p["x_plus"]))
        think_end = resp.find("</think>") + len("</think>")
        xm_boxed = _last_boxed_content(
            str(p["x_minus"]).replace("$", "").replace(" ", ""))
        work.append({
            "question_id": qid,
            "xminus_is_cheat": bool(reuses_numbers(xm_boxed, nums)),
            "bare_ctx": render(v2_bare_content(nums, target)),
            "pair_ctx": render(v4_pair_content(nums, target, a, b)),
            "warn_ctx": render(v5_pair_content(nums, target, a, b)),
            "prefix": resp[:think_end] + vis[:i],
            "span": vis[i:i + len("\\boxed{") + len(boxed) + 1],
        })

    # ONE model resident at a time: two fp32 4B models OOM'd together.
    from peft import PeftModel
    for pass_name in ("base", "rl"):
        m = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map=device)
        if pass_name == "rl":
            m = PeftModel.from_pretrained(m, args.adapter)
        m.eval()
        for j, w in enumerate(work):
            if pass_name == "base":
                w["base_bare"] = span_logp(m, tok, device, w["bare_ctx"],
                                           w["prefix"], w["span"])
                w["pair_base"] = span_logp(m, tok, device, w["pair_ctx"],
                                           w["prefix"], w["span"])
                w["warn_base"] = span_logp(m, tok, device, w["warn_ctx"],
                                           w["prefix"], w["span"])
            else:
                w["pair_rl"] = span_logp(m, tok, device, w["pair_ctx"],
                                         w["prefix"], w["span"])
                w["warn_rl"] = span_logp(m, tok, device, w["warn_ctx"],
                                         w["prefix"], w["span"])
            if (j + 1) % 25 == 0:
                print(f"[{pass_name}] {j + 1}/{len(work)}", flush=True)
        del m
        torch.cuda.empty_cache()

    df = pd.DataFrame([{k: v for k, v in w.items()
                        if k not in ("bare_ctx", "pair_ctx", "warn_ctx",
                                     "prefix", "span")}
                       for w in work]).dropna()
    df.to_json(args.out, orient="records", lines=True)

    def report(sub, label):
        if not len(sub):
            return
        print(f"\n--- {label} (n={len(sub)}) ---")
        print(f"  base_bare (no pair):      {sub['base_bare'].mean():+.4f}")
        for col, nm in (("pair_base", "base+pair (armB tchr)"),
                        ("warn_base", "base+pair+WARNING  "),
                        ("pair_rl", "RL'd+pair (armA tchr)"),
                        ("warn_rl", "RL'd+pair+WARNING  ")):
            d = sub[col] - sub["base_bare"]
            sem = d.std() / max(len(d) ** 0.5, 1)
            print(f"  {nm:16s} {sub[col].mean():+.4f} | delta {d.mean():+.4f} "
                  f"(sem {sem:.4f}, {d.mean()/max(sem,1e-9):+.1f} sigma) "
                  f"| frac positive {(d > 0).mean():.3f}")

    print(f"\n=== logprob of the CHEAT SPAN itself (n={len(df)}) ===")
    report(df, "ALL cheat spans")
    report(df[df.xminus_is_cheat], "pair's x- IS also a cheat (contaminated)")
    report(df[~df.xminus_is_cheat], "pair's x- is NOT a cheat (clean)")
    print("\n  POSITIVE delta = the teacher ENCOURAGES the cheat relative to")
    print("  the unconditioned model -> distillation transmits it.")


if __name__ == "__main__":
    main()
