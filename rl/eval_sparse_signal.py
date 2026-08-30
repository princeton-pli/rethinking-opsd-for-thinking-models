# Sparse-signal diagnostics for contrastive-OPSD (Sanjeev's 2% hypothesis,
# 2026-08-30): a near-zero whole-trace margin is compatible with meaningful
# signal concentrated on few tokens. Two forward-pass measurements on the
# BASE model over harvested WRONG rollouts (x-), where the outcome-gated
# loss would act:
#
# 1. ANSWER-SLOT STEERING. At the wrong trace's own \boxed{ slot, compare
#    mean-per-token logprob of the CORRECT expression (x+'s boxed content)
#    vs the wrong trace's own expression, given ctx + x- up to the slot.
#    The trace favors its own answer by construction; the question is the
#    PAIR-SPECIFIC increment: margin(pair ctx) - margin(bare ctx). If the
#    pair systematically shifts the slot toward the correct expression
#    inside a wrong trace, that is exactly the sparse distillable signal.
#
# 2. WHERE THE PAIR ACTS. Per-token KL( P(.|pair ctx, x-_<t) ||
#    P(.|bare ctx, x-_<t) ) along the wrong rollout: positional profile
#    (deciles) and the share of KL mass on "math" tokens (digits/operators)
#    vs their share of tokens. Concentration on math/decision tokens keeps
#    the sparse mechanism live; uniform spread over prose means the pair is
#    inert even locally.
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.eval_logprob_margin import build_contexts, mean_logprob

BOX = "\\boxed{"
MATH_CHARS = set("0123456789+-*/()=.")


def boxed_content(text):
    i = text.rfind(BOX)
    if i == -1:
        return None, None
    depth, j = 1, i + len(BOX)
    while j < len(text) and depth:
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1
    return text[i + len(BOX):j - 1], i


@torch.no_grad()
def token_kls(model, tok, device, ctx_a, ctx_b, continuation):
    """Per-token KL(P(.|ctx_a,cont_<t) || P(.|ctx_b,cont_<t)) over cont."""
    outs = []
    for ctx in (ctx_a, ctx_b):
        ids_ctx = tok(ctx, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids_cont = tok(continuation, return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
        ids = torch.cat([ids_ctx, ids_cont], dim=1).to(device)
        logits = model(ids).logits[0].float()
        n_ctx = ids_ctx.shape[1]
        # distribution BEFORE each continuation token t: position n_ctx-1+t
        outs.append(torch.log_softmax(logits[n_ctx - 1:-1], dim=-1))
    lp_a, lp_b = outs
    kl = (lp_a.exp() * (lp_a - lp_b)).sum(-1)     # [len(cont)]
    cont_ids = tok(continuation, add_special_tokens=False)["input_ids"]
    return kl.cpu(), [tok.decode([t]) for t in cont_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--n_slot", type=int, default=922)
    ap.add_argument("--n_kl", type=int, default=250)
    ap.add_argument("--template", default="v1", choices=["v1", "v2", "v4"])
    ap.add_argument("--adapter", default=None,
                    help="peft LoRA adapter dir (fp32 over fp32 base)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    pairs = pd.read_parquet(args.pairs)
    rows = []
    kl_deciles = torch.zeros(10)
    kl_counts = torch.zeros(10)
    kl_math_mass, kl_total_mass, n_math_tok, n_tok = 0.0, 0.0, 0, 0

    for k, (_, p) in enumerate(pairs.iterrows()):
        if k >= max(args.n_slot, args.n_kl):
            break
        plus_expr, _ = boxed_content(p["x_plus"])
        minus_expr, bidx = boxed_content(p["x_minus"])
        if plus_expr is None or minus_expr is None:
            continue
        a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                else (p["x_minus"], p["x_plus"]))
        pair_ctx, bare_ctx = build_contexts(tok, p["problem"], a, b,
                                            template=args.template)
        r = {"question_id": int(p["question_id"])}

        if k < args.n_slot:
            slot_prefix = p["x_minus"][:bidx + len(BOX)]
            for ctx_name, ctx in (("pair", pair_ctx), ("bare", bare_ctx)):
                lp_c = mean_logprob(model, tok, device, ctx + slot_prefix,
                                    plus_expr)
                lp_w = mean_logprob(model, tok, device, ctx + slot_prefix,
                                    minus_expr)
                r[f"slot_margin_{ctx_name}"] = lp_c - lp_w
            r["slot_increment"] = (r["slot_margin_pair"]
                                   - r["slot_margin_bare"])

        if k < args.n_kl:
            kl, toks = token_kls(model, tok, device, pair_ctx, bare_ctx,
                                 p["x_minus"])
            T = len(kl)
            dec = (torch.arange(T) * 10 // max(T, 1)).clamp(max=9)
            kl_deciles.index_add_(0, dec, kl)
            kl_counts.index_add_(0, dec, torch.ones(T))
            for t in range(T):
                is_math = any(c in MATH_CHARS for c in toks[t].strip())
                kl_total_mass += kl[t].item()
                n_tok += 1
                if is_math:
                    kl_math_mass += kl[t].item()
                    n_math_tok += 1
            r["kl_mean"] = kl.mean().item()
        rows.append(r)
        if (k + 1) % 100 == 0:
            print(f"{k + 1} problems done", flush=True)

    df = pd.DataFrame(rows)
    df.to_json(args.out, orient="records", lines=True)

    s = df.dropna(subset=["slot_increment"]) if "slot_increment" in df else df
    if len(s):
        inc = s["slot_increment"]
        sem = inc.std() / (len(inc) ** 0.5)
        print(f"\n=== answer-slot steering (n={len(inc)} wrong rollouts) ===")
        print(f"  margin(correct - own expr) | pair ctx: "
              f"{s['slot_margin_pair'].mean():+.4f}")
        print(f"  margin(correct - own expr) | bare ctx: "
              f"{s['slot_margin_bare'].mean():+.4f}")
        print(f"  PAIR-SPECIFIC increment: {inc.mean():+.4f} "
              f"(sem {sem:.4f}, {inc.mean()/sem:+.1f} sigma)  "
              f"frac positive {(inc > 0).mean():.3f}")
    if n_tok:
        print(f"\n=== where the pair acts: KL(pair||bare) along x- "
              f"(n={int(kl_counts.sum())} tokens) ===")
        print("  KL by position decile: "
              + " ".join(f"{(kl_deciles[i]/kl_counts[i].clamp(min=1)):.4f}"
                         for i in range(10)))
        print(f"  math tokens: {n_math_tok/n_tok:.1%} of tokens carry "
              f"{kl_math_mass/max(kl_total_mass,1e-9):.1%} of KL mass")


if __name__ == "__main__":
    main()
