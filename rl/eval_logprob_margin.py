# PRIMARY metric for grader-RL (SPEC §6, spec-review H2): forward-pass
# per-token logprob margin of x+ vs x- under the teacher, on held-out pairs.
#
# This is exactly the quantity the OPSD distillation gradient consumes --
# P(student tokens | teacher context) -- measured with forward passes only,
# no generation. Four cells: {pair context, bare problem} x {adapter on, off}.
# The pair-ablated cells attribute any gain to the pair rather than generic
# competence (spec-review M3). Success = margin improves with the adapter
# under pair context; the pre-registered interpretation (spec §5) keys off
# THIS number, not generative behavior.
#
# Candidates are scored as nothink continuations (enable_thinking=False),
# matching the probe's forward-pass regime and the visible-only exemplars.
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot.probe_adjudication import UNLABELED_TAIL


@torch.no_grad()
def mean_logprob(model, tok, device, context_text, candidate_text):
    """Mean per-token logprob of candidate tokens given context."""
    ctx = tok(context_text, return_tensors="pt", add_special_tokens=False)
    cand = tok(candidate_text, return_tensors="pt", add_special_tokens=False)
    n_ctx = ctx["input_ids"].shape[1]
    ids = torch.cat([ctx["input_ids"], cand["input_ids"]], dim=1).to(device)
    logits = model(ids).logits.float()
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    cand_lp = logprobs[torch.arange(n_ctx - 1, ids.shape[1] - 1),
                       targets[n_ctx - 1:]]
    return cand_lp.mean().item()


def build_contexts(tok, problem, a, b):
    """(pair context, bare context), both ending at the assistant nothink
    generation start so the candidate is scored as the visible response."""
    def render(content):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pair = render(problem + "\n\n" + UNLABELED_TAIL.format(a=a, b=b))
    bare = render(problem)
    return pair, bare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--test_parquet", required=True,
                    help="episode test split; its question_ids select the "
                         "held-out pairs")
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir; omit for base-only run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true",
                    help="build contexts, print one, exit before model load")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = pd.read_parquet(args.pairs).set_index("question_id")
    test = pd.read_parquet(args.test_parquet)
    qids = sorted({e["question_id"] for e in test["extra_info"]})
    qids = [q for q in qids if q in pairs.index]
    print(f"{len(qids)} held-out pairs")

    if args.dry_run:
        p = pairs.loc[qids[0]]
        a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                else (p["x_minus"], p["x_plus"]))
        pair_ctx, bare_ctx = build_contexts(tok, p["problem"], a, b)
        print("=== pair context (tail) ===\n" + pair_ctx[-400:])
        print("=== candidate (head) ===\n" + p["x_plus"][:200])
        return

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = []
    for qid in qids:
        p = pairs.loc[qid]
        a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                else (p["x_minus"], p["x_plus"]))
        pair_ctx, bare_ctx = build_contexts(tok, p["problem"], a, b)
        r = {"question_id": int(qid)}
        for ctx_name, ctx in (("pair", pair_ctx), ("bare", bare_ctx)):
            lp_plus = mean_logprob(model, tok, device, ctx, p["x_plus"])
            lp_minus = mean_logprob(model, tok, device, ctx, p["x_minus"])
            r[f"lp_plus_{ctx_name}"] = lp_plus
            r[f"lp_minus_{ctx_name}"] = lp_minus
            r[f"margin_{ctx_name}"] = lp_plus - lp_minus
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_json(args.out, orient="records", lines=True)
    tag = "adapter" if args.adapter else "base"
    for c in ("pair", "bare"):
        m = df[f"margin_{c}"]
        print(f"[{tag}] {c:4s} context: mean margin {m.mean():+.4f}  "
              f"frac positive {(m > 0).mean():.3f}  n={len(m)}")


if __name__ == "__main__":
    main()
