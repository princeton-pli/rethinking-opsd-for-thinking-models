# Failure audit (Sanjeev, 2026-09-04): are the arms' losses REAL reasoning
# degradation, or a grading/format artifact?
#
# The student evals stored only per-problem counts, so nothing could be
# inspected after the fact. This regenerates on a subset with EVERYTHING
# retained and classifies each failure:
#
#   no_think_close : hit the token cap mid-<think> (no </think>)  -> truncation
#   no_box         : finished but never emitted \boxed{}          -> format
#   unparseable    : boxed content fails the charset guard        -> format/regex
#   arith_wrong    : parses+evaluates, but != TARGET              -> real error
#   multiset_wrong : evaluates to TARGET, wrong number multiset   -> real error
#   correct
#
# Format-side categories rising in a trained arm would mean the -9/-10 pt
# "damage" is partly a grading artifact (e.g. longer traces -> more truncation).
# Reasoning-side categories rising means the damage is real. Also dumps raw
# failing boxed strings for eyeballing (the standing "regex results are not
# gold" discipline).
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rl.reward_countdown import _last_boxed_content, _FRAC_RE
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


def classify(response, nums, target, finish_reason):
    """Category + the boxed string that was judged."""
    closed = "</think>" in response
    visible = THINK_RE.sub("", response).strip() if closed else response
    clean = visible.replace("$", "").replace(" ", "")
    boxed = _last_boxed_content(clean)
    if not closed and finish_reason == "length":
        return "no_think_close", boxed
    if not boxed:
        return "no_box", boxed
    e = boxed.split("=")[0].strip()
    while r"\frac" in e or r"\dfrac" in e or r"\tfrac" in e:
        ne = _FRAC_RE.sub(r"((\1)/(\2))", e)
        if ne == e:
            break
        e = ne
    e = (e.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
         .replace(r"\left", "").replace(r"\right", "")
         .replace("{", "(").replace("}", ")"))
    if not re.match(r"^[\d\+\-\*\/\(\)\.]+$", e):
        return "unparseable", boxed
    try:
        val = eval(e)
    except Exception:
        return "unparseable", boxed
    if not isinstance(val, (int, float)) or abs(val - float(target)) > 1e-5:
        return "arith_wrong", boxed
    used = Counter(float(x) for x in re.findall(r"\d+(?:\.\d+)?", e))
    if used != Counter(float(n) for n in nums):
        return "multiset_wrong", boxed
    return "correct", boxed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n_problems", type=int, default=120)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_parquet(args.heldout).head(args.n_problems)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = [tok.apply_chat_template(
                   [{"role": "user", "content": r["problem"]}],
                   tokenize=False, add_generation_prompt=True,
                   enable_thinking=True)
               for _, r in d.iterrows()]

    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16",
              gpu_memory_utilization=0.9)
    outs = llm.generate(prompts, SamplingParams(
        n=args.k, temperature=1.0, top_p=1.0, top_k=50,
        max_tokens=args.max_tokens), use_tqdm=True)

    cats, rows = Counter(), []
    for (_, r), out in zip(d.iterrows(), outs):
        nums = [int(x) for x in NUMS_RE.search(r["problem"]).group(1).split(",")]
        target = int(TARGET_RE.search(r["problem"]).group(1))
        for o in out.outputs:
            cat, boxed = classify(o.text, nums, target, o.finish_reason)
            cats[cat] += 1
            rows.append({"question_id": int(r["question_id"]), "cat": cat,
                         "boxed": boxed, "n_tokens": len(o.token_ids),
                         "finish": o.finish_reason, "nums": nums,
                         "target": target,
                         "tail": o.text[-400:]})

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    tot = sum(cats.values())
    print(f"\n=== [{args.tag}] failure audit, {tot} generations ===")
    for c in ["correct", "arith_wrong", "multiset_wrong", "no_think_close",
              "no_box", "unparseable"]:
        print(f"  {c:16s} {cats[c]:5d}  {cats[c]/max(tot,1):6.1%}")
    print(f"\n  FORMAT-side failures (possible artifact): "
          f"{(cats['no_think_close']+cats['no_box']+cats['unparseable'])/max(tot,1):.1%}")
    print(f"  REASONING-side failures (real):            "
          f"{(cats['arith_wrong']+cats['multiset_wrong'])/max(tot,1):.1%}")
    print(f"\n--- sample failing boxed strings ---")
    shown = Counter()
    for row in rows:
        if row["cat"] == "correct" or shown[row["cat"]] >= 4:
            continue
        shown[row["cat"]] += 1
        print(f"  [{row['cat']}] nums={row['nums']} target={row['target']} "
              f"boxed={row['boxed']!r}")


if __name__ == "__main__":
    main()
