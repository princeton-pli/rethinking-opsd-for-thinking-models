# Build verl episode parquets for grader-RL (SPEC §5).
#
# Inputs: a usable-pairs parquet (countdown_pairs.py) + thinking-retained
# prefix rollouts (gen_prefix_rollouts.py). Output: train/test parquet whose
# `prompt` column is a PLAIN STRING rendered for PrefixContinuationDataset:
#
#   <|im_start|>user\n{problem + unlabeled-pair context}<|im_end|>\n
#   <|im_start|>assistant\n<think>\n{prefix}
#
# ending mid-think (no </think>): the policy continues the trace.
#
# H1 disjointness: prefixes come from a SEPARATE generation batch than the
# pair exemplars (different file), and we additionally assert no prefix
# rollout's visible text equals x_plus/x_minus for its problem.
# Prefix length: uniform fraction in [0, 1/2] of the rollout's think tokens.
# Prefix type (right/wrong final answer) recorded for the L1 diet logging.
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from rl.reward_countdown import grade_countdown
from pilot.grade_countdown_probe import NUMS_RE
from pilot.probe_adjudication import UNLABELED_TAIL

THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--model", default="/scratch/gpfs/ARORA/skaur/models/Qwen3-4B")
    ap.add_argument("--max_prompt_tokens", type=int, default=12288)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = pd.read_parquet(args.pairs).set_index("question_id")

    by_q = {}
    with open(args.prefixes) as f:
        for line in f:
            r = json.loads(line)
            by_q.setdefault(r["question_id"], []).append(r)

    rows, dropped_overlong, dropped_nothink, dup_of_exemplar = [], 0, 0, 0
    for qid, rollouts in sorted(by_q.items()):
        if qid not in pairs.index:
            continue
        p = pairs.loc[qid]
        nums = [int(x) for x in NUMS_RE.search(p["problem"]).group(1).split(",")]
        a, b = ((p["x_plus"], p["x_minus"]) if p["pos_first"]
                else (p["x_minus"], p["x_plus"]))
        content = p["problem"] + "\n\n" + UNLABELED_TAIL.format(a=a, b=b)
        head = ("<|im_start|>user\n" + content + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n")
        head_ids = tok.encode(head, add_special_tokens=False)

        for r in rollouts:
            m = THINK_RE.search(r["response"])
            if not m:                          # truncated / no closed think
                dropped_nothink += 1
                continue
            visible = THINK_RE.sub("", r["response"]).strip()
            if visible in (p["x_plus"], p["x_minus"]):   # H1 paranoia assert
                dup_of_exemplar += 1
                continue
            think = m.group(1).strip("\n")
            think_ids = tok.encode(think, add_special_tokens=False)
            frac = rng.uniform(0.0, 0.5)
            prefix_ids = think_ids[:int(len(think_ids) * frac)]
            if len(head_ids) >= args.max_prompt_tokens:
                dropped_overlong += 1        # pair context alone too long
                continue
            if len(head_ids) + len(prefix_ids) > args.max_prompt_tokens:
                prefix_ids = prefix_ids[:args.max_prompt_tokens - len(head_ids)]
            prompt = head + tok.decode(prefix_ids)
            right = grade_countdown(visible, p["answer"], nums) == 1.0
            rows.append({
                "data_source": "grader_prefix_countdown",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule",
                                 "ground_truth": str(p["answer"])},
                "extra_info": {
                    "index": len(rows), "question_id": int(qid),
                    "nums": nums, "prefix_type": "right" if right else "wrong",
                    "prefix_frac": round(frac, 4),
                    "prefix_gen_id": int(r["generation_id"]),
                },
            })

    qids = sorted({r["extra_info"]["question_id"] for r in rows})
    rng.shuffle(qids)
    test_q = set(qids[:max(1, int(len(qids) * args.test_frac))])
    train = [r for r in rows if r["extra_info"]["question_id"] not in test_q]
    test = [r for r in rows if r["extra_info"]["question_id"] in test_q]
    for split, data in (("train", train), ("test", test)):
        for i, r in enumerate(data):
            r["extra_info"]["split"] = split
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(train).to_parquet(f"{args.out_dir}/train.parquet", index=False)
    pd.DataFrame(test).to_parquet(f"{args.out_dir}/test.parquet", index=False)

    n_right = sum(1 for r in rows if r["extra_info"]["prefix_type"] == "right")
    print(f"{len(train)} train / {len(test)} test episodes over {len(qids)} "
          f"problems (split by problem)")
    print(f"  prefix diet: {n_right} right / {len(rows) - n_right} wrong")
    print(f"  dropped: {dropped_nothink} unfinished-think, "
          f"{dropped_overlong} overlong, {dup_of_exemplar} exemplar-dup")


if __name__ == "__main__":
    main()
