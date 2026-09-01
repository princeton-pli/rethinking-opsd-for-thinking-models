# STaR control, FIXED (arm D2): SFT on full CORRECT rollouts -- reasoning
# trace AND answer -- not the think-stripped x_plus that arm D used.
#
# Arm D trained on pilot/countdown_pairs.py's x_plus, which has
# <think>...</think> removed at harvest, so it taught answer-without-thinking
# (1,182-token outputs vs base 9,500) and its -42.8 pt result measures channel
# destruction, not self-training. This builder takes RAW rollouts (think
# retained), grades them with the deterministic task grader, and keeps the
# correct ones verbatim.
#
# Source: rollout jsonl files with {question_id, response} where response is
# raw text including <think>. Held-out question_ids are excluded by taking
# the train split's qid set.
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from rl.reward_countdown import grade_countdown
from pilot.grade_countdown_probe import NUMS_RE, TARGET_RE

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="countdown_opsd_train.parquet")
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--qid_offset", nargs="+", type=int, default=None,
                    help="per-rollout-file offset to match pool question_ids")
    ap.add_argument("--max_per_problem", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = pd.read_parquet(args.train)
    keep_qids = set(train["question_id"].tolist())
    prob = dict(zip(train["question_id"], train["problem"]))

    offsets = args.qid_offset or [0] * len(args.rollouts)
    assert len(offsets) == len(args.rollouts)

    rows, per_q = [], {}
    n_seen = n_correct = 0
    for f, off in zip(args.rollouts, offsets):
        for line in open(f):
            r = json.loads(line)
            qid = int(r["question_id"]) + off
            if qid not in keep_qids:
                continue
            n_seen += 1
            p = prob[qid]
            nums = [int(x) for x in NUMS_RE.search(p).group(1).split(",")]
            target = int(TARGET_RE.search(p).group(1))
            visible = THINK_RE.sub("", r["response"]).strip()
            if grade_countdown(visible, str(target), nums) != 1.0:
                continue
            if "<think>" not in r["response"] or "</think>" not in r["response"]:
                continue                      # truncated: no complete trace
            n_correct += 1
            if per_q.get(qid, 0) >= args.max_per_problem:
                continue
            per_q[qid] = per_q.get(qid, 0) + 1
            rows.append({"prompt": p, "response": r["response"].strip(),
                         "question_id": qid})

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    rc = out.response.str.len()
    print(f"{n_seen} rollouts seen, {n_correct} correct & complete")
    print(f"wrote {len(out)} rows over {len(per_q)} problems -> {args.out}")
    print(f"response chars: mean {rc.mean():.0f} median {rc.median():.0f}")
    print(f"sanity: all have <think>: {out.response.str.contains('<think>').all()}")


if __name__ == "__main__":
    main()
