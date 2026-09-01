# Build the Countdown contrastive-OPSD training parquet from the merged pair
# pool (pilot/countdown_pairs_pool.parquet, 3,726 problems, question_id unique
# after shard-offset remap).
#
#   envs/train/.venv/bin/python rl/build_opsd_dataset.py \
#       --pool pilot/countdown_pairs_pool.parquet \
#       --out_train data/raw/countdown_opsd_train.parquet \
#       --out_heldout data/raw/countdown_opsd_heldout.parquet \
#       --tokenizer /scratch/gpfs/ARORA/skaur/models/Qwen3-4B
#
# Mirrors data_tools/build_contrastive_dataset.py's OUTPUT CONTRACT (the shape
# arm1_launch.sh + train.py::prepare_distil_dataset consume, so no train.py
# data-path changes are needed):
#   problem         -> student prompt (PROMPT_KEY=problem), the bare harvest
#                      prompt, unchanged
#   teacher_context -> fully rendered teacher prompt
#                      (GOLD_ANSWER_KEY=teacher_context with
#                      TEACHER_PROMPT_TEMPLATE='{gold_answer}' passthrough)
#   answer          -> gate gold (GATE_GOLD_ANSWER_KEY=answer); for Countdown
#                      this is the TARGET integer as a string -- the trainer's
#                      math_equal gate then tests VALUE equality of the boxed
#                      expression, not the each-number-exactly-once
#                      constraint (documented in rl/arms/REVIEW.md)
#   wrong_answer    -> modal wrong expression (GATE_WRONG_ANSWER_KEY;
#                      inert while GATE_REQUIRE_DIFF_ANSWER=False, as in Arm 1)
#
# TEMPLATE CHOICE (documented per the arm-prep task): the teacher context is
# rl/templates.py v4_pair_content(nums, target, a, b, episode=False).
#   - v4 (not v2): v4 is the training template of record since commit dd20677
#     -- the channel-explicit no-reference contract -- and the RL'd teacher
#     (grader_grpo_lora_v4tpl) was trained under v4 phrasing. The COUPLING
#     note in rl/templates.py requires the OPSD teacher context to move with
#     it.
#   - episode=False (V4_PAIR_PROBE tail, not V4_PAIR_EPISODE): the OPSD
#     teacher never sees a partial thinking prefix -- it scores the student's
#     rollout from generation start, with the rollout masquerading as the
#     teacher's own <think> channel from token 0. The episode variant's
#     "Continue your reasoning from where it leaves off" clause presumes an
#     existing mid-<think> prefix (the grader-RL episode setup) and would
#     dangle here; the probe tail ("Now solve the instance. Reason step by
#     step...") matches the actual scoring situation. Residual mismatch --
#     the RL'd teacher was TRAINED with the episode clause + prefixes,
#     including short ones -- is flagged in rl/arms/REVIEW.md.
#   - (a, b) order: x_plus first iff pos_first, preserving the harvest's
#     deterministic order-balance (qid parity), so teacher context order is
#     uncorrelated with correctness.
#
# Held-out split: ~10% of question_ids (one row per question_id in this pool),
# deterministic under --seed, written to a separate parquet that never enters
# training. Use it for post-run instruments (slot/slip margins, probes).
import argparse
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.templates import v4_pair_content  # noqa: E402

# The harvest prompt states the instance exactly once, in this fixed form
# (pilot/gen_countdown_balanced.py / gen_hard_countdown.py):
#   Numbers: [3, 45, 12, 17, 2]
#   Target: 24
NUMS_RE = re.compile(r"Numbers:\s*\[([0-9,\s]+)\]")
TARGET_RE = re.compile(r"Target:\s*(-?\d+)")


def parse_instance(problem_text):
    """(nums, target) from the harvest prompt; raises if not found exactly."""
    m_nums = NUMS_RE.findall(problem_text)
    m_tgt = TARGET_RE.findall(problem_text)
    # The example in the prompt boilerplate says "the numbers are [2, 3, 5]"
    # without the "Numbers:" prefix, so exactly one match each is expected.
    assert len(m_nums) == 1 and len(m_tgt) == 1, \
        f"instance parse failed ({len(m_nums)} nums / {len(m_tgt)} target matches)"
    nums = [int(x) for x in m_nums[0].split(",")]
    return nums, int(m_tgt[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_heldout", required=True)
    ap.add_argument("--heldout_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None,
                    help="Optional HF tokenizer path; if set, prints real token-length "
                         "stats for teacher_context (sets OPSD_MAX_PROMPT_LENGTH)")
    ap.add_argument("--print_row", type=int, default=0,
                    help="Print the fully rendered teacher_context of this row index")
    args = ap.parse_args()

    df = pd.read_parquet(args.pool)
    assert df.question_id.is_unique, "question_id not unique -- shard offsets missing?"
    print(f"{len(df)} pool rows, shards: {sorted(df.shard.unique().tolist())}")

    out = []
    for r in df.itertuples(index=False):
        nums, target = parse_instance(r.problem)
        assert str(target) == str(r.answer).strip(), \
            f"qid {r.question_id}: parsed target {target} != pool answer {r.answer!r}"
        a, b = (r.x_plus, r.x_minus) if r.pos_first else (r.x_minus, r.x_plus)
        out.append({
            "problem": r.problem,
            "teacher_context": v4_pair_content(nums, target, a, b, episode=False),
            "answer": str(r.answer),
            "wrong_answer": str(r.wrong_answer),
            "x_plus": r.x_plus,
            "x_minus": r.x_minus,
            "pos_first": bool(r.pos_first),
            "question_id": int(r.question_id),
            "n_correct": int(r.n_correct),
            "n_gens": int(r.n_gens),
            "shard": int(r.shard),
        })
    full = pd.DataFrame(out)

    qids = sorted(full.question_id.tolist())
    rng = random.Random(args.seed)
    n_held = round(len(qids) * args.heldout_frac)
    held = set(rng.sample(qids, n_held))
    train_df = full[~full.question_id.isin(held)].reset_index(drop=True)
    held_df = full[full.question_id.isin(held)].reset_index(drop=True)
    assert len(train_df) + len(held_df) == len(full)
    assert not (set(train_df.question_id) & set(held_df.question_id))

    for path, d in [(args.out_train, train_df), (args.out_heldout, held_df)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d.to_parquet(path, index=False)
        print(f"wrote {len(d)} rows -> {path} "
              f"(x+ first in {int(d.pos_first.sum())}, x- first in {int((~d.pos_first).sum())})")

    ctx_chars = full["teacher_context"].str.len()
    print(f"teacher_context chars: mean {ctx_chars.mean():.0f} "
          f"p99 {ctx_chars.quantile(.99):.0f} max {ctx_chars.max()}")
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tlens = [len(tok(t).input_ids) for t in full["teacher_context"]]
        s = pd.Series(tlens)
        print(f"teacher_context TOKENS: mean {s.mean():.0f} p99 {s.quantile(.99):.0f} "
              f"max {s.max()}  -> set OPSD_MAX_PROMPT_LENGTH above max + chat-template overhead")

    print("\n===== rendered teacher_context, row", args.print_row, "=====")
    print(full.iloc[args.print_row]["teacher_context"])
    print("===== end =====")
    print("\nstudent prompt (problem), same row:")
    print(full.iloc[args.print_row]["problem"])


if __name__ == "__main__":
    main()
