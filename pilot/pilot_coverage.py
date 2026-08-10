# Coverage analysis + x+/x- pool construction from the harvest JSONL.
# Run from repo root (needs evaluation/ on path):
#   python pilot/pilot_coverage.py pilot/data/raw/ot15k_pilot512.parquet-temp_1.0-top_p_1.0.jsonl
#
# Reports, per the screening rule agreed for the contrastive-OPSD method:
#   usable(q) = (>=1 correct completed response) AND (>=2 distinct wrong answers)
# and writes pilot/pilot_pool.jsonl with per-question x+ / x- selections:
#   x+ = shortest correct visible solution (think-stripped)
#   x- = shortest visible solution among those ending in the MODAL wrong answer
import json
import re
import sys
import collections

from evaluation.utils import extract_answer_math, strip_string

THINK_RE = re.compile(r"<think>[\s\S]*?</think>")

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
print(f"{len(rows)} generations loaded from {path}")

by_q = collections.defaultdict(list)
for r in rows:
    by_q[r["question_id"]].append(r)

n_correct_hist = collections.Counter()
frac_stats = collections.Counter()
gen_stats = collections.Counter()
pool = []

for qid, gens in sorted(by_q.items()):
    correct_pool = []          # (visible_len, visible_text)
    wrong_by_answer = collections.defaultdict(list)  # norm_answer -> [(len, text)]
    for r in gens:
        resp = r["response"]
        gen_stats["total"] += 1
        # A usable demo needs a closed think block and non-empty visible text.
        visible = THINK_RE.sub("", resp).strip()
        completed = "</think>" in resp and len(visible) > 0
        pred = extract_answer_math(resp)
        if pred is None or pred == "":
            gen_stats["no_answer"] += 1
            continue
        if r.get("is_correct"):
            gen_stats["correct"] += 1
            if completed:
                correct_pool.append((len(visible), visible))
        else:
            gen_stats["wrong"] += 1
            norm = strip_string(str(pred))
            if completed:
                wrong_by_answer[norm].append((len(visible), visible))

    n_correct = sum(1 for r in gens if r.get("is_correct"))
    n_correct_hist[n_correct] += 1
    distinct_wrong = len(wrong_by_answer)

    has_pos = len(correct_pool) >= 1
    has_two_wrong = distinct_wrong >= 2
    if has_pos:
        frac_stats["ge1_correct"] += 1
    if has_two_wrong:
        frac_stats["ge2_distinct_wrong"] += 1
    if has_pos and has_two_wrong:
        frac_stats["usable"] += 1

        x_plus = min(correct_pool)[1]
        modal_answer, modal_traces = max(
            wrong_by_answer.items(), key=lambda kv: (len(kv[1]), -min(t[0] for t in kv[1]))
        )
        x_minus = min(modal_traces)[1]
        raw = gens[0]
        pool.append({
            "question_id": qid,
            "problem": raw.get("problem", ""),
            "gold_answer": raw.get("gold_answer", ""),
            "n_correct_of_8": n_correct,
            "wrong_answer_counts": {k: len(v) for k, v in wrong_by_answer.items()},
            "x_plus": x_plus,
            "x_minus": x_minus,
            "x_minus_answer": modal_answer,
        })

nq = len(by_q)
print(f"\nquestions: {nq}")
print(f"solve rate (mean correct/8): {sum(k * v for k, v in n_correct_hist.items()) / (8 * nq):.3f}")
print(f"n_correct histogram: {dict(sorted(n_correct_hist.items()))}")
print(f"generations: {dict(gen_stats)}")
for k in ("ge1_correct", "ge2_distinct_wrong", "usable"):
    print(f"{k}: {frac_stats[k]}/{nq} = {frac_stats[k] / nq:.3f}")

out = "pilot/pilot_pool.jsonl"
with open(out, "w") as f:
    for entry in pool:
        f.write(json.dumps(entry) + "\n")
print(f"\nwrote {len(pool)} usable questions -> {out}")

lens_p = [len(e["x_plus"]) for e in pool]
lens_n = [len(e["x_minus"]) for e in pool]
if pool:
    print(f"x+ visible chars: mean {sum(lens_p) / len(pool):.0f}, max {max(lens_p)}")
    print(f"x- visible chars: mean {sum(lens_n) / len(pool):.0f}, max {max(lens_n)}")
