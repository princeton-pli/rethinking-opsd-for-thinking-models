# Stage-1 screen: drop problems already settled as too easy at k=3.
#
#   python pilot/screen_stage1.py <stage1.jsonl> <band.parquet> <survivors.parquet>
#
# Keeps every problem that is NOT all-correct at stage 1 (all-correct problems
# have p(success) high enough that x- is unlikely and the gate would starve).
# All-wrong problems are kept: at k=8 some yield a rare x+ and those are the
# most informative training problems. Stage-1 samples are reused later by the
# pool builder, so stage 2 only tops survivors up from k=3 to k=8.
import json
import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

stage1_jsonl, band_parquet, out_parquet = sys.argv[1], sys.argv[2], sys.argv[3]

by_q = collections.defaultdict(list)
with open(stage1_jsonl) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_q[r["question_id"]].append(bool(r.get("is_correct")))

band = pd.read_parquet(band_parquet)
assert len(by_q) == len(band), f"stage1 covers {len(by_q)} questions, band has {len(band)}"

survivor_ids = sorted(qid for qid, oks in by_q.items() if not all(oks))
print(f"{len(band)} problems; all-correct at k=3: {len(band) - len(survivor_ids)}; "
      f"survivors: {len(survivor_ids)} ({len(survivor_ids) / len(band):.2f})")

# question_id is the 0-based row order of the band parquet (run_generate global_id).
survivors = band.iloc[survivor_ids].reset_index(drop=True)
survivors["orig_question_id"] = survivor_ids
survivors.to_parquet(out_parquet, index=False)
print(f"wrote {len(survivors)} rows -> {out_parquet}")
