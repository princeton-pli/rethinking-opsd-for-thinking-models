# Summarize the graded adjudication probe into the 3x2 decision table.
#   python pilot/probe_report.py pilot/probe_adjudication.graded.jsonl
import collections
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
cells = collections.defaultdict(lambda: [0, 0, 0])  # ok, n, total_len
for r in rows:
    c = cells[r["cell"]]
    c[0] += bool(r["is_correct"])
    c[1] += 1
    c[2] += len(r["response"])

print("%-22s %8s %8s %10s" % ("cell", "acc", "n", "mean chars"))
for cell in ["none/nothink", "unlabeled/nothink", "neglabel/nothink",
             "none/think", "unlabeled/think", "neglabel/think"]:
    ok, n, tl = cells[cell]
    print("%-22s %8.3f %8d %10.0f" % (cell, ok / max(n, 1), n, tl / max(n, 1)))

# Stratify the key cells by answer length (short answers = weakest discriminator).
strat = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r["cell"] in ("unlabeled/think", "unlabeled/nothink"):
        bucket = "short(<=3ch)" if r["answer_len"] <= 3 else "long(>3ch)"
        s = strat[(r["cell"], bucket)]
        s[0] += bool(r["is_correct"])
        s[1] += 1
print()
for (cell, bucket), (ok, n) in sorted(strat.items()):
    print("%-22s %-14s acc %.3f  (n=%d)" % (cell, bucket, ok / max(n, 1), n))
