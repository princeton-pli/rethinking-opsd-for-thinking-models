import json
import random

def load(tag):
    d = {}
    for line in open(f"rl/solve_{tag}.jsonl"):
        r = json.loads(line)
        d[r["question_id"]] = r["n_correct"] / r["k"]
    return d

base = load("base")
arms = {t: load(t) for t in ["armA", "armB", "armC", "armD_star"]}
qids = sorted(base)
rng = random.Random(0)

print(f"{'arm':10s} {'avg@8':>7s} {'delta':>8s} {'95% CI (paired bootstrap)':>28s}")
print(f"{'base':10s} {sum(base.values())/len(qids):7.4f}")
for tag, arm in arms.items():
    diffs = [arm[q] - base[q] for q in qids]
    mean = sum(diffs) / len(diffs)
    boots = []
    for _ in range(4000):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo, hi = boots[100], boots[3899]
    print(f"{tag:10s} {sum(arm.values())/len(qids):7.4f} {mean:+8.4f} "
          f"{'[' + f'{lo:+.4f}, {hi:+.4f}' + ']':>28s}")

print()
for a, b in [("armA", "armB"), ("armA", "armC"), ("armC", "armB")]:
    diffs = [arms[a][q] - arms[b][q] for q in qids]
    mean = sum(diffs) / len(diffs)
    boots = []
    for _ in range(4000):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boots.append(sum(s) / len(s))
    boots.sort()
    print(f"{a} - {b}: {mean:+.4f}  95% CI [{boots[100]:+.4f}, {boots[3899]:+.4f}]")
