# usage: python eval/mcnemar.py precode sft mbppplus224   [any env with scipy, repo root]
# Exact paired McNemar test, pooled over the seeds that exist for both methods. Appends to results/stats.md.
import json, os, sys
from scipy.stats import binomtest

A, B, ds = sys.argv[1:4]
n01 = n10 = 0; seeds = []
for s in (42, 7, 1234):
    pa, pb = f"results/per_task/{A}_s{s}_{ds}.json", f"results/per_task/{B}_s{s}_{ds}.json"
    if not (os.path.exists(pa) and os.path.exists(pb)): continue
    a, b = json.load(open(pa)), json.load(open(pb)); seeds.append(s)
    for k in set(a) & set(b):
        n01 += a[k] and not b[k]; n10 += b[k] and not a[k]
p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
line = f"{A} vs {B} on {ds}, seeds {seeds}: {A} only {n01}, {B} only {n10}, exact McNemar p = {p:.4f}"
print(line); open("results/stats.md", "a").write(line + "\n")
