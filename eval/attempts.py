# usage: python eval/attempts.py <run>   [WS, env evalv, repo root] -- contract C11
# Attempts-to-green: 5 independent samples at temperature 0.8; first passing attempt 1-5, or 6 if none.
import json, os, subprocess, sys

run = sys.argv[1]; model = f"/data/precode/models/{run}"
assert os.path.exists(f"{model}/DONE"), f"{run} not DONE"
out = f"/data/precode/samples/{run}"; os.makedirs(out, exist_ok=True)
samples = f"{out}/mbpp_n5.jsonl"
# EvalPlus writes <name>_eval_results.json (old) or <name>.eval_results.json (>=0.3) and reuses a stale one
resfs = [f"{out}/mbpp_n5_eval_results.json", f"{out}/mbpp_n5.eval_results.json"]
subprocess.run(["python", "eval/generate.py", "--model", model, "--dataset", "mbpp",
                "--n", "5", "--temperature", "0.8", "--out", samples], check=True)
for f in resfs:
    if os.path.exists(f): os.remove(f)
subprocess.run(["evalplus.evaluate", "--dataset", "mbpp", "--samples", samples], check=True)
res = json.load(open(next(f for f in resfs if os.path.exists(f))))["eval"]

ids = json.load(open("eval/mbppplus_test_ids.json"))
att = {}
for t in ids:
    ok = [r["base_status"] == "pass" and r["plus_status"] == "pass" for r in res[t]]
    assert len(ok) == 5, (t, len(ok))
    att[t] = next((i + 1 for i, o in enumerate(ok) if o), 6)  # 6 = never solved in 5
os.makedirs("results/attempts", exist_ok=True)
json.dump(att, open(f"results/attempts/{run}.json", "w"))
v = list(att.values())
print(run, "mean attempts", round(sum(v) / len(v), 3), "| solved by 1/3/5:",
      [round(sum(x <= k for x in v) / len(v), 3) for k in (1, 3, 5)])
