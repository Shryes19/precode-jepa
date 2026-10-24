#!/usr/bin/env bash
# usage: bash eval/run_para.sh [run ...]   [WS, env evalv, tmux eval, ~/nikhita/precode-jepa]
# No args = every {sft,jepa,precode}_s{42,7,1234} that is DONE and already scored by run_eval.sh.
# Paraphrase drop = para60_orig - para60_para, per run, on the same 60 tasks.
set -euo pipefail
P=eval/paraphrase/mbppplus_paraphrased.jsonl
[ -f $P ] || { echo "$P missing: git fetch origin && git merge origin/main"; exit 1; }
res() { local b=${1%.jsonl}; if [ -f "${b}_eval_results.json" ]; then echo "${b}_eval_results.json"; else echo "${b}.eval_results.json"; fi; }
RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  for s in 42 7 1234; do for m in sft jepa precode; do
    if [ -f /data/precode/models/${m}_s$s/DONE ] && [ -f results/per_task/${m}_s${s}_mbppplus224.json ]; then RUNS+=(${m}_s$s); fi
  done; done
fi
for r in "${RUNS[@]}"; do
  O=/data/precode/samples/$r
  echo "== $r"
  rm -f $O/mbpp_para_eval_results.json $O/mbpp_para.eval_results.json
  python eval/generate.py --model /data/precode/models/$r --dataset mbpp --prompts $P --out $O/mbpp_para.jsonl
  evalplus.evaluate --dataset mbpp --samples $O/mbpp_para.jsonl
  python eval/score.py "$(res $O/mbpp_para.jsonl)" $r para60_para --ids $P
  python eval/score.py "$(res $O/mbpp.jsonl)" $r para60_orig --ids $P   # same 60 tasks, original wording
done
