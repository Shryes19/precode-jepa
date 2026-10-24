#!/usr/bin/env bash
# usage: bash eval/watch.sh   [WS, env evalv, tmux eval, ~/nikhita/precode-jepa]; leave it running all night
# Every 5 min: evaluates each model whose DONE is newer than its scores (so a retrained run, e.g. LBD=0.5,
# is re-scored), then runs Attempts-to-green for it. A failed run (e.g. OOM) is retried on the next pass.
# precodecond_* is skipped: vLLM can't apply its soft prefix, so it is evaluated with HF (Gagan).
while true; do
  git pull -q || true
  git push -q 2>/dev/null || true                              # retry a push that failed last time
  for d in /data/precode/models/*/; do
    r=$(basename "$d")
    [[ $r == smoke* || $r == precodecond* ]] && continue
    [ -f "$d/DONE" ] || continue
    if [ ! results/per_task/${r}_mbppplus224.json -nt "$d/DONE" ]; then
      echo "$(date +%H:%M) eval $r"
      bash eval/run_eval.sh "$r" && git add results/ && git commit -qm "[results] $r" && git push -q
    fi
    if [ results/per_task/${r}_mbppplus224.json -nt "$d/DONE" ] && [ ! results/attempts/$r.json -nt "$d/DONE" ]; then
      echo "$(date +%H:%M) attempts $r"
      python eval/attempts.py "$r" && git add results/ && git commit -qm "[results] attempts $r" && git push -q
    fi
  done
  sleep 300
done
