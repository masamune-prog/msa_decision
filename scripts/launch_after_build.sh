#!/bin/bash
# Wait for the input builders to finish, then start one opportunistic scheduler
# per GPU. The schedulers only fold while a GPU is otherwise idle, so this is
# safe to leave running on a shared machine.
set -u
: "${MSA_DECISION_ROOT:?set MSA_DECISION_ROOT to the experiment work directory}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$MSA_DECISION_ROOT"

while pgrep -f "build_ladder.py|build_masked.py" >/dev/null; do sleep 30; done
echo "builds done $(date): mk_in=$(ls mk_in | wc -l) e3b_in=$(ls e3b_in | wc -l)"

rm -rf sched/claims; mkdir -p sched/claims
for g in 0 1 2; do
  nohup python3 -I "$HERE/sched_gpu.py" "$g" > "sched/gpu$g.log" 2>&1 &
done
echo "schedulers launched; they wait for idle GPUs"
