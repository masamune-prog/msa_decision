#!/bin/bash
# E2 seed replication: seeds 2 and 3, both arms, over GPUs 0,1,2.
# Fold names now carry the seed (<bid>__<cond>__s<seed>), so output dirs are
# identifiable and this is genuinely resumable.
set -u
ROOT=${MSA_DECISION_ROOT:?set MSA_DECISION_ROOT to the experiment work directory}
AF3=$ROOT/alphafold3
OUT=$ROOT/e2_seeds_out
NGPU=3
mkdir -p "$OUT"

pending=()
for s in 2 3; do
  for c in full none; do
    for f in "$ROOT/e2s${s}_in_${c}"/*.json; do
      n="$(basename "$f" .json)__s${s}"
      compgen -G "$OUT/$n/*_confidences.json" >/dev/null || pending+=("$f")
    done
  done
done
echo "pending: ${#pending[@]} of 14128"
(( ${#pending[@]} == 0 )) && { echo "all done"; exit 0; }

for g in $(seq 0 $((NGPU-1))); do rm -rf "$ROOT/e2s_shard$g"; mkdir -p "$ROOT/e2s_shard$g"; done
i=0
for f in "${pending[@]}"; do
  ln -s "$f" "$ROOT/e2s_shard$((i % NGPU))/$(basename "$f")"
  i=$((i+1))
done
for g in $(seq 0 $((NGPU-1))); do echo "  shard $g -> GPU $g: $(ls "$ROOT/e2s_shard$g" | wc -l)"; done

pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g "$AF3/.venv/bin/python" "$AF3/run_alphafold.py" \
    --input_dir="$ROOT/e2s_shard$g" --output_dir="$OUT" \
    --model_dir="$ROOT" --norun_data_pipeline > "$ROOT/e2s_gpu$g.log" 2>&1 &
  pids+=($!)
done
echo "${pids[@]}" > "$ROOT/e2s_pids.txt"
echo "launched on GPUs 0,1,2: ${pids[*]}  $(date +%H:%M:%S)"
