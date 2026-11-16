#!/bin/bash
# E2: full target MSA vs single-sequence, across all single-chain designs.
#
# Shards the pending folds round-robin over 3 GPUs. Resumable: an input whose
# output already contains a top-level *_confidences.json is skipped, so
# re-running after an interruption only does what is left.
#
# Shards hold SYMLINKS, not copies -- the `full` inputs are 24 GB.

set -u
ROOT=${MSA_DECISION_ROOT:?set MSA_DECISION_ROOT to the experiment work directory}
AF3=$ROOT/alphafold3
OUT=$ROOT/e2_out
NGPU=3

mkdir -p "$OUT"

# ---- work out what is still pending -------------------------------------
pending=()
for cond in full none; do
  for f in "$ROOT"/e2_in_$cond/*.json; do
    name=$(basename "$f" .json)                 # <binder_id>__<cond>
    if ! compgen -G "$OUT/$name/*_confidences.json" > /dev/null; then
      pending+=("$f")
    fi
  done
done

total=${#pending[@]}
echo "pending folds: $total"
if (( total == 0 )); then echo "nothing to do"; exit 0; fi

# ---- shard round-robin ---------------------------------------------------
for g in $(seq 0 $((NGPU-1))); do
  rm -rf "$ROOT/e2_shard$g"; mkdir -p "$ROOT/e2_shard$g"
done
i=0
for f in "${pending[@]}"; do
  ln -s "$f" "$ROOT/e2_shard$((i % NGPU))/$(basename "$f")"
  i=$((i+1))
done
for g in $(seq 0 $((NGPU-1))); do
  echo "  shard $g: $(ls "$ROOT/e2_shard$g" | wc -l) folds -> GPU $g"
done

# ---- launch one AF3 process per GPU -------------------------------------
pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g "$AF3/.venv/bin/python" "$AF3/run_alphafold.py" \
      --input_dir="$ROOT/e2_shard$g" \
      --output_dir="$OUT" \
      --model_dir="$ROOT" \
      --norun_data_pipeline > "$ROOT/e2_gpu$g.log" 2>&1 &
  pids+=($!)
  echo "  GPU $g -> pid ${pids[-1]}"
done

echo "started $(date +%H:%M:%S); waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

done_n=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)
echo "=== E2 PASS FINISHED $(date +%H:%M:%S) ==="
echo "outputs present: $done_n"
echo "expected total : $(( $(ls "$ROOT"/e2_in_full/*.json | wc -l) * 2 ))"
exit $fail
