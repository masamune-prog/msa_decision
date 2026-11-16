#!/bin/bash
# Excision for the 9 receptors not covered by ex_out. Seed 1, both arms.
# GPU list via $GPUS (default "2"). Resumable: finished folds are skipped.
set -u
ROOT=${MSA_DECISION_ROOT:?set MSA_DECISION_ROOT to the experiment work directory}
AF3=$ROOT/alphafold3; OUT=$ROOT/ex2_out
GPUS=(${GPUS:-2}); NG=${#GPUS[@]}
mkdir -p "$OUT"
pending=()
for c in full none; do for f in "$ROOT"/ex2_in_$c/*.json; do
  n=$(basename "$f" .json)
  compgen -G "$OUT/$n/*_confidences.json" >/dev/null || pending+=("$f")
done; done
echo "pending: ${#pending[@]}"; (( ${#pending[@]} == 0 )) && exit 0
for i in $(seq 0 $((NG-1))); do rm -rf "$ROOT/ex2_shard${GPUS[$i]}"; mkdir -p "$ROOT/ex2_shard${GPUS[$i]}"; done
i=0; for f in "${pending[@]}"; do ln -s "$f" "$ROOT/ex2_shard${GPUS[$((i % NG))]}/$(basename "$f")"; i=$((i+1)); done
pids=()
for g in "${GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g "$AF3/.venv/bin/python" "$AF3/run_alphafold.py" --input_dir="$ROOT/ex2_shard$g" \
    --output_dir="$OUT" --model_dir="$ROOT" --norun_data_pipeline > "$ROOT/ex2_gpu$g.log" 2>&1 &
  pids+=($!)
done
echo "started $(date +%H:%M:%S) pids ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done
echo "finished $(date +%H:%M:%S); outputs: $(ls -d $OUT/*/ | wc -l)"
