#!/bin/bash
# Generate parent MSAs for every target sequence, 3 at a time.
#
# Skips pMHC targets: their "targets" include 9-10 residue peptides where
# alignment depth is not a meaningful quantity, and they carry 3 of the 394
# confirmed binders between them.
#
# Resumable: a sequence whose *_data.json already exists is skipped.

set -u
ROOT=${MSA_DECISION_ROOT:?set MSA_DECISION_ROOT to the experiment work directory}
AF3=$ROOT/alphafold3
DB=$AF3/msa
OUT=$ROOT/msa_out
LOGS=$OUT/logs
PARALLEL=3
NCPU=4          # 3 targets x 4 databases x 4 cpu = 48 threads = all cores

mkdir -p "$OUT" "$LOGS"

mapfile -t KEYS < <(awk -F'\t' 'NR>1 && $2 !~ /^pMHC/ {print $1"\t"$4}' "$ROOT/targets/manifest.tsv" \
                    | sort -t$'\t' -k2,2nr | cut -f1)

echo "queued: ${#KEYS[@]} target sequences (pMHC excluded), longest first"
echo "parallelism: $PARALLEL targets x $NCPU cpu per jackhmmer"
echo

run_one() {
  local key=$1
  if [[ -f "$OUT/$key/${key}_data.json" ]]; then
    echo "[skip] $key (already done)"
    return 0
  fi
  local t0=$(date +%s)
  echo "[start] $key  $(date +%H:%M:%S)"
  "$AF3/.venv/bin/python" "$AF3/run_alphafold.py" \
      --json_path="$ROOT/targets/${key}.json" \
      --output_dir="$OUT" \
      --db_dir="$DB" \
      --jackhmmer_n_cpu=$NCPU \
      --nhmmer_n_cpu=$NCPU \
      --norun_inference > "$LOGS/${key}.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  if [[ $rc -eq 0 && -f "$OUT/$key/${key}_data.json" ]]; then
    echo "[done]  $key  $(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
  else
    echo "[FAIL]  $key  rc=$rc  see $LOGS/${key}.log"
  fi
}

for key in "${KEYS[@]}"; do
  while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n; done
  run_one "$key" &
done
wait
echo
echo "=== ALL MSA JOBS FINISHED $(date +%H:%M:%S) ==="
ls -1 "$OUT"/*/*_data.json 2>/dev/null | wc -l | xargs echo "data.json files produced:"
