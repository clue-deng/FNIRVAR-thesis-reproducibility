#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3)"
RUNNER="$ROOT/scripts/run_rolling_grid.py"
CONFIG="$ROOT/configs/canonical_thesis.json"
LOGDIR="$ROOT/runs/formal_launcher_logs_20260816"
mkdir -p "$LOGDIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_replication() {
  local rep="$1"
  local tag
  tag=$(printf "%02d" "$rep")
  local out="$ROOT/runs/formal_preferred_rep${tag}_20260816"
  "$PY" "$RUNNER" \
    --run-id "formal_preferred_20260816" \
    --outdir "$out" \
    --replication "$rep" \
    --n-total-replications 20 \
    --r-grid 1,2,3,4,5,6,7,8,9 \
    --origin-limit 499 \
    --progress-every 499 \
    --config "$CONFIG" \
    >"$LOGDIR/rep${tag}.log" 2>&1
}
export -f run_replication
export ROOT PY RUNNER CONFIG LOGDIR

seq 1 19 | xargs -n 1 -P 4 bash -c 'run_replication "$1"' _
