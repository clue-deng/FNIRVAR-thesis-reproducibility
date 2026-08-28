#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/models/SimulationStudies/r_to_d_ari_diagnostic"
RUNS="$ROOT/reproduced_runs"
RESULTS="$ROOT/reproduced_results"
mkdir -p "$RUNS" "$RESULTS"
export MPLCONFIGDIR="$RESULTS/.matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"

export FNIRVAR_REPO_ROOT="$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if ! env | grep -q '^CONFIRM_FULL_REPRODUCTION=yes$'; then
  echo "This launches the full simulation suite and may take several hours."
  echo "Re-run with CONFIRM_FULL_REPRODUCTION=yes to continue."
  exit 2
fi

python "$D/preliminary_suite/preliminary_diagnostics.py"   --repo-root "$ROOT" --outdir "$RESULTS/preliminary_suite"

for rep in $(seq 0 19); do
  tag=$(printf "%02d" "$rep")
  python "$D/thesis_main_completed_20260816/scripts/run_rolling_grid.py"     --run-id reproduced_main --outdir "$RUNS/main_rep$tag"     --replication "$rep" --n-total-replications 20     --r-grid 1,2,3,4,5,6,7,8,9 --origin-limit 499     --config "$D/thesis_main_completed_20260816/configs/canonical_thesis.json"
done
python "$D/thesis_main_completed_20260816/scripts/summarize_run.py"   --inputs "$RUNS"/main_rep*/origin_results.csv   --outdir "$RESULTS/thesis_main"

python "$D/stage_c_subspace_alignment/scripts/run_stage_c.py"   --run-id reproduced_stage_c --outdir "$RUNS/stage_c"   --replications 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19   --r-grid 3,5,7   --origins 0,20,41,62,83,103,124,145,166,186,207,228,249,269,290,311,332,352,373,394,415,435,456,477,498   --config "$D/stage_c_subspace_alignment/configs/stage_c_config.json"
python "$D/stage_c_subspace_alignment/scripts/summarize_stage_c.py"   --run-dir "$RUNS/stage_c" --out-dir "$RESULTS/stage_c"

python "$D/stage_c_full_grid_extension/scripts/run_full_grid_stage_c.py"   --mode formal --run-id reproduced_stage_c_full   --replications 0-19 --origins 25 --out-dir "$RUNS/stage_c_full"
python "$D/stage_c_full_grid_extension/scripts/summarize_full_grid_stage_c.py"   --run-dir "$RUNS/stage_c_full" --out-dir "$RESULTS/stage_c_full"

python "$D/stage_b_m1_population_spectrum/scripts/run_stage_b.py"   --mode formal --run-id reproduced_stage_b --worlds 0-19   --out-dir "$RUNS/stage_b"
python "$D/stage_b_m1_population_spectrum/scripts/summarize_stage_b.py"   --run-dir "$RUNS/stage_b" --out-dir "$RESULTS/stage_b"

python "$D/stage_d_iterative_loop/scripts/run_stage_d.py"   --run-id reproduced_stage_d --outdir "$RUNS/stage_d"   --config "$D/stage_d_iterative_loop/configs/stage_d_config.json"
python "$D/stage_d_iterative_loop/scripts/run_initialisation.py"   --run-id reproduced_stage_d_init --outdir "$RUNS/stage_d_init"   --config "$D/stage_d_iterative_loop/configs/stage_d_config.json"
python "$D/stage_d_iterative_loop/scripts/summarize_stage_d.py"   --run "$RUNS/stage_d" --init-run "$RUNS/stage_d_init"   --outdir "$RESULTS/stage_d"

(
  cd "$D/stage_e_lite_heldout_r_selection"
  export FNIRVAR_STAGE_E_OUTDIR="$RESULTS/stage_e_lite"
  python scripts/run_stage_e_lite.py
  python scripts/summarise_stage_e_lite.py
  python scripts/make_stage_e_lite_figures.py
)

python "$D/stage_f_dgp_boundary/scripts/run_stage_f.py"   --run-id reproduced_stage_f --outdir "$RUNS/stage_f"   --config "$D/stage_f_dgp_boundary/configs/stage_f_config.json"
python "$D/stage_f_dgp_boundary/scripts/run_initialisation_stage_f.py"   --run-id reproduced_stage_f_init --outdir "$RUNS/stage_f_init"   --config "$D/stage_f_dgp_boundary/configs/stage_f_config.json"
python "$D/stage_f_dgp_boundary/scripts/summarize_stage_f.py"   --run "$RUNS/stage_f" --init-run "$RUNS/stage_f_init"   --outdir "$RESULTS/stage_f" --figure-dir "$RESULTS/stage_f_figures"

echo "Full reproduction completed."
