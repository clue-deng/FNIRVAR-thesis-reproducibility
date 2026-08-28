#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/models/SimulationStudies/r_to_d_ari_diagnostic"
export MPLCONFIGDIR="$ROOT/reproduced_results/.matplotlib-cache"
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$ROOT/reproduced_results/.cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python "$D/thesis_main_completed_20260816/scripts/make_formal_figures.py" --summary "$D/thesis_main_completed_20260816/formal_results_20260816/across_replication_summary.csv" --outdir "$D/thesis_main_completed_20260816/formal_results_20260816/figures"
(cd "$D/stage_c_subspace_alignment" && python scripts/make_stage_c_figures.py)
(cd "$D/stage_c_full_grid_extension" && python scripts/make_full_grid_figures.py)
(cd "$D/stage_b_m1_population_spectrum" && python scripts/make_stage_b_figures.py)
python "$D/stage_d_iterative_loop/scripts/make_stage_d_figures.py" --results "$D/stage_d_iterative_loop/results" --outdir "$D/stage_d_iterative_loop/results/figures"
(cd "$D/stage_e_lite_heldout_r_selection" && python scripts/make_stage_e_lite_figures.py)
python "$D/stage_f_dgp_boundary/scripts/make_supplementary_figures.py" --summary "$D/stage_f_dgp_boundary/results/formal_anchor_summary.csv" --figure-dir "$D/stage_f_dgp_boundary/figures"
echo "Figures regenerated."
