# Stage C full-grid extension

## Purpose

Do the Stage C mechanisms and candidate feedback observables persist over r_used=1,...,9?

## Design

Extends the same paired worlds and origins to the full r grid; applies pre-specified paired contrasts and Holm correction.

## File guide

| File or directory | Role |
|---|---|
| scripts/common_full_grid.py | Generalised missing/extra subspaces and observable definitions. |
| scripts/run_full_grid_stage_c.py | Runs the new r cells. |
| scripts/summarize_full_grid_stage_c.py | Combines three-point and new cells and computes contrasts. |
| scripts/make_full_grid_figures.py | Regenerates six mechanism/observable figures. |
| results/ | Combined cell-level data, contrasts and feasibility summaries. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
