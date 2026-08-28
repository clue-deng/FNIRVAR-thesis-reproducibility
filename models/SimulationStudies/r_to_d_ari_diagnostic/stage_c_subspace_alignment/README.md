# Stage C: three-point subspace mechanism

## Purpose

When r is under- or over-specified, do selected residual eigenvectors align with factor-loading directions or community directions?

## Design

Paired r_used={3,5,7} comparison on the same 20 worlds and 25 forecast origins; principal-angle, canonical-correlation and projection-energy summaries.

## File guide

| File or directory | Role |
|---|---|
| scripts/common_stage_c.py | Constructs factor/community target subspaces and alignment metrics. |
| scripts/run_stage_c.py | Runs cell-level subspace calculations. |
| scripts/summarize_stage_c.py | Builds world-level summaries and paired contrasts. |
| scripts/make_stage_c_figures.py | Regenerates Stage C figures. |
| results/ | Cell, eigenvector and matched-incremental-space outputs. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
