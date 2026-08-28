# Stage D: iterative feedback map

## Purpose

Does the proposed factor/network feedback rule have the true r as a fixed point, and from which starting values does it converge?

## Design

Three operational variants, released and zero-capable Bai-Ng branches, fixed-K and K=d_hat branches, strong/weak SBM worlds and r0=1,...,9.

## File guide

| File or directory | Role |
|---|---|
| scripts/common_stage_d.py | Feedback-map definitions and canonical DGP. |
| scripts/run_stage_d.py | Builds transition maps and trajectories. |
| scripts/run_initialisation.py | Evaluates IC_p2, zero-fixed, ER and GR initialisers. |
| scripts/summarize_stage_d.py | Summarises fixed points, basins and end-to-end success. |
| scripts/make_stage_d_figures.py | Regenerates transition and basin figures. |
| results/ | Transition maps, trajectories, initialisation and basin summaries. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
