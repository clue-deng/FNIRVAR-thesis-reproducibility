# Stage F: DGP signal-boundary experiment

## Purpose

Under which community-separation and factor-loading regimes does iterative feedback remain feasible?

## Design

Five nominal-density-matched community-separation anchors and two loading-scale anchors, using paired structural worlds. Achieved eigengap is recorded as a mechanism variable, not independently manipulated.

## File guide

| File or directory | Role |
|---|---|
| scripts/common_stage_f.py | Seven DGP anchors and population diagnostics. |
| scripts/reused_stage_d.py | Self-contained Stage D transition machinery reused by Stage F. |
| scripts/run_stage_f.py | Runs transition cells for all anchors. |
| scripts/run_initialisation_stage_f.py | Composes initialisers with each anchor transition map. |
| scripts/summarize_stage_f.py | Computes registered contrasts and the primary figure. |
| scripts/make_supplementary_figures.py | Generates ARI/feasibility and eigengap supplementary figures. |
| results/ | World metrics, anchor summaries, contrasts and instability summaries. |
| figures/ | Primary and supplementary figures. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
