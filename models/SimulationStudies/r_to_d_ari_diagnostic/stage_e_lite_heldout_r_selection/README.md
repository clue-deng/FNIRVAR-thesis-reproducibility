# Stage E-lite: held-out MSPE selection

## Purpose

Within the released pipeline, does held-out predictive performance select an r that also preserves structural recovery?

## Design

Post-hoc reanalysis of the 20 frozen origin datasets: first 249 origins tune r, last 250 evaluate MSPE and ARI.

## File guide

| File or directory | Role |
|---|---|
| scripts/sources.py | Exact 20-world source manifest, including the special replication-0 path. |
| scripts/run_stage_e_lite.py | Creates tuning/evaluation metrics and chooses r. |
| scripts/summarise_stage_e_lite.py | Computes selection distributions, regret and near-tie summaries. |
| scripts/tdist.py | Small t-distribution helper used by the summary code. |
| scripts/make_stage_e_lite_figures.py | Regenerates four Stage E-lite figures. |
| results/ | Selection, regret, curve and near-tie outputs. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. The full-suite launcher sets `FNIRVAR_STAGE_E_OUTDIR`, so the supplied frozen results are not overwritten. Without that variable, the scripts retain their original behaviour and write under this stage directory.
