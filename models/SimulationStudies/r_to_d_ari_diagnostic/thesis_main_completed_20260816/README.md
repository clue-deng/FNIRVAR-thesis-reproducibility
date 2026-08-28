# Main propagation experiment

## Purpose

How does imposing the wrong factor dimension r affect the MP-selected embedding dimension, community recovery and forecast error?

## Design

Twenty structural worlds; r_used=1,...,9; paired fixed-K and K=d_hat branches; 499 rolling forecast origins.

## File guide

| File or directory | Role |
|---|---|
| scripts/common.py | Canonical DGP, seed tree and shared helpers. |
| scripts/run_rolling_grid.py | Runs one structural replication across r and both branches. |
| scripts/run_formal_parallel.sh | Portable launcher for replications 1 to 19. |
| scripts/run_reconstructed_reference.py | Reconstructs the released MSPE reference cell. |
| scripts/summarize_run.py | Aggregates origin rows to world-level and cross-world tables. |
| scripts/make_formal_figures.py | Regenerates the d_hat, ARI and MSPE figures. |
| configs/ | Frozen DGP and run settings. |
| runs/ | The 20 origin CSVs plus minimal manifests needed by downstream stages. |
| formal_results_20260816/ | Publication-facing summary tables and figures. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
