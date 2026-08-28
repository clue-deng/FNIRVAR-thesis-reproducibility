# Preliminary pipeline sanity check

## Purpose

Does the released data-generation to factor-adjustment to MP-dimension to clustering pipeline run coherently before a formal Monte Carlo study?

## Design

A small multi-seed and T-sweep diagnostic comparing true and estimated residuals. It is a smoke test, not a final inferential result.

## File guide

| File or directory | Role |
|---|---|
| preliminary_diagnostics.py | Runs the compact sanity experiment. |
| preliminary_results.csv | Run-level outputs. |
| preliminary_summary.csv | Grouped pilot summaries. |
| mechanism_diagnostics/common_inputs/M1/ | Five frozen Phi/Omega inputs required by the Stage B replay branch. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
