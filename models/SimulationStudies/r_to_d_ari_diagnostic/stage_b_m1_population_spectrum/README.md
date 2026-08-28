# Stage B/M1: population spectrum and finite-sample detectability

## Purpose

Why can d_hat remain below K even when the true residual is observed?

## Design

Lyapunov-solved population covariance/correlation, IID-versus-VAR sampling, zero-start/burn-in checks, a T grid and 50 paths per world.

## File guide

| File or directory | Role |
|---|---|
| scripts/common_stage_b.py | Population/sample spectrum and branch generators. |
| scripts/run_stage_b.py | Runs the five spectrum branches. |
| scripts/summarize_stage_b.py | Computes detection frequencies and paired contrasts. |
| scripts/make_stage_b_figures.py | Regenerates four spectrum figures. |
| results/population_spectra.csv | World-level population eigenvalues. |
| results/sample_spectra.csv | Path-level finite-sample eigenvalues and d_hat. |

## Reproduction

Run commands from the repository root. See reproduce/FULL_REPRODUCTION_COMMANDS.md for the exact command sequence. Existing supplied results are not overwritten by the full-run commands.
