# Formal imposed-`r` experiment: results and interpretation

## Executive result

The complete preferred design ran successfully: 20 structural replications,
`r_used=1,...,9`, two downstream branches, and 499 forecast origins per cell
(`179,640` branch-level origin rows, representing `89,820` unique residual
`r`-origin cells). All 24 full-run validation checks passed, all package MP
counts agreed with independent symmetric-eigenvalue counts, all 20 DGPs were
stationary, and no origin failed.

The controlled results support the claim that deliberately misspecifying the
factor count changes the MP-selected embedding, community recovery, and
forecasting performance. They also show that `d_hat` is not a reliable stand-alone
proxy for community recovery: at `r_used=3`, `d_hat` is numerically equal to the
target four in every run, yet ARI remains low.

## Design frozen before the formal run

- DGP: `N=100`, evaluation `T=1500`, `r_true=5`, `K_true=4`, strong SBM
  (`p_in=0.9`, `p_out=0.1`), 499 rolling origins with lookback 1000.
- Primary: MP-estimated `d_hat`, GMM fixed at `K_true=4`.
- Robustness: same residual and same recorded `d_hat`, GMM uses `K=d_hat`.
- Monte Carlo uncertainty: 20 structural replications; origins are not treated as
  independent replications.
- Released correlation-based MP implementation is primary; package source was
  not modified.

## Main means

All intervals and standard deviations are in `main_results_table.csv` and
`across_replication_summary.csv`.

| `r_used` | mean `d_hat` | Pr(`d_hat=4`) | ARI, fixed K | ARI, K=d_hat | MSPE, fixed K | MSPE, K=d_hat |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.474 | 0.054 | 0.011 | 0.008 | 2.146 | 2.164 |
| 2 | 4.975 | 0.025 | 0.027 | 0.016 | 2.070 | 2.077 |
| 3 | 4.000 | 1.000 | 0.072 | 0.072 | 2.005 | 2.005 |
| 4 | 3.000 | 0.000 | 0.237 | 0.212 | 1.926 | 1.930 |
| **5** | **2.051** | **0.000** | **0.640** | **0.332** | **1.901** | **1.926** |
| 6 | 1.735 | 0.000 | 0.293 | 0.177 | 1.922 | 1.952 |
| 7 | 2.284 | 0.007 | 0.025 | 0.018 | 1.905 | 1.922 |
| 8 | 2.332 | 0.007 | 0.011 | 0.009 | 1.906 | 1.924 |
| 9 | 2.392 | 0.029 | 0.007 | 0.006 | 1.907 | 1.924 |

## Finding 1: the `r_used -> d_hat` map is strong but non-monotone

Under-specification raises the selected dimension: mean `d_hat` is 5.474 at
`r=1`, 4.975 at `r=2`, and exactly 4 at `r=3`. It then falls to 2.051 at the true
`r=5`, reaches 1.735 at `r=6`, and rises modestly above two for `r=7,...,9`.
Therefore this experiment does not support a simple monotone update rule from
`d_hat` back to `r`.

The most important cell is `r_used=3`: the selected dimension equals the pinned
population target `K_true=4` with probability one, but mean ARI is only 0.072.
Thus `|d_hat-K_true|=0` can coexist with poor community recovery. This establishes
ambiguity in the count, not its mechanism; a subspace diagnostic is required
before calling the four directions factor contamination.

## Finding 2: community recovery is sharply maximised near the correct factor count

In the primary fixed-`K` branch, mean ARI is 0.640 at `r=5` (95% CI
`[0.617, 0.663]`). It falls to 0.237 at `r=4`, 0.293 at `r=6`, and is near zero
for `r<=2` and `r>=7`. Paired contrasts relative to `r=5` are negative for every
other grid value; for example, `r=3` differs by -0.568 (95% CI
`[-0.600, -0.536]`) and `r=7` by -0.615 (`[-0.642, -0.588]`).

At `r=5`, allowing `K=d_hat` rather than holding `K=4` lowers mean ARI by 0.308
(paired 95% CI `[-0.339, -0.277]`). This quantifies the additional downstream
cost of propagating the selected dimension into the cluster count in this DGP.

## Finding 3: forecast performance and community recovery are related but not identical

The primary fixed-`K` MSPE is smallest at the true `r=5` (mean 1.901; 95% CI
`[1.889, 1.914]`). Under-specification is clearly costly: relative to `r=5`, the
paired MSPE difference is +0.245 at `r=1`, +0.169 at `r=2`, +0.104 at `r=3`, and
+0.025 at `r=4`, with all corresponding 95% intervals above zero.

Over-specification is less clear in forecasting even though ARI collapses. At
`r=7`, primary MSPE differs from `r=5` by only +0.0034 with a 95% interval
`[-0.0004, 0.0073]`; at `r=8` and `r=9` the increases are small but positive.
Hence good forecasting does not imply good community recovery.

At `r=5`, the `K=d_hat` robustness branch raises MSPE by 0.0249 relative to fixed
`K` (paired 95% CI `[0.0219, 0.0279]`).

## Validation and implementation findings

- Formal checks: 24/24 passed (`FORMAL_VALIDATION.json`).
- Qualification checks: 14/14 passed; deterministic-subset hashes matched.
- Package versus independent MP count: all 89,820 unique residual cells agree
  (and therefore all 179,640 branch-level rows agree).
- Failures: 0/179,640; complete and conditional MSPE therefore coincide here.
- Realised VAR spectral radius: min 0.900000, mean 0.912934, SD 0.017614,
  max 0.961094; all 20 worlds stationary.
- Reconstructed pinned-`d`/pinned-`K` reference MSPE: 1.8743774708, numerically
  matching the repository's top-level stored 1.874377 value.

## What is verified

1. The released correlation-based MP path was executed and independently checked.
2. Imposed `r_used` changes `d_hat`, ARI, and MSPE under a paired controlled DGP.
3. Fixed `K_true` and `K=d_hat` measure distinct downstream channels.
4. A numerically correct `d_hat` does not guarantee community recovery.
5. Forecast and clustering objectives can diverge.

## What is not verified and must not be claimed

- The above-edge directions at `r=3` have not yet been shown to align with the
  factor-loading space; “leftover-factor contamination” remains a hypothesis.
- The reason MP selects about two dimensions at the correct `r=5` is not resolved
  by this experiment.
- No iterative feedback rule, factor-selection consistency result, covariance-MP
  sensitivity, corrected-radius sensitivity, weak-SBM sensitivity, or real-data
  result has been established.
- Results are for one strong-SBM DGP and should not be presented as universal.

## Next smallest experiment

Run a pre-specified subspace-alignment diagnostic at `r_used in {3,5,7}` using
the same structural worlds. Compare the above-MP residual-correlation eigenvectors
with the true loading span and the true community-indicator span. Do not build an
iterative feedback loop until this test clarifies what `d_hat` represents when
`r` is wrong.
