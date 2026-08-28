# Stage B/M1 results

20 structural worlds, 4 x 50 path replications x 5 T-values + 1 deterministic
replay per world = 1,005 cells/world, 20,100 total sample rows. All 20 worlds
stationary. 12/12 qualification gates, 12/12 formal gates, both independently
recomputed from raw outputs (`reports/QUALIFICATION_REPORT.md`).

## 1. Population spectrum — the headline finding

The **population** (not sample) leading-eigenvalue profile of the stationary
idiosyncratic correlation matrix, averaged across 20 structural worlds
(`results/population_summary.csv`, `results/figures/figure1_population_eigenvalue_profile.png`):

| `theta_j` | mean | 95% CI |
|---|---:|---|
| theta_1 | 5.667 | [5.172, 6.161] |
| theta_2 | 4.082 | [3.892, 4.271] |
| theta_3 | 1.276 | [1.265, 1.287] |
| theta_4 | 1.209 | [1.200, 1.219] |
| theta_5 | 1.019 | [1.018, 1.020] |
| theta_6 | 1.015 | [1.014, 1.017] |
| theta_7 | 1.012 | [1.011, 1.013] |
| theta_8 | 1.009 | [1.009, 1.010] |

`theta_2 − theta_3 = 2.805` [2.620, 2.991] — a huge gap. `theta_4 − theta_5 =
0.190` [0.181, 0.200] — a small gap, only slightly above the theta_5-8 "bulk"
level (~1.0-1.02). `theta_3/theta_2 = 0.316` and `theta_4/theta_2 = 0.299` —
**both** the 3rd and 4th population directions sit at under a third of the
2nd direction's strength, and much closer in magnitude to the bulk
(theta_5-8 ≈ 1.0-1.02) than to theta_1/theta_2.

**This is a population-level fact, verified via `scipy.linalg.solve_discrete_lyapunov`
and cross-checked against an independent truncated matrix-power series
(agreement confirmed for every world, qualification+formal gate 4) — not a
finite-sample artifact and not a claim about the sample MP edge.** Per
execution-prompt §5's guard, this is reported strictly as a population
description; no population eigenvalue is compared against, or counted
relative to, the sample MP edge.

The phrase "weak population separation" is descriptive, not a theoretical
spike-detectability test. The classification's `<0.5*theta_1` rule is a frozen
design heuristic; it is not a BBP threshold and does not prove that the
population dimension is two. Population truth remains `d=K_true=4` by the
simulation convention.

## 2. Sample d_hat across branches and T

`results/across_world_summary.csv`, `results/figures/figure2_mean_dhat_vs_T.png`,
`figure3_pr_dhat_ge_j_vs_T.png`:

| Branch | T=1500 mean d_hat | T=3000 mean d_hat | T=1500 Pr(d_hat≥3) | T=3000 Pr(d_hat≥3) | T=3000 Pr(d_hat≥4) |
|---|---:|---:|---:|---:|---:|
| A iid_marginal | 2.015 | 2.396 | 0.015 | 0.395 | 0.001 |
| B var_stationary_start | 2.024 | 2.401 | 0.024 | 0.393 | 0.008 |
| C var_zero_start | 2.024 | 2.403 | 0.024 | 0.395 | 0.008 |
| D var_burnin_500 | 2.033 | 2.433 | 0.033 | 0.423 | 0.010 |
| E released_replay | 2.050 | 2.500 | 0.050 | 0.500 | 0.000 |

**All five branches essentially overlap at every T** (95% CIs heavily
overlapping at every row above; see figures 2-3). d_hat frequency at T=3000
(`results/dhat_frequency_table.csv`) is dominated by {2,3}: e.g. for
`iid_marginal`, 605/1000 (60.5%) give `d_hat=2`, 394/1000 (39.4%) give
`d_hat=3`, and only 1/1000 (0.1%) reach `d_hat=4` — even at the largest
tested T, the population's 4th direction is essentially never detected by the
sample selector.

## 3. Formal paired-contrast families (Holm-corrected within family, 20 worlds)

`results/paired_contrasts.csv`, `results/figures/figure4_paired_contrasts.png`:

- **Family S (serial dependence, `var_stationary_start − iid_marginal`)**:
  **0/10 Holm-significant.** Largest raw effect: `j=3,T=1500`, mean +0.009,
  95% CI [−0.002, 0.020], `p_holm=1.0`.
- **Family I (zero-start init., `var_zero_start − var_stationary_start`)**:
  **0/10 Holm-significant.** Largest raw effect: `j=3,T=3000`, mean +0.002,
  95% CI [−0.001, 0.005], `p_holm=1.0`.
- **Family B (burn-in 500, `var_burnin_500 − var_stationary_start`)**:
  **0/10 Holm-significant.** Largest raw effect: `j=3,T=3000`, mean +0.030,
  95% CI [−0.002, 0.062], unadjusted `p=0.064`, `p_holm=0.640`.
- **Family T (finite-T detectability under iid, `T=3000 − T=1500`)**:
  **1/2 Holm-significant.** `j=3`: mean **+0.380**, 95% CI [0.282, 0.478],
  `p_holm=2.7e-7`. `j=4`: mean +0.001, 95% CI [−0.001, 0.003], `p_holm=0.330`
  (not significant — the 4th direction remains essentially undetectable even
  at the largest T tested, consistent with §1's population finding).

None of the null families (S, I, B) is claimed as "no effect" merely because
their intervals include zero — no equivalence margin was prospectively
specified, so these are reported descriptively as "no detected effect at this
design's precision," not as proof of exact equality (execution-prompt §9).

## 4. Cause classification (§10 decision tree, `results/interpretation_classification.json`)

- **A (weak population/finite-T detectability): SUPPORTED.**
  theta_3, theta_4 are population-level weak relative to theta_1/theta_2
  (§1); `iid_marginal` gives `d_hat` rounding to 2 in 100% of world-level
  means at `T≤1500`; detectability of the 3rd direction improves
  significantly with T under pure iid sampling (Family T, `j=3`).
- **B (serial-dependence distortion): NOT SUPPORTED.** Family S: 0/10
  Holm-significant.
- **C (initialization/transient effects): NOT SUPPORTED.** Family I: 0/10
  Holm-significant. (Family B being also null is consistent with, not
  independent confirmation of, this — if there were no zero-start effect to
  begin with, burn-in recovering "toward stationary-start" is close to
  vacuous.)
- **Frozen decision-tree output: A only — not mixed, not undetermined.** This
  is the actual computed classification from the pre-specified rule. The
  cautious inferential reading is narrower: A is the best-supported of the
  explanations tested, while B and C were not supported by any
  Holm-significant contrast. Because no equivalence margin or prospective
  power target for small effects was specified, the null B/C screens do not
  prove absence and should not be described as rejections.
