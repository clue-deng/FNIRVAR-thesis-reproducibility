# Stage C: subspace-mechanism diagnostic — results

## What this experiment is (and is not)

Stage C asks: **where do the sample above-edge residual-correlation directions
point**, in a fixed, matched set of formal-run cells? It is not a new estimator,
not a mathematical proof, and not the separate Stage B/M1 question of why the
true-residual MP count plateaus near two (see the limitation paragraph in
`CLAIMS_AND_LIMITATIONS.md` — the two questions are kept fully separate here).

Design: 20 structural replications × `r_used ∈ {3,5,7}` × 25 fixed forecast
origins = 1,500 cells, reusing the exact formal DGP, seeds, and rolling window.
No GMM, OLS, forecasting, or MSPE was run in this pass; formal ARI/MSPE are
joined in read-only for context only.

Validation: qualification 16/16, full-run validation 16/16 (both independently
re-derivable; see `reports/QUALIFICATION_REPORT.md` and
`results/STAGE_C_VALIDATION.json`). Every one of the **1,500** regenerated
residual hashes (20 replications × 3 r-values × 25 origins) matches the
corresponding formal raw row exactly. (Corrected 2026-08-17: an earlier draft of
this line double-counted the r-values as `1,500×3=4,500`; the design and the
1,500-cell/12,000-eigenvector-row totals were always correct — only this one
prose line had the arithmetic error. See `reports/CLEANUP_AUDIT.md`.)

## Context from the formal run (read-only, not recomputed here)

| `r_used` | mean `d_hat` | fixed-K ARI |
|---:|---:|---:|
| 3 | 4.000 (every cell) | 0.072 |
| 5 | 2.051 | 0.640 |
| 7 | 2.284 | 0.025 |

At `r=3`, `d_hat` numerically equals `K_true=4` in every formal cell, yet ARI is
far below the `r=5` peak — the count-ambiguity puzzle Stage C was built to probe.

## Pre-specified hypothesis tests

### H-under (r=3 vs r=5)

| Limb | Metric | Paired mean (r=3 − r=5) | 95% CI | Required |
|---|---|---:|---|---|
| 1 (factor contamination) | `U_MP` factor-unique purity | **+0.434** | `[0.423, 0.445]` | strictly > 0 |
| 2 (community capture drop) | `U_4` capture of `Q_C` | **−0.085** | `[−0.104, −0.066]` | strictly < 0 |

**Both limbs pass → H-under is supported** under the pre-specified criterion,
including the stronger "supports the proposed spectral explanation of the
community-recovery failure" bar (limb 1 AND limb 2 both hold).

Levels (not just the contrast): `U_MP` factor-unique purity is 0.494 at `r=3`
versus 0.060 at `r=5` — at `r=3`, essentially half of the MP-selected subspace's
energy sits in factor directions that the true idiosyncratic residual should not
contain after correct factor removal (`r_used=3` < `r_true=5`, so the residual
still carries 2 true factor directions). Corroborating evidence: `Q_missing_3` —
the two additional estimated loading/PC directions present in the rank-5
`FactorAdjustment` fit but absent from the rank-3 fit, not defined as true
factors by construction — has mean purity 0.990 against `Q_F` and only 0.049
against `Q_C`. This high measured purity is what supports (not defines)
interpreting them as factor-aligned in this DGP: they are almost purely
factor-space directions, not community directions, which is exactly what leaks
into the top of the residual spectrum at `r=3` (see
`figures/figure_stage_c_eigenvector_projection_profile.png`: at `r=3` the first
two eigenvectors are ~97% factor-aligned and only ranks 3-4 are
community/population-aligned — i.e. `d_hat=4` is numerically correct but two of
its four dimensions are factor leftovers, not community signal).

### H-over (r=7 vs r=5)

| Limb | Metric | Value | 95% CI | Required |
|---|---|---:|---|---|
| (a) (removed-space community capture) | `Q_extra_7` community-unique excess purity (level, matched series) | **+0.650** | `[0.642, 0.659]` | strictly > 0 |
| (b) (community capture drop) | `U_4` capture of `Q_C` (r=7 − r=5) | **−0.426** | `[−0.438, −0.413]` | strictly < 0 |

**Both limbs pass → H-over is supported** under the pre-specified criterion.

`Q_extra_7` — the two directions removed going from `r=5` to `r=7`, isolated as a
rank-2 object independent of the rank-5-vs-rank-7 whole-space comparison — has
mean purity **0.625562** against `Q_C` (community contrast) at a random-subspace
baseline of only 0.03, i.e. these "extra" removed directions are disproportionately
community geometry, not leftover factor structure (`Q_extra_7` purity against
`Q_F` is only **0.006561**). Levels: `U_4` capture of `Q_C` falls from 0.520 at
`r=5` to 0.094 at `r=7` — over-specification empties the fixed-rank embedding of
most of its community content.

### `Q_P4` (dynamics-aware population target) — corroborating, not decisive

`U_MP` purity against `Q_P4` follows the same qualitative pattern as against
`Q_C` (0.463 at `r=3`, 0.881 at `r=5`, 0.117 at `r=7`), corroborating both limbs
without being required by either pre-specified criterion. `Q_P4` and `Q_C` are
not the same object at baseline (see `results/baseline_overlap.csv`); reported
side by side, not substituted for each other.

### `U_MP` community metrics (corroborating only, per pre-specified rules)

`U_MP` purity against `Q_C` is 0.326 at `r=3`, 0.616 at `r=5`, 0.087 at `r=7` —
consistent with, but not one of, the two required decisive limbs (which use
factor-unique purity and fixed-rank `U_4` capture specifically, so that the
variable rank of `U_MP` does not mechanically drive the comparison).

## Interpretation caveats (binding)

- This is controlled mechanism evidence for one strong-SBM DGP with one set of
  network/factor parameters — not a proof that alignment causes every part of
  the observed ARI decline, and not evidence about any other DGP.
- Origins (25 per replication), eigenvectors (8 per cell), and cells (1,500
  total) are repeated measurements, not independent Monte Carlo draws; every
  reported uncertainty interval is based on the 20 structural replications only.
- `d_hat` numerically equaling `K_true` (as at `r=3`, every cell) does not by
  itself imply the selected eigenspace is community-aligned — this is exactly
  what H-under's factor-unique-purity limb was built to distinguish, and here it
  does: half the `r=3` MP-selected space is factor-contaminated.

## 1. Evidence supporting H-under

Both pre-specified limbs pass with large, tightly-estimated effects (n=20
structural replications, paired 95% CIs excluding zero by a wide margin in both
directions). Corroborating eigenvector-level and `Q_missing_3`-based evidence
(factor purity 0.990, `Q_P4` pattern) is consistent with the same mechanism:
under-specifying `r` leaves two true factor directions in the residual, and
those directions occupy roughly half of the MP-selected top-4 eigenspace at
`r=3`, displacing community-aligned energy relative to `r=5`.

## 2. Evidence supporting H-over

Both pre-specified limbs pass. The rank-2 `Q_extra_7` object — the two
directions specifically removed going from `r=5` to `r=7`, not a mechanical
rank-5-vs-rank-7 whole-space comparison — is disproportionately
community-aligned (purity 0.625562 vs a ~0.03 random baseline) and almost
entirely non-factor-aligned (purity 0.006561 against `Q_F`). Fixed-rank `U_4`
capture of
`Q_C` falls sharply from `r=5` to `r=7`. Together these support over-specification
directly removing community-relevant signal from the residual, consistent with
the formal run's near-zero ARI at `r=7`.

## 3. What remains unresolved

- **Why `d_hat` plateaus near 2 at the correctly-specified `r=5`** is a separate
  question (Stage B/M1) and is not addressed here — Stage C only characterizes
  what the sample above-edge directions align with, given whatever count is
  selected, at three specific `r_used` values.
- Only three `r_used` values (`3,5,7`) were tested against the H-under/H-over
  contrasts; the non-monotone full `d_hat` vs `r_used` curve from the formal run
  (§ context table above and the formal report) is not itself mechanistically
  explained by Stage C for `r_used ∈ {1,2,4,6,8,9}`.
- `Q_P4`, the dynamics-aware population target, corroborates but was not required
  by either decisive limb; its own relationship to `Q_C`/`Q_C_full` at baseline
  (`results/baseline_overlap.csv`) shows the two targets are related but not
  identical, and Stage C does not resolve why they diverge where they do.
- This is one DGP (strong SBM, `p_in=0.9`, `p_out=0.1`, one loading scale/noise
  setting). Generalization to weaker block structure, different `l_F`/`rho_F`, or
  real data is untested here and explicitly out of scope for this pass.

## 4. Whether Stage 6 is scientifically justified yet

**Partially — for `r` in a broad under/over neighborhood of the true value, not
for an unconditional iterative rule.** Stage C gives controlled mechanism
evidence that both directions of misspecification tested (`r=3` under, `r=7`
over) corrupt the MP-selected embedding's community content, and gives a
concrete, validated diagnostic (`U_4` capture of `Q_C`, `Q_extra_7`/`Q_missing_3`
purity) that a candidate iterative rule could in principle be checked against.
It does **not** establish that these two mechanisms hold at every `r_used` in
the formal grid (only 3, 5, 7 were tested), does not establish monotonicity of
any candidate correction signal, and does not test whether an iterative
`r`→`d_hat` feedback loop would converge, oscillate, or systematically mis-correct
given the non-monotone `d_hat` vs `r_used` relationship already documented in the
formal run. Building Stage 6 on the strength of Stage C alone would extrapolate
past what was tested. The next smallest justified step is extending the same
matched-cell diagnostic to the remaining formal `r_used` grid values before any
feedback-loop code is written — not Stage 6 directly.

**Stage 6 is not implemented in this pass**, per the frozen scope.
