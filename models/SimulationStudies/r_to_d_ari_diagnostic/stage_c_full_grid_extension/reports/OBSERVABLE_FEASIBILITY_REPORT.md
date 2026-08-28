# Observable Stage-6-feasibility panel

This is the **primary scientific deliverable** of this pass, together with
`STAGE6_GATE_DECISION.md` (frozen formal screen: **A**) and
`OPERATIONAL_STAGE6_DECISION.md` (post-result methodological audit:
**operational no-go**) — read all three together; the formal "A" below is
correct as a mechanical rule-application but is not, by itself, a recommendation.

> **Post-Stage-D scope note (2026-08-19):** this panel screens five
> spectral/residual observables. It does not screen Proposal §4.2's
> within-block Bai–Ng update. That distinct map was later implemented in
> `../../stage_d_iterative_loop/`; the present negative finding remains valid
> for the five quantities below.

Every quantity here uses only the candidate data,
the estimated residual, the candidate `r`, and the residual correlation
spectrum — **never** `Q_F`, `Q_C`, `Q_P4`, true labels, ARI, MSPE, or `r_true`.

## Classification result

| Observable | Info class | Under-side monotone (Holm) | Over-side monotone (Holm) | Mirror 4/4 significant, consistent direction | Sign frequency ≥16/20 | Classification |
|---|---|---|---|---|---|---|
| `d_hat` | K-agnostic | yes | **no** | yes | yes | **not a usable correction signal** |
| `selected_excess_spectral_mass` | K-agnostic | yes | **no** | yes | yes | **not a usable correction signal** |
| `gap_4_5` | K-informed | **no** | **no** | yes | yes | **not a usable correction signal** |
| `lambda4_over_edge` | K-informed | **no** | **no** | **no** | yes | **not a usable correction signal** |
| `residual_variance_ratio` | K-agnostic | yes | yes | yes | yes | **direction-capable** |

(Classification uses the frozen rule from execution-prompt section 8, applied
with the Holm-adjusted-significance gate consistently on both the
under-/over-monotonicity check and the mirror check — see the implementation
fix disclosed in `QUALIFICATION_REPORT.md`.)

## Why the four spectrum-shape observables fail

Figure `results/figures/figure4_primary_observable_curves.png` shows why
directly: `d_hat`, `selected_excess_spectral_mass`, and — once the boundary
step at r7→r8 is correctly held to the Holm bar — `lambda4_over_edge` and
`gap_4_5` all **plateau or dip-and-rise on the over-specified side**
(`r=5..9`), because that is the true shape of the residual spectrum's response
to over-specification in this DGP (already documented in the formal run's own
non-monotone `d_hat`-vs-`r` curve, and reproduced here at the observable
level, not invented for this test). `gap_4_5` additionally breaks on the
under-side (a sign flip at r2→r3), and `lambda4_over_edge`'s mirror contrasts
are not all in the same direction (the (r4,r6) pair flips sign relative to
the other three). None of these four failures are borderline-and-suspicious;
each is driven by a specific, visible, non-monotone segment of the real curve.

## The one survivor requires a hard caveat: `residual_variance_ratio`

`residual_variance_ratio = ||residual||_F^2 / ||centered window||_F^2` is the
**only** primary observable that formally satisfies every leg of the
pre-specified rule. Its full-grid mean is:

```
r=1: 0.770   r=2: 0.581   r=3: 0.434   r=4: 0.323   r=5: 0.277
r=6: 0.259   r=7: 0.247   r=8: 0.243   r=9: 0.238
```

**This shape is a mechanical consequence of nested PCA truncation, not
evidence specific to this DGP or to `r_true=5`.** `FactorAdjustment(window, r,
l_F)` estimates loadings as the top-`r` eigenvectors of the sample covariance
and removes their span; because PCA subspaces are nested (the top-`r` space is
always contained in the top-`(r+1)` space), the residual variance is
**mathematically guaranteed to be non-increasing in `r` for any dataset**,
regardless of whether `r` matches the true factor count. The curve above shows
exactly this: a smooth, monotone decrease across the *entire* grid with no
kink, minimum, or feature at `r=5` — visually confirmed in
`figures/figure4_primary_observable_curves.png` (rightmost panel), which
should be compared against `d_hat`'s panel (leftmost), which visibly dips and
rises around the true `r`.

The classification rule as pre-specified does not distinguish "this
observable's monotonicity is specific to where `r_true` sits" from "this
observable is monotone in `r` for purely mechanical reasons, independent of
`r_true`." `residual_variance_ratio` satisfies the letter of the rule but is a
poor practical candidate for a Stage-6 "did I pick the right r" signal,
because it would very likely show the same qualitative monotone-decreasing
shape in a DGP with a different true `r`, or even in a DGP with **no** low-
rank factor structure at all (variance removed by discarding principal
components is always non-increasing in the number discarded). This is
disclosed here as a limitation of the observable itself, not walked back from
the classification, which is reported exactly as computed (see
`results/observable_signal_classification.csv`,
`results/observable_feasibility_tests.csv`).

## Mirror-contrast detail (all five observables)

| Observable | (r1,r9) | (r2,r8) | (r3,r7) | (r4,r6) | consistent? |
|---|---:|---:|---:|---:|---|
| `d_hat` | +3.11 | +2.65 | +1.69 | +1.26 | yes |
| `selected_excess_spectral_mass` | +54.2 | +43.9 | +30.8 | +13.0 | yes |
| `gap_4_5` | +4.34 | +1.04 | +1.54 | +0.02 | yes |
| `lambda4_over_edge` | +3.01 | +0.94 | +0.68 | **−0.08** | **no** |
| `residual_variance_ratio` | +0.532 | +0.339 | +0.187 | +0.064 | yes |

All values are `mean(r_lo) − mean(r_hi)`; all significant after Holm
correction except noted. `residual_variance_ratio`'s shrinking-toward-zero
mirror gap as `(r_lo, r_hi)` narrows toward `(4,6)` is again exactly what pure
monotone decay produces near its own midpoint — not distinguishing evidence.
