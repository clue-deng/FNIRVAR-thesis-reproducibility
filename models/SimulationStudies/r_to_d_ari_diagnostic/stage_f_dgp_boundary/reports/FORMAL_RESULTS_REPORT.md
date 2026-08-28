# Stage F formal results — DGP operating boundary

## 1. Question and scope

Stage F asks when the **already-defined Stage-D feedback map** retains the true
factor count (`r_true = 5`) as a fixed point, has a usable basin of attraction,
and can be reached from the released IC_p2 initialiser. It is an operating-boundary
experiment, not a new estimator and not a free `(r,d)` search.

Two registered one-dimensional interventions were run without crossing them:

1. **Family S:** reduce community separation from S0 `(0.90, 0.10)` to the
   no-community negative control S4 `(0.30, 0.30)`, while holding nominal density
   at 0.30.
2. **Family E:** change loading scale from the shared canonical S0 value 0.40 to
   0.20 and 0.60, with the rest of the canonical DGP fixed.

Achieved eigengap `Δ` is an induced variable in both interventions. Family S
changes separation **and** achieved `Δ`; Family E changes loading amplitude **and**
achieved `Δ`. The experiment therefore does not identify a causal effect of the
sign of `Δ` alone.

## 2. Estimands and inference

The co-primary configuration was frozen as
`B_absolute × zero_fixed × primary_fixed_K`. Its three co-primary metrics are:

* `fixed_point_indicator`: whether `F(5) = 5`;
* `basin_fraction`: the fraction of starts `r0 = 1,…,9` reaching 5;
* `end_to_end_success`: whether the map reaches 5 from released IC_p2.

Five origins were averaged inside each structural world. The world is the
independent unit. Across-world summaries use untruncated Student-t intervals.
Paired contrasts use the stable-world intersection, with Holm correction applied
separately by family and metric exactly as frozen in F-24--F-27.

## 3. Execution and validation

* 20 assigned worlds per anchor; 5 origins; 9 starting `r` values; 2 K branches;
  3 map variants; 2 Bai-Ng branches.
* 71,820 valid transition rows and seven recorded unstable-DGP singleton rows.
* 665 valid initialisation rows and the same seven exclusions.
* Formal raw-output validator: **33/33 gates pass**.
* Canonical S0 reproduces frozen Stage D in **10,800/10,800** overlapping cells,
  with zero discrete mismatch and maximum ARI error 0.
* Result validator independently reconstructs 665 origin metrics, 133 world rows
  and all 21 registered contrast rows; all gates pass.
* Every registered contrast has paired `n ≥ 17`; no F-27 descriptive-only rule
  was triggered.

Invalid rates were 0% at S0, 5% at S1, 5% at S2, 10% at S3 and 15% at S4; both
loading-scale anchors had 0%. Inference is conditional on both assigned DGPs being
stable. The increasing invalid rate at the weak-separation end is itself a design
limitation, not missingness to be repaired.

## 4. Co-primary anchor estimates

| anchor | valid worlds | P[F(5)=5] | basin fraction | released IC_p2 success |
|---|---:|---:|---:|---:|
| S0 | 20 | 1.000 | 0.552 | 0.100 |
| S1 | 19 | 0.916 | 0.418 | 0.021 |
| S2 | 19 | 0.768 | 0.246 | 0.000 |
| S3 | 18 | 0.133 | 0.020 | 0.000 |
| S4 | 17 | 0.000 | 0.000 | 0.000 |
| loading scale 0.20 | 20 | 0.540 | 0.228 | 0.270 |
| loading scale 0.60 | 20 | 1.000 | 0.510 | 0.100 |

The full Student-t intervals are in `results/formal_anchor_summary.csv`; the
figure is `figures/stage_f_primary_metrics.pdf`.

## 5. Registered paired contrasts

### Family S — weaker community separation versus canonical S0

The basin contracts at every registered step relative to S0:

| contrast | Δ fixed-point probability | Holm p | Δ basin fraction | Holm p |
|---|---:|---:|---:|---:|
| S1 − S0 | −0.084 | 0.149 | −0.140 | 0.000399 |
| S2 − S0 | −0.232 | 0.0465 | −0.312 | 2.08e-08 |
| S3 − S0 | −0.867 | 3.31e-10 | −0.543 | 3.36e-15 |
| S4 − S0 | **−1.000** | **deterministic** | −0.546 | 1.12e-12 |

**Degenerate contrasts carry no p-value.** In three registered contrasts the
paired difference is constant across worlds, so its sample standard deviation is
exactly zero and a one-sample `t` statistic is undefined. These are recorded with
`p_raw = p_holm = reject_holm_0_05 = NA` and
`inferential_status = deterministic_no_variance`, and Holm is recomputed within
each `(family, metric)` group over the **testable** contrasts only, so a degenerate
contrast never inflates its family's multiplicity count. The three are:

| contrast | metric | paired n | difference |
|---|---|---:|---|
| S4 − S0 | fixed-point indicator | 17 | **−1.000 in 17/17 worlds** |
| scale 0.60 − S0 | fixed-point indicator | 20 | 0.000 in 20/20 worlds |
| scale 0.60 − S0 | end-to-end success | 20 | 0.000 in 20/20 worlds |

For `S4 − S0` the conclusion is stronger than any p-value would be: **all 17
paired worlds differ by exactly −1**, i.e. `F(5) = 5` holds in every S0 world and
in no S4 world. This is a deterministic in-sample result, not a `t`-test.

The fixed point first shows a Holm-significant loss at S2; by S3 it is rarely a
fixed point, and S4 has no basin at all.

**Per-world monotonicity (post-hoc descriptive, no significance test).** A
monotone decline in the anchor means does not imply that every world declines
monotonically. On the 15 complete-case worlds valid at all five S anchors:

| metric | weakly monotone decreasing across S0→S1→S2→S3→S4 |
|---|---|
| basin fraction | **14 / 15 (0.93)**, 45 of 60 steps strictly decreasing |
| fixed-point indicator | **13 / 15 (0.87)**, 23 of 60 steps strictly decreasing |
| end-to-end success | 15 / 15 (1.00), but only 3 of 60 steps strictly decreasing (floor) |

Exact values: `results/formal_per_world_monotonicity.csv`. The ladder result is
therefore not an artefact of averaging, but it is **not** universal at the cell
level and must not be described as such. The S4 result is expected for a
no-community negative control and must not be described as an algorithmic failure
in isolation.

Released-IC_p2 end-to-end success is already only 0.10 at S0 and falls to zero by
S2. None of its four Family-S contrasts is Holm-significant (`p_Holm = 0.344` for
all four). This is not evidence of equivalence: the floor at the canonical anchor
leaves little dynamic range and only 17--20 worlds are available.

How little dynamic range is worth making explicit. The 20 per-world S0 values are

```
[0.0 x 16,  0.2,  0.4,  0.4,  1.0]     mean 0.100,  95% t [-0.016, 0.216]
```

so the canonical estimate is carried by 4 of 20 worlds and 16 worlds are already
at zero. A contrast against that baseline has almost no room to move downward,
which is why the four null end-to-end results are uninformative rather than
evidence of no effect.

### Family E — loading scale

Relative to S0, loading scale 0.20 reduces the fixed-point probability by 0.460
(95% paired interval `[−0.650, −0.270]`, `p_Holm = 1.36e-04`) and the basin by
0.324 (`[−0.434, −0.215]`, `p_Holm = 1.85e-05`). Scale 0.60 keeps the fixed-point
probability at 1.00 — the contrast against S0 is exactly zero in all 20 worlds and
is therefore reported as deterministic, with no p-value — but its basin is 0.042
smaller than S0 (`[−0.064, −0.021]`, `p_Holm = 0.000617`). The 0.60-minus-0.20
endpoint contrast is positive for both fixed-point probability (+0.460) and basin
(+0.282), with Holm-adjusted p-values 1.36e-04 and 9.77e-05.

(The Family-E fixed-point and end-to-end Holm values are computed over two
testable contrasts rather than three, because the `scale 0.60 − S0` contrast in
those two metrics is degenerate.)

Released-IC_p2 success is numerically higher at scale 0.20 (0.27) than at S0
(0.10), but the paired difference +0.17 includes zero
(`[−0.023, 0.363]`, `p_Holm = 0.161`). This does **not** mean weaker loadings improve
the feedback map: its fixed point and basin are substantially worse. The numerical
increase arises because the initial factor count shifts downward and is therefore
more likely to fall inside the remaining lower-side basin.

## 6. Mechanism-variable description, not causal attribution

Mean achieved `Δ/λ_max(Γ_ξ)` moves from +1.322 at S0 through +0.914 (S1), +0.405
(S2), −0.219 (S3), and −0.799 (S4). It is −0.419 at loading scale 0.20 and +4.225
at scale 0.60. The loss of the fixed point along Family S therefore occurs in the
same region where achieved `Δ` crosses zero, and the loading-scale endpoints show
the same descriptive ordering. But because the registered interventions also
change community separation or loading amplitude, this is evidence of a joint
operating boundary, **not proof that the eigengap sign caused the transition**.

**Use the normalised Δ, not the raw Δ, for any sign statement from S3 onward.**
The raw achieved eigengap becomes heavy-tailed once the ladder reaches weak
separation, because a small number of worlds have realised `ρ(Φ)` close to one and
therefore a very large `λ_max(Γ_ξ)`. At S3 the raw Δ has mean −43.45, SD 125.11 and
range −515.95 to +4.91, and its Student-`t` interval `[−105.67, +18.76]` **spans
zero**, whereas the normalised `Δ/λ_max(Γ_ξ)` is −0.219 with interval
`[−0.418, −0.021]`. Two of the 18 valid S3 worlds have `ρ(Φ) > 0.99` (maximum
0.9974). The raw-Δ column in `results/formal_anchor_summary.csv` must therefore
not be read as a sign statement on its own.

## 7. Interpretation for Stage D and the thesis

Stage F strengthens, rather than reverses, Stage D:

1. In the canonical strong-SBM DGP, the proposed map has the correct fixed point
   and a lower-side basin, but released IC_p2 usually starts above it.
2. As separation weakens, the fixed point itself disappears and the basin closes.
   The method can therefore fail even with a deliberately varied starting value,
   not only because of bad initialisation.
3. Lower loading scale also damages the map. Increasing loading scale preserves
   the fixed point, but does not rescue the released initialiser in the registered
   primary analysis.

The defensible conclusion is:

> The investigated feedback procedure is conditionally feasible, but is not
> reliably self-starting and loses its fixed point and basin along the examined
> weak-signal regimes.

It is **not** defensible to claim that joint selection is universally impossible,
that eigengap sign alone causes failure, or that Stage F supplies a generally
reliable new estimator.

## 8. Secondary findings

ER and GR initialisers are secondary only. At S0 their success probabilities are
0.59 and 0.41; these decline to zero at S4. At loading scale 0.60 they reach 0.97
and 0.90, respectively. These results identify initialisation as a possible
improvement target, but they were not part of the registered co-primary contrast
families and do not establish a validated replacement rule.

All exact values are in:

* `results/formal_world_metrics.csv`
* `results/formal_anchor_summary.csv`
* `results/formal_registered_contrasts.csv`
* `results/formal_secondary_initialisers.csv`
* `results/formal_instability_by_anchor.csv`
