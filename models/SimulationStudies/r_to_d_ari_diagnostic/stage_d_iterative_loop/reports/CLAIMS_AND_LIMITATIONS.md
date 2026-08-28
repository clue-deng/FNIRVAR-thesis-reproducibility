# Stage D — claims and limitations

## What is verified (gate-checked, not asserted)

1. **The released `baing()` cannot report zero factors.** Reproduced on four
   `(T, N)` panels of pure noise: the criterion's argmin lands on the zero-factor
   slot every time, and the released function returns 1. Non-degenerate recovery at
   `k = 1,2,3,5,7` is exact, so this is a boundary defect, not a broken estimator.
   `reports/baing_audit.json`.
2. **The implemented criterion is Bai and Ng (2002) IC_p2**, not PC_p2. The
   docstring label is wrong; the formula is right. These are separate facts
   (`../DECISIONS.md` D-05a/D-05b) and only the second has numerical consequences.
3. **The independent IC reconstruction agrees with the released `baing()` on
   1.000 of all non-Variant-C rows** (13,320 rows). The 19,980 figure is the count
   of all valid rows including Variant C, which has no released equivalent. Zero
   criterion ties occurred.
4. **The 19 checked worlds regenerate with exactly matching seed trees** and
   spectral radii within `4.2e-15` of the completed formal run.
5. **`F` is a total function on `r ∈ {1,…,9}`** in all 2,220 transition tables, so
   every trajectory is a well-defined table walk (this is what `../DECISIONS.md`
   D-09's frozen per-world GMM seed buys).
6. The Stage D pipeline **independently reproduces the completed formal run's
   `d̂`-by-`r` and ARI-by-`r` profiles** to within Monte Carlo error, without
   reading any of its rows.

## What is claimed, precisely

* At the canonical strong-SBM DGP, with the zero-factor indexing repaired,
  `r_true = 5` **is a fixed point** of all three frozen operationalisations. In
  the primary fixed-`K` branch, starts at `r^(0)≤5` reach it with probability
  0.81–1.00 in one or two iterations.
* The region `r≥6` is **nearly absorbing / empirically sticky**, not strictly
  absorbing: `P(reach r_true)` is 0.00–0.07 there.
* Under the **released** `baing()`, Variant A never converges — 900/900 primary
  fixed-`K` trajectories and 1,800/1,800 trajectories across both `K` branches
  leave the grid. This is a property of the released indexing.
* The **default IC_p2 initialisation** lands at `r^(0)≥6` in 90% of cells, giving
  end-to-end `P(reach r_true)` of 0.10 for A/B and 0.12 for C. The released **ER**
  initialiser gives 0.59 [0.38, 0.80] and **GR** 0.41 [0.19, 0.63] descriptively.
* At the second anchor, which combines weaker separation with greater expected
  network density, the dominant attractors are 6–7 and end-to-end
  `P(reach r_true)=0.00` for every tested variant and initialiser.

## What is NOT claimed

* **Not** that the iterative procedure is infeasible. The opposite is now
  demonstrated on part of the domain; what fails is the entry point and the
  over-specified half of the map.
* **Not** that failure is located only at the entry point. At the canonical DGP,
  end-to-end failure is the interaction of an upward-biased initialiser with an
  update map that rarely corrects from above.
* **Not** that switching the initialiser to ER *fixes* the procedure. ER/GR were
  frozen as Stage D descriptive robustness initialisers, but ER was not designated
  as a confirmatory replacement rule or tested on an independent DGP. Its 0.59
  [0.38, 0.80] interval is wide.
* **Not** that the converged loop forecasts better. Stage D does not re-run rolling
  MSPE (`../DECISIONS.md` D-01). No forecasting claim is made.
* **Not** that `d̂` becomes a usable signal. Stage D says nothing about the five
  spectral observables the Stage 6 gate screened; that screen is untouched.
* **Not** an equivalence claim anywhere. A `P(reach r_true)` interval containing 0
  means no detection at this precision, not proof of impossibility.
* **Not** a general or causal statement about weak community separation. The
  `(0.6, 0.4)` anchor also raises expected network density from about 0.30 to 0.45.

## Limitations, carried forward into the thesis

1. **Two DGP anchors, not a grid.** `N`, `T`, `ρ(Φ)`, factor strength and `l_F` are
   fixed at the canonical values throughout. Every statement above is conditional on
   those.
2. **Realised spectral radius is not the nominal 0.9.** `GenerateNIRVAR.phi`
   rescales by `np.max(phi_eigs)`, the lexicographic max of a complex spectrum, not
   the modulus max (DGP-001). At the weak anchor this pushed 3 of 20 worlds to a
   realised radius `≥ 1`; they are excluded and counted, never silently dropped
   (`../DECISIONS.md` D-21). The nominal multiplier must never be reported as the
   realised radius.
3. **Five origins per world, not 499.** Origins are repeated measurements; the
   inference unit is the world. The five origins are a coarser sample of the
   forecast path than the formal run's 499 and are not claimed to characterise
   within-world origin variability.
4. **Bit-identity with the completed formal run does not hold on this platform.**
   Seed-level and numeric-level gates pass; SHA-256 equality of `Phi`/`Omega` is
   expected only on the original macOS/numpy-2.3.5 machine.
5. **Three variants, not one.** Proposal §4.2 does not determine the update
   uniquely. All three readings are reported precisely so that no single one can be
   selected after seeing the outcome. They agree on every qualitative conclusion
   above, which is itself reassuring, but the thesis must present all three.
6. **The second-anchor failure is confounded.** Basin width is associated with the
   ARI profiles (0.635 at strong, 0.052 at the second anchor, both at `r=5`), but
   separation, network density, coefficient structure and instability frequency
   all change together.
7. **Provenance cleanup.** The first failed formal launch was not retained as an
   immutable directory. The initialisation run also admitted two nonstationary
   weak-anchor worlds; stable-world matching leaves end-to-end results unchanged,
   but the original weak-anchor initialiser inventory is superseded.

## Next smallest experiments

1. **Density-matched separation dose–response.** Vary `p_in-p_out` while holding
   expected edge density fixed, and locate where `r_true` stops being a reliable
   fixed point. This is the smallest experiment that can isolate separation from
   the density change in the current second anchor.
2. **Frozen initialiser comparison.** Pre-register ER versus IC_p2 as the loop's
   entry point, with a stopping rule, and evaluate on an independent DGP draw. Only
   then is the 0.59 figure a claim.
3. **MSPE of the converged loop** versus sequential FNIRVAR and versus an
   `(r, d)` grid-search upper bound, with parameter selection and evaluation on
   disjoint origin sets.
