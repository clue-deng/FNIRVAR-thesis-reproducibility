# Full-grid mechanism results: pairwise support at every misspecified r

Combined mechanism panel: 4,500 cells (1,500 frozen Stage C + 3,000 new),
36,000 eigenvector rows, 4,000 nonzero incremental-space rows. All counts
match section-3 accounting exactly; validated 16/16 at qualification and
16/16 at the formal run (`reports/QUALIFICATION_REPORT.md`).

This report is **pairwise** (each r vs r=5, or a level test at r): see
`DOSE_RESPONSE_RESULTS.md` for whether the pattern is also monotone across
adjacent r, which is a separate, stronger claim not implied by pairwise
support alone.

## Holm families U1/U2/O1/O2 — all four families fully supported

Each family = 4 pre-specified directional tests, Holm-corrected within family
(4 comparisons). A family is "supported" only if every one of its 4 tests has
the required-sign 95% CI **and** Holm-adjusted `p<0.05`.

**U1 — under-grid `U_MP` factor-unique purity, r vs 5 (required: positive)**

| r | mean (r−5) | 95% CI | p (Holm) |
|---|---:|---|---:|
| 1 | +0.647 | [0.617, 0.678] | 9.5e-21 |
| 2 | +0.527 | [0.516, 0.539] | 1.9e-26 |
| 3 | +0.434 | [0.423, 0.445] | 1.6e-25 |
| 4 | +0.287 | [0.278, 0.297] | 2.7e-23 |

**U1 supported.**

**U2 — under-grid `U_4` capture of `Q_C`, r vs 5 (required: negative)**

| r | mean (r−5) | 95% CI | p (Holm) |
|---|---:|---|---:|
| 1 | −0.458 | [−0.484, −0.431] | 2.2e-18 |
| 2 | −0.222 | [−0.247, −0.198] | 2.1e-13 |
| 3 | −0.085 | [−0.104, −0.066] | 3.9e-08 |
| 4 | −0.031 | [−0.044, −0.017] | 1.1e-04 |

**U2 supported.**

**O1 — over-grid `Q_extra_r` community-unique excess-purity, level (required: positive)**

| r | mean level | 95% CI | p (Holm) |
|---|---:|---|---:|
| 6 | 0.821 | [0.793, 0.849] | 3.1e-23 |
| 7 | 0.650 | [0.642, 0.659] | 1.2e-30 |
| 8 | 0.470 | [0.459, 0.481] | 5.9e-26 |
| 9 | 0.367 | [0.356, 0.378] | 2.9e-24 |

**O1 supported.**

**O2 — over-grid `U_4` capture of `Q_C`, r vs 5 (required: negative)**

| r | mean (r−5) | 95% CI | p (Holm) |
|---|---:|---|---:|
| 6 | −0.268 | [−0.278, −0.259] | 1.7e-22 |
| 7 | −0.426 | [−0.438, −0.413] | 5.9e-24 |
| 8 | −0.453 | [−0.470, −0.435] | 4.8e-22 |
| 9 | −0.467 | [−0.486, −0.447] | 1.3e-21 |

**O2 supported.**

## What this establishes, and what it does not

- The **oracle mechanism** (factor-vs-community alignment of the sample
  above-edge / fixed-rank residual eigenspace) generalizes cleanly across the
  **full misspecification range tested**: every under-specified value
  `r∈{1,2,3,4}` shows the same qualitative pattern as the original `r=3`
  finding, and every over-specified value `r∈{6,7,8,9}` shows the same pattern
  as the original `r=7` finding.
- Effect sizes for the **under**-grid families (U1, U2) shrink as r approaches
  5 (e.g. U1: 0.647 at r=1 down to 0.287 at r=4). Effect sizes for the
  **over**-grid families (O1, O2) *grow* in magnitude as r moves away from 5
  (e.g. O2: −0.268 at r=6 out to −0.467 at r=9) — both patterns are the
  expected "more misspecified = more mechanism effect" direction, just phrased
  relative to different reference points (under: distance from 5 shrinking
  toward 5; over: distance from 5 growing away from 5). Whether the pattern is
  also **monotone across every adjacent step**, not just directionally
  consistent pairwise vs r=5, is a stronger claim tested separately in
  `DOSE_RESPONSE_RESULTS.md`.
- This is oracle-space evidence (`Q_F`, `Q_C`, `Q_P4` are simulation-only
  targets). It is **not** an operational signal — see
  `OBSERVABLE_FEASIBILITY_REPORT.md` for what, if anything, is available
  without those oracle spaces.
- Family-wise support at r∈{1,2,4,6,8,9} plus the original r∈{3,7} findings
  covers **8 of the 9 grid points** (r=5 is the reference, not tested against
  itself). It does not by itself establish monotonicity — that is tested
  separately and reported in `DOSE_RESPONSE_RESULTS.md`.
