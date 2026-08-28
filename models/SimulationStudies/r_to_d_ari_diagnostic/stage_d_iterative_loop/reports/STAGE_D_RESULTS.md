# Stage D results — the Proposal §4.2 iterative feedback procedure, implemented

**Run:** `runs/formal_20260819` (transition map) + `runs/formal_init_20260819`
(realised initialisation). **Summaries:** `results/`. **Design:** `../DECISIONS.md`,
frozen before outcomes were inspected. **Released package: not modified.**

---

## Executive result

The Proposal §4.2 loop had never been implemented. It now has been, in all three
defensible operationalisations (`../DECISIONS.md` §2), under both Bai–Ng branches,
both `K` branches, and at two DGP anchors.

**At the canonical strong-SBM DGP, the loop closes reliably from below in the
primary fixed-`K` branch, but rarely repairs over-specification. A zero-factor
indexing defect breaks the released incremental variant, and the standard
initialiser starts in the empirically sticky upper region in 90% of cells.**

Four findings, in the order they matter:

1. Under zero-fixed Bai–Ng, `r_true = 5` **is** a fixed point of all three variants
   at the canonical DGP. In the primary fixed-`K` branch, starts at `r^(0)≤5`
   reach it with probability 0.81–1.00 in **one or two iterations**. The
   `K=d_hat` branch has the same shape but a weaker far-left edge.
2. Over-specification is **nearly absorbing / empirically sticky**, not strictly
   absorbing. `P(reach r_true)` from `r^(0)≥6` is only 0.00–0.07.
3. The released `baing()` zero-factor defect **converts Variant A from a
   converging map into a monotone escape** — 900/900 primary fixed-`K`
   trajectories (1,800/1,800 across both `K` branches) leave the grid.
4. The realised Bai–Ng IC_p2 initialisation lands at `r^(0) ≥ 6` in **90%** of
   cells, i.e. outside the basin. End-to-end `P(reach r_true)` is therefore
   **0.10–0.12**, depending on the variant. Swapping the initialiser to the
   released `ER` estimator gives **0.59 [0.38, 0.80]** descriptively, but this is
   not an independently validated replacement rule.

At the second anchor, which has both weaker block separation and greater expected
network density, the dominant attractors are **wrong** (6 or 7). End-to-end
`P(reach r_true)=0.00` for every tested variant and initialiser. This contrast does
not identify a pure community-separation effect.

---

## 1. Validation

| gate | result |
|---|---|
| rows | 19,983 of 21,600 nominal; the 1,617-row gap is exactly the 3 excluded `weak_sbm` worlds (D-21) |
| cell status | 19,980 `ok`; 2 `invalid_unstable_DGP`; 1 `world_generation_failure` — all at `weak_sbm` |
| worlds used | `strong_sbm` 20/20, `weak_sbm` 17/20 |
| independent reconstruction vs released `baing()` | **1.000** agreement on all **13,320** non-Variant-C rows |
| criterion ties | **0** |
| `F` total on `r ∈ {1,…,9}` | true for all 2,220 transition tables |
| world reproduction, seed level | 19/19 **exact** (structural, network, factor, GMM seeds) |
| world reproduction, numeric level | 19/19 pass; max `|Δ|` spectral radius `4.2e-15` |
| `Phi`/`Omega` bit-identity | 0/19 — platform difference, see `../DECISIONS.md` D-19 |

### Post-run cleanup validation

The preserved transition rows were not rewritten. Cleanup instead archived the
exact executed source, hardened stationarity handling, reran qualification with
the current scripts, regenerated initialisation on the stable-world set, and
recomputed all summaries:

| cleanup check | result |
|---|---|
| exact executed source | `scripts/executed_20260819/` |
| current-code qualification | 1,296/1,296 transition rows complete |
| corrected initialisation | 185 stable rows + 3 explicit `invalid_unstable_DGP` rows |
| stable-world join | 185/185 matched; 0 unmatched valid rows |
| end-to-end accounting | 8,880/8,880 expected rows |
| hardened validator | every Boolean gate true |

See `results/STAGE_D_VALIDATION_HARDENED.json` and
`reports/CODEX_CLEANUP_AUDIT.md`.

The pipeline also independently reproduces the completed formal run's behaviour:
mean `d̂` by `r` is `5.47, 4.99, 4.00, 3.00, 2.07, 1.78, 2.34, 2.31, 2.40` here
versus `5.474, 4.975, 4.000, 3.000, 2.051, 1.735, 2.284, 2.332, 2.392` there, and
mean ARI is `0.012, 0.025, 0.074, 0.241, 0.635, 0.289, 0.026, 0.009, 0.009` versus
`0.011, 0.027, 0.072, 0.237, 0.640, 0.293, 0.025, 0.011, 0.007`.

---

## 2. The transition map (mean `F(r)`, `K = K_true`, 20 worlds × 5 origins)

`results/transition_map.csv`; figure `results/figures/fig_stage_d_transition_map.pdf`.

### strong SBM (`p_in`=0.9, `p_out`=0.1) — the canonical thesis DGP

| variant / Bai–Ng | r=1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| A / released | 5.19 | 5.07 | 5.02 | 5.00 | **6.00** | 7.00 | 8.00 | 9.00 | 10.00 |
| A / zero-fixed | 5.19 | 5.07 | 5.02 | 5.00 | **5.00** | 6.04 | 7.00 | 8.00 | 9.00 |
| B / either | 5.19 | 5.06 | 5.02 | 5.00 | **5.00** | 5.98 | 6.40 | 6.40 | 6.39 |
| C / either | 5.04 | 5.01 | 5.01 | 5.00 | **5.00** | 6.00 | 6.59 | 6.57 | 6.61 |

Read the `r = 5` column first. Under `zero_fixed`, all three variants return exactly
5: **`r_true` is a fixed point.** Under `released`, Variant A returns 6 — it cannot
report "zero factors left", so it must add at least one every round.

Read the left half next. `F(r) ≈ 5` for every `r ≤ 4`, so under-specification is
corrected in a **single step**, from any starting point. This is the opposite of
what the ARI table would suggest: at `r = 3` community recovery has already
collapsed (ARI 0.074) and yet the update is still almost exactly right. This is
consistent with the block filter leaving leftover factor signal visible even when
ARI is poor. It does **not** prove that clustering quality is irrelevant; an
oracle/estimated/random-block intervention would be needed for that mechanism
claim.

Read the right half last. Most cells remain at or above six, although Variants B/C
occasionally return to five. This near-absorbing pattern is consistent with Stage
C's O1/O2 result, but Stage D does not by itself prove that removed
community-aligned directions cause the transition failure.

Variants B and C are **identical** across Bai–Ng branches, because their argmin
never falls on the zero-factor slot: an absolute re-selection on a panel that still
contains the common component always finds factors. The defect is specific to the
incremental reading.

### weak SBM (`p_in`=0.6, `p_out`=0.4)

| variant / Bai–Ng | r=1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| A / zero-fixed | 6.00 | 6.01 | 6.00 | 6.04 | **6.60** | 6.81 | 7.00 | 8.00 | 9.00 |
| B / either | 6.00 | 6.01 | 6.00 | 6.02 | **6.58** | 6.74 | 6.94 | 6.94 | 6.94 |
| C / either | 6.92 | 6.94 | 6.92 | 6.87 | **6.75** | 6.79 | 6.95 | 6.94 | 6.96 |

`r_true=5` is **not a reliable fixed point** at this anchor: A/B never return five
from five, while C does so in only 3.5% of primary fixed-`K` cells and 8.2% of
`K=d_hat` cells. The dominant fixed points are six or seven. Mean ARI at `r=5` is
0.052 versus 0.635 at the strong anchor, which is consistent with a poor block
filter, but separation, density and stability all changed together and the causal
mechanism is therefore unresolved.

---

## 3. Basin of attraction

`results/basin_summary.csv`; figure `results/figures/fig_stage_d_basin.pdf`.
Probabilities are averaged over the 5 origins within a world, then across worlds,
with 95% Student-`t` intervals on 19 df (16 df at `weak_sbm`).

### strong SBM, `K = K_true`, zero-fixed Bai–Ng

| `r^(0)` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.81 [0.66,0.96] | 0.93 [0.84,1.02] | 0.98 [0.94,1.02] | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B | 0.81 [0.66,0.96] | 0.94 [0.85,1.03] | 0.98 [0.94,1.02] | 1.00 | 1.00 | 0.03 | 0.07 | 0.07 | 0.07 |
| C | 0.96 [0.89,1.03] | 0.99 [0.97,1.01] | 0.99 [0.97,1.01] | 1.00 | 1.00 | 0.04 | 0.07 | 0.07 | 0.04 |

Under `released`, Variant A's whole row is 0.00: 900/900 trajectories terminate
`out_of_grid`. Variants B and C are unchanged by the branch.

Convergence is fast where it happens: mean iterations to a fixed point are 1.00 at
`r^(0) = 5` and 2.00–2.01 for `r^(0) ∈ {1,2,3,4}`. Representative trajectories
(Variant B, strong SBM): `3->5->5` in 98/100 cells, `1->5->5` in 81/100,
`5->5` in 100/100, `7->7` in 47/100 and `7->6->6` in 46/100.

**Robustness branch `K = d̂`** (`results/basin_summary.csv`): the same shape with a
weaker left edge — `P(reach r_true)` at `r^(0) = 1` falls from 0.81 to 0.52 for A/B
and from 0.96 to 0.87 for C, while `r^(0) ∈ {4,5}` stays at 1.00 and `r^(0) ≥ 6`
stays at 0.00–0.05. Propagating `d̂` into `K` costs basin width at the far
under-specified end and changes nothing else.

### weak SBM

Every cell is 0.00–0.04. There is no basin.

---

## 4. Where the loop actually starts

`results/initialisation.csv`, `results/end_to_end_summary.csv`. A basin result is
not usable on its own; the practical question is whether the realised `r^(0)` lands
inside it.

Realised `r^(0)` over 100 (dgp × world × origin) cells at strong SBM:

| initialiser | distribution of `r^(0)` | mean | share `≤ 5` |
|---|---|---|---|
| Bai–Ng IC_p2 (`baing(jj=2)`, the repository default) | 5: 10, **6: 60, 7: 30** | 6.20 | **10%** |
| ER (released) | 4: 25, 5: 34, 7: 41 | 5.57 | **59%** |
| GR (released) | 4: 9, 5: 32, 7: 59 | 6.09 | **41%** |

Composing initialisation with the basin gives the end-to-end probability that a
practitioner running this loop reaches `r_true` (strong SBM, `K = K_true`):

| variant / Bai–Ng | init = IC_p2 | init = ER | init = GR |
|---|---|---|---|
| A / released | 0.00 | 0.00 | 0.00 |
| A / zero-fixed | 0.10 [−0.02, 0.22] | **0.59 [0.38, 0.80]** | 0.41 [0.19, 0.63] |
| B / either | 0.10 [−0.02, 0.22] | **0.59 [0.38, 0.80]** | 0.41 [0.19, 0.63] |
| C / either | 0.12 [0.00, 0.24] | **0.59 [0.38, 0.80]** | 0.41 [0.19, 0.63] |

At weak SBM every entry is 0.00, for every initialiser.

The default initialiser over-specifies `r` 90% of the time and therefore usually
starts in the sticky upper region. End-to-end failure is an **interaction** between
this upward-biased entry point and an asymmetric update that corrects from below
but rarely from above. It cannot be attributed to the initialiser alone.

---

## 5. What this changes about the Stage 6 no-go

The Stage 6 gate screened five spectral observables and found none that is both
stable and locally identifying. That result stands and is not contradicted here:
Stage D's map is not built from any of them.

What Stage D changes is the **scope** of the conclusion. It is now demonstrated,
not assumed, that:

* a within-block Bai–Ng update does define a map with `r_true` as a fixed point at
  the canonical DGP, so "no usable update exists" was too strong;
* the binding constraints are the interaction of (i) a nearly absorbing
  over-specified region and (ii) an upward-biased default initialiser;
* at the second, weaker-separation and denser-network anchor, dominant attractors
  are wrong. This is a DGP contrast, not an isolated separation effect.

The correct wording everywhere is now:

> No reliable correction signal was found among the five tested spectral
> observables. The Proposal §4.2 within-block Bai–Ng update, implemented in Stage D,
> does have `r_true` as a fixed point at the canonical DGP and corrects
> under-specification effectively in the primary fixed-K branch. The
> over-specified region is empirically sticky and the default IC_p2 initialisation
> lands there in 90% of cells. At a second anchor with weaker separation and
> greater density, the dominant attractors are wrong.

---

## 6. Reproduction

The exact executed source is preserved in `scripts/executed_20260819/`. Top-level
scripts contain the later stationarity/join cleanup. Raw transition outputs were
not rerun. Use `reports/REPRODUCTION_COMMANDS.md` for the executed-source and
post-cleanup validation commands.

Deterministic given `MASTER_SEED = 20260727`. On the original macOS machine the
world-reproduction gate is additionally expected to pass at the bit level
(`../DECISIONS.md` D-19).
