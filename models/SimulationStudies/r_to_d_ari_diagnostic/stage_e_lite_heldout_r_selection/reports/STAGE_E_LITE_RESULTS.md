# Stage E-lite results — can held-out forecasting MSPE select a structurally adequate `r`?

**Design:** `../DECISIONS.md` (post-hoc; see E-00). **Sources and provenance:**
`SOURCE_AUDIT.md`. **Gates:** `../validation/STAGE_E_LITE_VALIDATION.json`
(30/30 pass). **No simulation was executed.**

Scope: only `r` is selected. For every `r`, `d̂(r)` remains the value the
released Marchenko–Pastur rule produced in the frozen run. This is **not** a
free `(r, d)` search (E-02).

---

## Executive result

Held-out forecasting MSPE is an excellent predictor of *its own* objective and
close to uninformative about community structure.

| within-world Spearman across `r = 1,…,9` | deployable `K = d̂` | controlled `K = K_true` |
|---|---|---|
| tuning MSPE vs **evaluation MSPE** | **0.88** | **0.93** |
| tuning MSPE vs **evaluation ARI** | **−0.11** | **−0.21** |

The criterion transfers almost perfectly out of sample *for prediction*. It
carries essentially no information about the quantity the network step exists to
recover.

The consequence is an asymmetry: choosing `r` by past forecast error costs
**about 0.2% of attainable prediction** and **about two-thirds of attainable
community recovery**.

---

## 1. What gets selected

20 structural worlds; selection uses the first 249 origins, all reported metrics
the last 250. No tuning minimum was numerically tied in any world.

**Deployable pipeline (`K = d̂`) — the practitioner's actual configuration**

| | `r = 4` | `r = 5` | `r = 6` | `r = 7` | `r = 8` | `r = 9` |
|---|---|---|---|---|---|---|
| selected by tuning MSPE | 4 | **3** | 0 | **9** | 3 | 1 |
| structure oracle (max evaluation ARI) | 5 | **14** | 1 | 0 | 0 | 0 |

`Pr(r_selected = r_true) = 0.15` (Wilson 95% `[0.05, 0.36]`);
`Pr(|r_selected − r_true| ≤ 1) = 0.35` `[0.18, 0.57]`; mean `|r_selected − 5| = 1.75`.
The modal selection is `r = 7`, where mean evaluation ARI is 0.02.

**Controlled diagnostic (`K = K_true`)**

| | `r = 5` | `r = 7` | `r = 8` |
|---|---|---|---|
| selected by tuning MSPE | **15** | 2 | 3 |
| structure oracle | **20** | 0 | 0 |

`Pr(r_selected = r_true) = 0.75` `[0.53, 0.89]`. The structure oracle is `r = 5`
in **20/20** worlds — with `K` pinned, the structurally correct answer is
unambiguous, and the criterion still misses it in a quarter of worlds.

Pinning `K` to the truth is not available to a practitioner. The gap between the
two branches (0.75 versus 0.15) is itself a result: propagating `d̂` into `K`
does not merely add noise downstream, it **relocates the predictive optimum**.

## 2. The two regrets

| | deployable `K = d̂` | controlled `K = K_true` |
|---|---|---|
| predictive regret (evaluation MSPE) | 0.0040 `[0.0016, 0.0064]` | 0.0025 `[0.0005, 0.0044]` |
| — as a share of oracle MSPE | **0.21%** | **0.13%** |
| structural regret (evaluation ARI) | 0.221 `[0.147, 0.294]` | 0.149 `[0.025, 0.274]` |
| — as a share of attainable ARI | **65.1%** `[44%, 86%]` | **24.2%** `[4%, 44%]` |
| ARI at selected `r` | 0.114 | 0.488 |
| ARI at `r_true` | 0.316 | 0.638 |
| MSPE at selected `r` | 1.9223 | 1.9029 |
| MSPE at `r_true` | 1.9295 | 1.9033 |

Note the sign in the last two rows: in the deployable branch the selected `r`
has a **lower** evaluation MSPE than `r_true` (by 0.0072). The criterion is not
failing at its own job; `r_true` is genuinely not the prediction optimum here.
The two objectives simply do not coincide.

## 3. Models indistinguishable by prediction, far apart in structure

Near-optimal sets are defined on **tuning** MSPE only (E-08):
`S_ε = {r : tuningMSPE(r) ≤ (1+ε)·min}`.

| ε | mean \|S_ε\| | mean evaluation-ARI range in `S_ε` (95% t) | `Pr(r_true ∈ S_ε)` |
|---|---|---|---|
| **deployable `K = d̂`** | | | |
| 0.1% | 1.20 | 0.011 `[−0.003, 0.025]` | 0.15 |
| 0.5% | 3.00 | **0.220** `[0.143, 0.296]` | 0.65 |
| 1.0% | 4.35 | **0.300** `[0.252, 0.349]` | 0.90 |
| 2.0% | 5.60 | 0.332 `[0.312, 0.351]` | 1.00 |
| **controlled `K = K_true`** | | | |
| 0.5% | 3.10 | **0.448** `[0.312, 0.585]` | 0.85 |
| 1.0% | 4.60 | **0.609** `[0.547, 0.672]` | 0.95 |

Read the 0.5% row of the deployable branch: three values of `r` that a
practitioner could not separate on past forecast error to better than half a
percent differ by 0.22 in future ARI — roughly the entire attainable range in
that branch. Widening the tolerance to 1% brings `r_true` into the set 90% of the
time, but by then the set spans essentially the whole ARI range: the criterion
localises the truth only by becoming uninformative.

## 4. Figures

* `results/figures/fig_stage_e_lite_curves.pdf` — tuning MSPE (top) and
  evaluation ARI (bottom) against `r`, one column per branch. The MSPE profile is
  a plateau from `r = 4` onward; the ARI profile is a spike at `r = 5`.
* `..._selection.pdf` — selected `r` against the structure oracle, per branch.
* `..._regret.pdf` — predictive regret against structural regret, one point per world.
* `..._near_tie.pdf` — evaluation-ARI range inside tuning-defined `S_ε`.

Shown as small multiples; no dual-axis chart is used, and every series is
labelled as well as coloured.

## 5. What this adds to Stage D

Stage D showed the Proposal §4.2 feedback map has `r_true` as a fixed point but
corrects only from below, while the default IC_p2 initialisation starts above it
in 90% of cells. The obvious rejoinder is that a practitioner would not iterate
at all — they would tune on held-out forecast error.

Stage E-lite tests that rejoinder inside the released pipeline and finds it does
not rescue the situation: the criterion is highly reliable for prediction and
nearly orthogonal to structure, and in the deployable configuration it lands on
`r_true` in 3 of 20 worlds.

## 6. Reproduction

See `REPRODUCTION_COMMANDS.md`.
