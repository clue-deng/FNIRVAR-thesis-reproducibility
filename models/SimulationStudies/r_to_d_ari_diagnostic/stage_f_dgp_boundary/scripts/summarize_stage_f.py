#!/usr/bin/env python3
"""Summarise the formal Stage-F operating-boundary experiment.

The frozen inferential unit is the structural world. The five origins are first
averaged within world; Student-t intervals and paired tests are then computed
across worlds. Registered contrasts and Holm families follow DECISIONS.md
F-24--F-27 exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reused_stage_d as C  # noqa: E402

PRIMARY = {
    "K_branch": "primary_fixed_K",
    "variant": "B_absolute",
    "baing_branch": "zero_fixed",
}
METRICS = ["fixed_point_indicator", "basin_fraction", "end_to_end_success"]
ANCHOR_ORDER = [
    "S0", "S1", "S2", "S3", "S4",
    "loading_scale_negative_gap", "loading_scale_positive_gap",
]
CONTRASTS = [
    ("S", "S1", "S0"),
    ("S", "S2", "S0"),
    ("S", "S3", "S0"),
    ("S", "S4", "S0"),
    ("E", "loading_scale_negative_gap", "S0"),
    ("E", "loading_scale_positive_gap", "S0"),
    ("E", "loading_scale_positive_gap", "loading_scale_negative_gap"),
]


def mean_ci(values: pd.Series | np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, mcse=np.nan,
                    ci_low=np.nan, ci_high=np.nan)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    mcse = sd / math.sqrt(n) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * mcse) if n > 1 and sd > 0 else 0.0
    return dict(n=n, mean=mean, sd=sd, mcse=mcse,
                ci_low=mean - half, ci_high=mean + half)


def paired_result(diff: np.ndarray) -> dict:
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    out = mean_ci(x)
    # A one-sample t-test on a CONSTANT paired difference is undefined (sd = 0
    # gives 0/0 or +/-inf), so no p-value is emitted for such a contrast. The
    # estimate itself is deterministic in the sample and is reported as such.
    if len(x) < 2 or out["sd"] == 0.0:
        p = np.nan
    else:
        p = float(stats.ttest_1samp(x, 0.0).pvalue)
    out["p_raw"] = p
    out["degenerate_zero_variance"] = bool(len(x) >= 2 and out["sd"] == 0.0)
    return out


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values, preserving input index."""
    valid = pvalues.dropna().astype(float)
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    if valid.empty:
        return out
    ordered = valid.sort_values(kind="mergesort")
    m = len(ordered)
    running = 0.0
    for rank, (idx, p) in enumerate(ordered.items()):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out.loc[idx] = running
    return out


def primary_origin_metrics(cells: pd.DataFrame, composition: pd.DataFrame) -> pd.DataFrame:
    ok = cells[cells.status.eq("ok")].copy()
    pri = ok[
        ok.K_branch.eq(PRIMARY["K_branch"])
        & ok.variant.eq(PRIMARY["variant"])
        & ok.baing_branch.eq(PRIMARY["baing_branch"])
    ]
    rows = []
    keys = ["anchor", "replication", "origin_index"]
    for key, group in pri.groupby(keys, sort=True):
        if set(group.r_used.astype(int)) != set(C.R_GRID) or len(group) != len(C.R_GRID):
            raise ValueError(f"Incomplete primary transition table: {key}")
        transition = {
            int(row.r_used): {"next": int(row.r_next), "status": "ok"}
            for _, row in group.iterrows()
        }
        walks = [C.iterate_from_table(transition, r0) for r0 in C.R_GRID]
        r5 = group[group.r_used.eq(C.R_TRUE)].iloc[0]
        rows.append({
            "anchor": key[0],
            "replication": int(key[1]),
            "origin_index": int(key[2]),
            "fixed_point_indicator": int(transition[C.R_TRUE]["next"] == C.R_TRUE),
            "basin_fraction": float(np.mean([
                w["stop_state"] == "correct_fixed_point" for w in walks
            ])),
            "d_hat_at_r_true": float(r5.d_hat),
            "ARI_at_r_true": float(r5.ARI),
        })
    origin = pd.DataFrame(rows)

    comp = composition[
        composition.K_branch.eq(PRIMARY["K_branch"])
        & composition.variant.eq(PRIMARY["variant"])
        & composition.baing_branch.eq(PRIMARY["baing_branch"])
        & composition.initialiser.eq("released")
    ][keys + ["r0", "reached", "stop_state", "terminal_r", "n_iter"]].copy()
    if comp.duplicated(keys).any():
        raise ValueError("Duplicate primary released-initialiser composition rows")
    comp = comp.rename(columns={
        "r0": "r0_released",
        "reached": "end_to_end_success",
        "stop_state": "released_stop_state",
        "terminal_r": "released_terminal_r",
        "n_iter": "released_n_iter",
    })
    out = origin.merge(comp, on=keys, how="left", validate="one_to_one")
    if out.end_to_end_success.isna().any():
        raise ValueError("Unmatched primary initialisation composition row")
    return out


def world_metrics(origin: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    metrics = METRICS + ["r0_released", "d_hat_at_r_true", "ARI_at_r_true"]
    world = origin.groupby(["anchor", "replication"], as_index=False)[metrics].mean()
    metadata = (cells[cells.status.eq("ok")]
                .drop_duplicates(["anchor", "replication"])[[
                    "anchor", "family", "p_in", "p_out", "separation",
                    "loading_scale", "loading_sigma", "replication",
                    "structural_seed", "actual_spectral_radius",
                    "realised_offdiag_density", "delta",
                    "delta_over_lambda_max_Gamma_xi",
                ]])
    world = world.merge(metadata, on=["anchor", "replication"],
                        how="left", validate="one_to_one")
    world["n_origins"] = origin.groupby(["anchor", "replication"]).size().values
    return world


def anchor_summary(world: pd.DataFrame) -> pd.DataFrame:
    rows = []
    report_metrics = METRICS + [
        "r0_released", "d_hat_at_r_true", "ARI_at_r_true", "delta",
        "delta_over_lambda_max_Gamma_xi", "actual_spectral_radius",
        "realised_offdiag_density",
    ]
    for anchor in ANCHOR_ORDER:
        group = world[world.anchor.eq(anchor)]
        for metric in report_metrics:
            rows.append({"anchor": anchor, "metric": metric,
                         **mean_ci(group[metric].values)})
    return pd.DataFrame(rows)


def registered_contrasts(world: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, anchor_a, anchor_b in CONTRASTS:
        for metric in METRICS:
            a = world[world.anchor.eq(anchor_a)][["replication", metric]]
            b = world[world.anchor.eq(anchor_b)][["replication", metric]]
            paired = a.merge(b, on="replication", suffixes=("_a", "_b"),
                             validate="one_to_one")
            result = paired_result(paired[f"{metric}_a"] - paired[f"{metric}_b"])
            degenerate = result.pop("degenerate_zero_variance", False)
            rows.append({
                "family": family,
                "contrast": f"{anchor_a}-{anchor_b}",
                "anchor_a": anchor_a,
                "anchor_b": anchor_b,
                "metric": metric,
                "n_paired": result.pop("n"),
                "estimate_a_minus_b": result.pop("mean"),
                "sd_difference": result.pop("sd"),
                "degenerate_zero_variance": degenerate,
                **result,
            })
    table = pd.DataFrame(rows)
    # Holm is recomputed within each pre-registered (family, metric) group over
    # the TESTABLE contrasts only; a zero-variance contrast contributes no
    # p-value and is therefore excluded from its family's multiplicity count.
    table["p_holm"] = np.nan
    for (_, _), idx in table.groupby(["family", "metric"]).groups.items():
        table.loc[idx, "p_holm"] = holm_adjust(table.loc[idx, "p_raw"])
    table["reject_holm_0_05"] = np.where(
        table.p_holm.isna(), pd.NA, table.p_holm.lt(0.05))
    table["inferential_status"] = np.where(
        table.degenerate_zero_variance, "deterministic_no_variance",
        np.where(table.n_paired.ge(10), "formal_paired_inference",
                 "descriptive_n_below_10"))
    return table


def secondary_initialisers(composition: pd.DataFrame) -> pd.DataFrame:
    pri = composition[
        composition.K_branch.eq(PRIMARY["K_branch"])
        & composition.variant.eq(PRIMARY["variant"])
        & composition.baing_branch.eq(PRIMARY["baing_branch"])
    ].copy()
    within = (pri.groupby(["anchor", "initialiser", "replication"], as_index=False)
              .agg(reached=("reached", "mean"), mean_r0=("r0", "mean")))
    rows = []
    for (anchor, initialiser), group in within.groupby(["anchor", "initialiser"], sort=False):
        reach = mean_ci(group.reached.values)
        r0 = mean_ci(group.mean_r0.values)
        rows.append({
            "anchor": anchor, "initialiser": initialiser,
            "n_worlds": reach["n"], "P_reach_r_true": reach["mean"],
            "reach_sd": reach["sd"], "reach_mcse": reach["mcse"],
            "reach_ci_low": reach["ci_low"], "reach_ci_high": reach["ci_high"],
            "mean_r0": r0["mean"], "r0_sd": r0["sd"],
        })
    return pd.DataFrame(rows)


def stop_state_summary(origin: pd.DataFrame) -> pd.DataFrame:
    counts = (origin.groupby(["anchor", "released_stop_state"])
              .size().rename("n_origin_rows").reset_index())
    totals = counts.groupby("anchor").n_origin_rows.transform("sum")
    counts["proportion"] = counts.n_origin_rows / totals
    return counts


def instability_table(cells: pd.DataFrame, config: dict) -> pd.DataFrame:
    invalid = (cells[cells.status.ne("ok")]
               .drop_duplicates(["anchor", "replication"])
               .groupby("anchor").size().to_dict())
    rows = []
    for anchor in ANCHOR_ORDER:
        assigned = len(config["replications"])
        n_invalid = int(invalid.get(anchor, 0))
        rows.append({"anchor": anchor, "n_assigned": assigned,
                     "n_invalid_unstable": n_invalid,
                     "invalid_rate": n_invalid / assigned,
                     "n_valid": assigned - n_invalid})
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, out: Path) -> None:
    labels = ["S0", "S1", "S2", "S3", "S4", "scale 0.20", "scale 0.60"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True)
    for ax, metric, title in zip(
        axes, METRICS,
        ["True r is a fixed point", "Basin fraction", "Released IC_p2 success"],
    ):
        sub = (summary[summary.metric.eq(metric)]
               .set_index("anchor").loc[ANCHOR_ORDER])
        x = np.arange(len(sub))
        ax.errorbar(x, sub["mean"],
                    yerr=np.vstack([sub["mean"] - sub["ci_low"],
                                    sub["ci_high"] - sub["mean"]]),
                    fmt="o", capsize=3, color="#2455a4")
        ax.axvline(4.5, color="0.75", linewidth=1)
        ax.set_title(title)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("World-level mean (95% t interval)")
    fig.suptitle("Stage F: feedback feasibility across registered DGP anchors", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



def per_world_monotonicity(world: pd.DataFrame,
                           ladder=("S0", "S1", "S2", "S3", "S4")) -> pd.DataFrame:
    """
    Post-hoc descriptive diagnostic (no significance test).

    A monotone decline in the ANCHOR MEANS does not imply that every structural
    world declines monotonically along the ladder. This reports, on the
    complete-case worlds (valid at all five S anchors), the fraction whose own
    trajectory is weakly monotone decreasing across S0 -> S1 -> S2 -> S3 -> S4.
    """
    rows = []
    for metric in ["fixed_point_indicator", "basin_fraction", "end_to_end_success"]:
        piv = (world[world.anchor.isin(ladder)]
               .pivot(index="replication", columns="anchor", values=metric)
               .reindex(columns=list(ladder)))
        cc = piv.dropna()
        if len(cc) == 0:
            continue
        v = cc.to_numpy(dtype=float)
        mono = (v[:, :-1] >= v[:, 1:]).all(axis=1)
        strict_steps = int((v[:, :-1] > v[:, 1:]).sum())
        rows.append({
            "metric": metric,
            "ladder": "->".join(ladder),
            "n_complete_case_worlds": int(len(cc)),
            "n_weakly_monotone_decreasing": int(mono.sum()),
            "fraction_weakly_monotone_decreasing": float(mono.mean()),
            "n_strictly_decreasing_steps": strict_steps,
            "n_total_steps": int(v.shape[0] * (v.shape[1] - 1)),
            "note": "post-hoc descriptive; no significance test is attached",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--init-run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.run / "transition_cells.csv", low_memory=False)
    composition = pd.read_csv(args.init_run / "composed_reachability.csv")
    config = json.loads((args.run / "config_used.json").read_text())

    origin = primary_origin_metrics(cells, composition)
    world = world_metrics(origin, cells)
    summary = anchor_summary(world)
    contrasts = registered_contrasts(world)
    secondary = secondary_initialisers(composition)
    stop_states = stop_state_summary(origin)
    instability = instability_table(cells, config)

    origin.to_csv(args.outdir / "formal_origin_metrics.csv", index=False,
                  lineterminator="\n")
    world.to_csv(args.outdir / "formal_world_metrics.csv", index=False,
                 lineterminator="\n")
    summary.to_csv(args.outdir / "formal_anchor_summary.csv", index=False,
                   lineterminator="\n")
    contrasts.to_csv(args.outdir / "formal_registered_contrasts.csv", index=False,
                     lineterminator="\n")
    secondary.to_csv(args.outdir / "formal_secondary_initialisers.csv", index=False,
                     lineterminator="\n")
    stop_states.to_csv(args.outdir / "formal_released_stop_states.csv", index=False,
                       lineterminator="\n")
    per_world_monotonicity(world).to_csv(
        args.outdir / "formal_per_world_monotonicity.csv", index=False,
        lineterminator="\n")
    instability.to_csv(args.outdir / "formal_instability_by_anchor.csv", index=False,
                       lineterminator="\n")
    make_figure(summary, args.figure_dir / "stage_f_primary_metrics.pdf")

    print(json.dumps({
        "origin_rows": len(origin),
        "world_rows": len(world),
        "contrast_rows": len(contrasts),
        "all_registered_paired_n_at_least_10": bool(contrasts.n_paired.ge(10).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
