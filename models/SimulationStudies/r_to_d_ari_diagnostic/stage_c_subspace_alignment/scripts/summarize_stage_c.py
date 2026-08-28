#!/usr/bin/env python3
"""
Stage C Monte Carlo summaries: within-replication (25-origin) means, then
across-replication (20 structural reps) summaries and paired r-vs-r contrasts.

NaN-aware throughout: a metric undefined in a degenerate cell (e.g. U_MP purity
when d_hat=0) is NaN, never silently replaced with zero. Uncertainty is based on
structural replications, never on origins/eigenvectors/cells treated as
independent Monte Carlo draws.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRIC_SUFFIXES = (
    "_dim_U", "_dim_Q", "_shared_energy", "_purity", "_capture",
    "_largest_canonical_correlation", "_smallest_principal_angle",
)
ID_COLS_CELL = {
    "run_id", "replication", "structural_seed", "network_seed", "factor_seed",
    "gmm_seed", "r_used", "forecast_origin_index", "target_index", "T_window",
}
ID_COLS_MATCHED = {
    "run_id", "replication", "structural_seed", "forecast_origin_index", "target_index",
}


def metric_columns(df: pd.DataFrame, id_cols: set) -> list:
    cols = []
    for c in df.columns:
        if c in id_cols:
            continue
        if any(c.endswith(s) for s in METRIC_SUFFIXES):
            cols.append(c)
    return cols


def t_ci95(mean, sd, n):
    if n is None or n < 2 or np.isnan(sd):
        return (np.nan, np.nan)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return (mean - tcrit * se, mean + tcrit * se)


def within_replication_means(df: pd.DataFrame, group_cols: list, metric_cols: list, series_label: str) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for keys, g in df.groupby(group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(group_cols, keys))
            base["series"] = series_label
            for m in metric_cols:
                vals = g[m].to_numpy(dtype=float)
                n_total = len(vals)
                valid = vals[~np.isnan(vals)]
                n_valid = len(valid)
                mean = float(np.mean(valid)) if n_valid > 0 else np.nan
                rows.append({
                    **base, "metric": m, "within_rep_mean": mean,
                    "n_valid_origins": n_valid, "n_total_origins": n_total,
                    "valid_fraction": n_valid / n_total if n_total else np.nan,
                })
    return pd.DataFrame(rows)


def across_replication_summary(within_df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for keys, g in within_df.groupby(group_cols + ["metric"]):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(group_cols + ["metric"], keys))
            vals = g["within_rep_mean"].to_numpy(dtype=float)
            n_reps_with_any_valid_origin = int((g["n_valid_origins"] > 0).sum())
            valid = vals[~np.isnan(vals)]
            n = len(valid)
            mean = float(np.mean(valid)) if n > 0 else np.nan
            sd = float(np.std(valid, ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan)
            mcse = sd / np.sqrt(n) if n > 0 else np.nan
            lo, hi = t_ci95(mean, sd, n)
            rows.append({
                **base,
                "n_replications_planned": len(g),
                "n_replications_valid": n,
                "coverage_fraction": n / len(g) if len(g) else np.nan,
                "mean": mean, "sd": sd, "mcse": mcse,
                "ci95_low": lo, "ci95_high": hi,
                "n_replications_any_valid_origin": n_reps_with_any_valid_origin,
            })
    return pd.DataFrame(rows)


def paired_contrast(within_df: pd.DataFrame, id_col_extra: list, r_a: int, r_b: int, label: str) -> pd.DataFrame:
    """delta_i(metric) = within_rep_mean(r_a) - within_rep_mean(r_b) for each replication
    with both terms observed; paired t-test against zero."""
    rows = []
    for metric, g in within_df.groupby("metric"):
        a = g[g.r_used == r_a].set_index("replication")["within_rep_mean"]
        b = g[g.r_used == r_b].set_index("replication")["within_rep_mean"]
        common = a.index.intersection(b.index)
        delta = (a.loc[common] - b.loc[common]).dropna()
        n = len(delta)
        if n > 0:
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1)) if n > 1 else 0.0
            mcse = sd / np.sqrt(n) if n > 0 else np.nan
            lo, hi = t_ci95(mean, sd, n)
            if n > 1 and sd > 0:
                tstat = mean / (sd / np.sqrt(n))
                pval = float(2 * (1 - stats.t.cdf(abs(tstat), df=n - 1)))
            else:
                tstat, pval = np.nan, np.nan
        else:
            mean = sd = mcse = lo = hi = tstat = pval = np.nan
        rows.append({
            "contrast": label, "r_a": r_a, "r_b": r_b, "metric": metric,
            "n_paired": n, "mean_delta": mean, "sd_delta": sd, "mcse_delta": mcse,
            "t_stat": tstat, "p_value": pval, "ci95_low": lo, "ci95_high": hi,
            "deltas_by_replication": ";".join(f"{i}:{v:.10g}" for i, v in delta.items()),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cell_df = pd.read_csv(args.run_dir / "cell_level_alignment.csv")
    matched_df = pd.read_csv(args.run_dir / "matched_incremental_space_alignment.csv")

    cell_metrics = metric_columns(cell_df, ID_COLS_CELL)
    matched_metrics = metric_columns(matched_df, ID_COLS_MATCHED)

    within_cell = within_replication_means(cell_df, ["replication", "r_used"], cell_metrics, "cell")
    within_matched = within_replication_means(matched_df, ["replication"], matched_metrics, "matched")
    within_matched["r_used"] = np.nan
    within_all = pd.concat([within_cell, within_matched], ignore_index=True, sort=False)
    within_all.to_csv(args.out_dir / "within_replication_summary.csv", index=False)

    across_cell = across_replication_summary(within_cell, ["r_used"])
    across_cell["series"] = "cell"
    across_matched = across_replication_summary(within_matched, [])
    across_matched["series"] = "matched"
    across_matched["r_used"] = np.nan
    across_all = pd.concat([across_cell, across_matched], ignore_index=True, sort=False)
    across_all.to_csv(args.out_dir / "across_replication_summary.csv", index=False)

    contrasts = pd.concat([
        paired_contrast(within_cell, [], 3, 5, "H-under: r=3 minus r=5"),
        paired_contrast(within_cell, [], 7, 5, "H-over: r=7 minus r=5"),
    ], ignore_index=True)
    contrasts.to_csv(args.out_dir / "paired_contrasts.csv", index=False)

    print(f"within_replication_summary.csv: {len(within_all)} rows")
    print(f"across_replication_summary.csv: {len(across_all)} rows")
    print(f"paired_contrasts.csv: {len(contrasts)} rows")


if __name__ == "__main__":
    main()
