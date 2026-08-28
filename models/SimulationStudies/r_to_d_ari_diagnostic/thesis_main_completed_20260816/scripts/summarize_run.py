#!/usr/bin/env python3
"""Create within-replication and across-replication summaries."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def within_summary(df):
    rows = []
    keys = ["run_id", "replication", "structural_seed", "r_used", "branch"]
    for values, g in df.groupby(keys, dropna=False):
        ok = g["origin_status"] == "ok"
        n_total = len(g)
        n_ok = int(ok.sum())
        sq = pd.to_numeric(g.loc[ok, "squared_error_sum"], errors="coerce")
        denom = pd.to_numeric(g.loc[ok, "squared_error_denominator"], errors="coerce")
        ari = pd.to_numeric(g.loc[ok, "ARI"], errors="coerce")
        dh = pd.to_numeric(g["d_hat_package"], errors="coerce")
        complete = n_ok == n_total
        rows.append({
            **dict(zip(keys, values)),
            "n_origins_total": n_total,
            "n_origins_successful": n_ok,
            "n_origins_failed": n_total - n_ok,
            "failure_rate": (n_total - n_ok) / n_total,
            "success_coverage": n_ok / n_total,
            "MSPE_complete_numerator": float(sq.sum()) if complete and n_ok else np.nan,
            "MSPE_complete_denominator": float(denom.sum()) if complete and n_ok else np.nan,
            "MSPE_complete": float(sq.sum() / denom.sum()) if complete and n_ok else np.nan,
            "MSPE_conditional_numerator": float(sq.sum()) if n_ok else np.nan,
            "MSPE_conditional_denominator": float(denom.sum()) if n_ok else np.nan,
            "MSPE_conditional": float(sq.sum() / denom.sum()) if n_ok else np.nan,
            "ARI_coverage": int(ari.notna().sum()) / n_total,
            "mean_ARI_conditional": float(ari.mean()) if n_ok else np.nan,
            "mean_d_hat": float(dh.mean()) if dh.notna().any() else np.nan,
            "sd_d_hat_within": float(dh.std(ddof=1)) if dh.notna().sum() > 1 else np.nan,
            "Pr_d_hat_equals_K_true_within": (
                float((dh.dropna() == 4).mean()) if dh.notna().any() else np.nan
            ),
            "actual_spectral_radius": float(g["actual_spectral_radius"].iloc[0]),
        })
    return pd.DataFrame(rows)


def mean_ci(x):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    n = len(x)
    if not n:
        return (0, np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    mcse = sd / np.sqrt(n) if n > 1 else np.nan
    half = float(student_t.ppf(0.975, n - 1) * mcse) if n > 1 else np.nan
    return n, mean, sd, mcse, half


def across_summary(within):
    metrics = [
        "failure_rate", "success_coverage", "MSPE_complete", "MSPE_conditional",
        "ARI_coverage",
        "mean_ARI_conditional", "mean_d_hat", "Pr_d_hat_equals_K_true_within",
        "actual_spectral_radius",
    ]
    rows = []
    for (r, branch), g in within.groupby(["r_used", "branch"]):
        row = {"r_used": r, "branch": branch, "n_replications_planned_in_files": len(g)}
        for metric in metrics:
            n, mean, sd, mcse, half = mean_ci(g[metric])
            row[f"{metric}_n"] = n
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_mcse"] = mcse
            row[f"{metric}_ci95_low"] = mean - half if np.isfinite(half) else np.nan
            row[f"{metric}_ci95_high"] = mean + half if np.isfinite(half) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrasts(within):
    metrics = ["MSPE_complete", "MSPE_conditional", "mean_ARI_conditional", "mean_d_hat"]
    rows = []
    for branch, b in within.groupby("branch"):
        ref = b[b.r_used == 5].set_index("replication")
        for r, g in b.groupby("r_used"):
            cur = g.set_index("replication")
            common = cur.index.intersection(ref.index)
            for metric in metrics:
                diff = cur.loc[common, metric] - ref.loc[common, metric]
                n, mean, sd, mcse, half = mean_ci(diff)
                rows.append({
                    "branch": branch, "r_used": r, "reference_r": 5, "metric": metric,
                    "n_pairs": n, "mean_paired_difference": mean, "sd_paired_difference": sd,
                    "mcse": mcse,
                    "ci95_low": mean - half if np.isfinite(half) else np.nan,
                    "ci95_high": mean + half if np.isfinite(half) else np.nan,
                })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for pat in args.inputs for p in sorted(glob.glob(pat))]
    dfs = [pd.read_csv(p) for p in paths]
    if not dfs:
        raise SystemExit("no inputs")
    raw = pd.concat(dfs, ignore_index=True)
    within = within_summary(raw)
    across = across_summary(within)
    paired = paired_contrasts(within)
    within.to_csv(args.outdir / "within_replication_summary.csv", index=False)
    across.to_csv(args.outdir / "across_replication_summary.csv", index=False)
    paired.to_csv(args.outdir / "paired_contrasts.csv", index=False)
    validation = {
        "raw_rows": len(raw),
        "within_rows": len(within),
        "across_rows": len(across),
        "paired_rows": len(paired),
        "status_counts": raw.origin_status.value_counts(dropna=False).to_dict(),
    }
    (args.outdir / "summary_validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
