#!/usr/bin/env python3
"""Create thesis-ready figures from across-replication summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "primary_fixed_K": r"Primary: $K=K_{true}$",
    "robustness_K_equals_d_hat": r"Robustness: $K=\hat d$",
}
COLORS = {"primary_fixed_K": "#1f77b4", "robustness_K_equals_d_hat": "#d95f02"}


def draw(data, metric, ylabel, out, horizontal=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for branch in LABELS:
        g = data[data.branch == branch].sort_values("r_used")
        x = g.r_used.to_numpy()
        y = g[f"{metric}_mean"].to_numpy()
        lo = g[f"{metric}_ci95_low"].to_numpy()
        hi = g[f"{metric}_ci95_high"].to_numpy()
        ax.plot(x, y, marker="o", lw=2, color=COLORS[branch], label=LABELS[branch])
        ax.fill_between(x, lo, hi, color=COLORS[branch], alpha=0.16)
    ax.axvline(5, color="0.25", linestyle="--", lw=1.2, label=r"true $r=5$")
    if horizontal is not None:
        ax.axhline(horizontal, color="0.5", linestyle=":", lw=1.1)
    ax.set_xlabel(r"Imposed factor count $r_{used}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 10))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_propagation_panels(data, out):
    """Combine the branch-invariant dimension curve with branch-specific ARI."""
    ari_labels = {
        "primary_fixed_K": r"Fixed $K=K_{true}$",
        "robustness_K_equals_d_hat": r"End-to-end $K=\hat d$",
    }
    primary = data[data.branch == "primary_fixed_K"].sort_values("r_used")
    robustness = data[data.branch == "robustness_K_equals_d_hat"].sort_values("r_used")
    for column in ("mean_d_hat_mean", "mean_d_hat_ci95_low", "mean_d_hat_ci95_high"):
        if not np.allclose(primary[column], robustness[column]):
            raise ValueError(f"Expected branch-invariant MP summary in {column}")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharex=True)
    ax = axes[0]
    x = primary.r_used.to_numpy()
    y = primary.mean_d_hat_mean.to_numpy()
    lo = primary.mean_d_hat_ci95_low.to_numpy()
    hi = primary.mean_d_hat_ci95_high.to_numpy()
    ax.plot(x, y, marker="o", lw=2, color=COLORS["primary_fixed_K"],
            label=r"Mean selected $\hat d$")
    ax.fill_between(x, lo, hi, color=COLORS["primary_fixed_K"], alpha=0.16)
    ax.axhline(4, color="0.45", linestyle=":", lw=1.2, label=r"$K_{true}=4$")
    ax.axvline(5, color="0.25", linestyle="--", lw=1.2, label=r"$r_{true}=5$")
    ax.set_title(r"(a) Residual dimension")
    ax.set_ylabel(r"Mean MP-selected dimension $\hat d$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    for branch in LABELS:
        g = data[data.branch == branch].sort_values("r_used")
        x = g.r_used.to_numpy()
        y = g.mean_ARI_conditional_mean.to_numpy()
        lo = g.mean_ARI_conditional_ci95_low.to_numpy()
        hi = g.mean_ARI_conditional_ci95_high.to_numpy()
        ax.plot(x, y, marker="o", lw=2, color=COLORS[branch], label=ari_labels[branch])
        ax.fill_between(x, lo, hi, color=COLORS[branch], alpha=0.16)
    ax.axvline(5, color="0.25", linestyle="--", lw=1.2)
    ax.set_title(r"(b) Community recovery")
    ax.set_ylabel("Mean ARI")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in axes:
        ax.set_xlabel(r"Imposed factor count $r_{used}$")
        ax.set_xticks(range(1, 10))
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.summary)
    draw(d, "mean_d_hat", r"Mean MP-selected dimension $\hat d$", args.outdir / "figure_d_hat_vs_r", horizontal=4)
    draw(d, "mean_ARI_conditional", "Mean ARI", args.outdir / "figure_ari_vs_r")
    draw(d, "MSPE_complete", "Mean complete MSPE", args.outdir / "figure_mspe_vs_r")
    draw_propagation_panels(d, args.outdir / "figure_propagation_panels")


if __name__ == "__main__":
    main()
