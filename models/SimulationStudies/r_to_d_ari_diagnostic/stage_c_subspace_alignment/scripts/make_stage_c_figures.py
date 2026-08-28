#!/usr/bin/env python3
"""Two summary figures for the Stage C report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def fig_paired_contrasts():
    c = pd.read_csv(RESULTS / "paired_contrasts.csv")
    a = pd.read_csv(RESULTS / "across_replication_summary.csv")
    rows = [
        ("H-under limb 1:\nU_MP factor-unique purity\n(r=3 minus r=5)",
         c[(c.metric == "UMP_Q_F_unique_purity") & (c.contrast.str.contains("H-under"))].iloc[0]),
        ("H-under limb 2:\nU_4 capture of Q_C\n(r=3 minus r=5)",
         c[(c.metric == "U4_Q_C_capture") & (c.contrast.str.contains("H-under"))].iloc[0]),
        ("H-over limb (b):\nU_4 capture of Q_C\n(r=7 minus r=5)",
         c[(c.metric == "U4_Q_C_capture") & (c.contrast.str.contains("H-over"))].iloc[0]),
    ]
    qextra7 = a[(a.series == "matched") & (a.metric == "Qextra7_Q_C_unique_excess_purity")].iloc[0]
    labels = [r[0] for r in rows] + ["H-over limb (a):\nQ_extra_7 community-unique\nexcess purity (level)"]
    means = [r[1]["mean_delta"] for r in rows] + [qextra7["mean"]]
    los = [r[1]["ci95_low"] for r in rows] + [qextra7["ci95_low"]]
    his = [r[1]["ci95_high"] for r in rows] + [qextra7["ci95_high"]]
    errs_lo = [m - lo for m, lo in zip(means, los)]
    errs_hi = [hi - m for hi, m in zip(his, means)]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(labels))
    colors = ["#c0392b" if m < 0 else "#2471a3" for m in means]
    ax.barh(y, means, xerr=[errs_lo, errs_hi], color=colors, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Paired replication-level mean (95% t interval)")
    ax.set_title("Stage C pre-specified hypothesis-test metrics (n=20 structural replications)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure_stage_c_paired_contrasts.png", dpi=150)
    fig.savefig(FIGDIR / "figure_stage_c_paired_contrasts.pdf")
    plt.close(fig)


def fig_eigenvector_projection_profile():
    e = pd.read_csv(RESULTS / "eigenvector_level_alignment.csv")
    agg = e.groupby(["r_used", "eig_rank"])[["sq_proj_Q_F", "sq_proj_Q_C", "sq_proj_Q_P4"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, r_used in zip(axes, [3, 5, 7]):
        sub = agg[agg.r_used == r_used]
        ax.plot(sub.eig_rank, sub.sq_proj_Q_F, marker="o", label="Q_F (factor)")
        ax.plot(sub.eig_rank, sub.sq_proj_Q_C, marker="s", label="Q_C (community-contrast)")
        ax.plot(sub.eig_rank, sub.sq_proj_Q_P4, marker="^", label="Q_P4 (population)")
        ax.set_title(f"r_used={r_used}")
        ax.set_xlabel("correlation eigenvector rank (1-8)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean squared projection\n(averaged over 20 reps x 25 origins)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Stage C: where do the leading residual-correlation eigenvectors point?")
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure_stage_c_eigenvector_projection_profile.png", dpi=150)
    fig.savefig(FIGDIR / "figure_stage_c_eigenvector_projection_profile.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_paired_contrasts()
    fig_eigenvector_projection_profile()
    print("wrote figures to", FIGDIR)
