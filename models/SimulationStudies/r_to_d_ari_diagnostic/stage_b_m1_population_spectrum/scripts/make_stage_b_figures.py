#!/usr/bin/env python3
"""Four required Stage B figures. Reads only results/*.csv; writes only
results/figures/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_b as cb  # noqa: E402

RESULTS = HERE.parent / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BRANCH_COLORS = {
    "iid_marginal": "tab:blue", "var_stationary_start": "tab:orange",
    "var_zero_start": "tab:green", "var_burnin_500": "tab:red",
    "released_replay_validation": "black",
}
BRANCH_LABELS = {
    "iid_marginal": "A: iid_marginal", "var_stationary_start": "B: var_stationary_start",
    "var_zero_start": "C: var_zero_start", "var_burnin_500": "D: var_burnin_500",
    "released_replay_validation": "E: released_replay",
}


def savefig(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_population_profile():
    pop = pd.read_csv(RESULTS / "population_spectra.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    theta_cols = [f"theta_{i}" for i in range(1, cb.LEADING_EIGS + 1)]
    for _, row in pop.iterrows():
        ax.plot(range(1, cb.LEADING_EIGS + 1), row[theta_cols].values, "o-", color="gray", alpha=0.4, lw=1)
    mean_theta = pop[theta_cols].mean(axis=0)
    ax.plot(range(1, cb.LEADING_EIGS + 1), mean_theta.values, "o-", color="tab:red", lw=2.5,
             label="mean across 20 worlds")
    ax.axvline(4.5, color="black", ls="--", lw=0.8, label="K_true=4 boundary")
    ax.set_xlabel(r"population correlation eigenvalue rank $j$")
    ax.set_ylabel(r"$\theta_j$")
    ax.set_title("Population leading-eigenvalue profile (Lyapunov-solved stationary correlation)")
    ax.legend(fontsize=8)
    savefig(fig, "figure1_population_eigenvalue_profile")


def fig2_mean_dhat_vs_T():
    across = pd.read_csv(RESULTS / "across_world_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for branch in cb.ALL_BRANCHES:
        sub = across[across.branch == branch].sort_values("T_window")
        if not len(sub):
            continue
        ax.errorbar(sub.T_window, sub.mean_d_hat_mean,
                     yerr=[sub.mean_d_hat_mean - sub.mean_d_hat_ci_low,
                           sub.mean_d_hat_ci_high - sub.mean_d_hat_mean],
                     marker="o", label=BRANCH_LABELS[branch], color=BRANCH_COLORS[branch], capsize=3)
    ax.axhline(cb.K_TRUE, color="gray", ls="--", lw=1, label="K_true=4")
    ax.set_xscale("log")
    ax.set_xlabel("T"); ax.set_ylabel("mean d_hat (across-world 95% CI)")
    ax.set_title("Mean d_hat vs T, by branch")
    ax.legend(fontsize=7.5, loc="upper left")
    savefig(fig, "figure2_mean_dhat_vs_T")


def fig3_pr_dhat_ge_j():
    across = pd.read_csv(RESULTS / "across_world_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, j in zip(axes, [3, 4]):
        metric = f"Pr_dhat_ge_{j}"
        for branch in cb.ALL_BRANCHES:
            sub = across[across.branch == branch].sort_values("T_window")
            if not len(sub):
                continue
            ax.errorbar(sub.T_window, sub[f"{metric}_mean"],
                         yerr=[sub[f"{metric}_mean"] - sub[f"{metric}_ci_low"],
                               sub[f"{metric}_ci_high"] - sub[f"{metric}_mean"]],
                         marker="o", label=BRANCH_LABELS[branch], color=BRANCH_COLORS[branch], capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("T"); ax.set_ylabel(f"Pr(d_hat >= {j})")
        ax.set_title(f"Pr(d_hat >= {j}) vs T, by branch")
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=7, loc="upper left")
    savefig(fig, "figure3_pr_dhat_ge_j_vs_T")


def fig4_paired_contrasts():
    c = pd.read_csv(RESULTS / "paired_contrasts.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, family, title in zip(
        axes, ["S", "I", "B"],
        ["Family S: serial dependence\n(VAR_stationary - IID)",
         "Family I: zero-start init.\n(VAR_zero - VAR_stationary)",
         "Family B: burn-in 500\n(VAR_burnin500 - VAR_stationary)"],
    ):
        sub = c[c.family == family].reset_index(drop=True)
        y = np.arange(len(sub))
        colors = ["tab:green" if s else "tab:gray" for s in sub["holm_significant"]]
        ax.barh(y, sub["mean"], xerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]],
                 color=colors, alpha=0.85, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels(sub["test"], fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:green"), plt.Rectangle((0, 0), 1, 1, color="tab:gray")]
    fig.legend(handles, ["Holm-significant", "not Holm-significant"], loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Paired structural-world contrasts (mean +/- 95% CI), 20 worlds")
    fig.subplots_adjust(bottom=0.22, wspace=0.4)
    savefig(fig, "figure4_paired_contrasts")


if __name__ == "__main__":
    fig1_population_profile()
    fig2_mean_dhat_vs_T()
    fig3_pr_dhat_ge_j()
    fig4_paired_contrasts()
    print("wrote 4 figures to", FIGDIR)
