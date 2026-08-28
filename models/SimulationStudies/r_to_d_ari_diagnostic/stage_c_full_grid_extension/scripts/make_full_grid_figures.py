#!/usr/bin/env python3
"""Six required figures for the Stage C full-grid extension. Reads only
frozen/derived CSVs under results/; writes only results/figures/. Figures are
selected by the required list in the execution prompt, not by which looks
cleanest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_full_grid as cfg  # noqa: E402

RESULTS = HERE.parent / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_context():
    """full-grid d_hat, fixed-K ARI, full-pipeline ARI, MSPE context (frozen formal)."""
    combined = pd.read_csv(RESULTS / "combined_mechanism_cell_level.csv")
    m = combined.groupby("r_used").agg(
        d_hat=("d_hat_package", "mean"),
        ARI_fixed_K=("formal_ARI", "mean"),
    ).reset_index()
    formal_rob = pd.read_csv(cfg.FORMAL_DIR / "formal_results_20260816" / "main_results_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    axes[0].plot(m.r_used, m.d_hat, "o-")
    axes[0].axhline(cfg.K_TRUE, color="gray", ls="--", lw=1, label="K_true=4")
    axes[0].set_title("mean d_hat vs r_used"); axes[0].set_xlabel("r_used"); axes[0].legend()
    axes[1].plot(formal_rob.r_used, formal_rob.ARI_fixed_K, "o-", label="fixed K=K_true")
    axes[1].plot(formal_rob.r_used, formal_rob.ARI_K_equals_d_hat, "s--", label="K=d_hat")
    axes[1].set_title("formal ARI vs r_used"); axes[1].set_xlabel("r_used"); axes[1].legend()
    axes[2].plot(formal_rob.r_used, formal_rob.MSPE_fixed_K, "o-", label="fixed K")
    axes[2].plot(formal_rob.r_used, formal_rob.MSPE_K_equals_d_hat, "s--", label="K=d_hat")
    axes[2].set_title("formal MSPE vs r_used"); axes[2].set_xlabel("r_used"); axes[2].legend()
    fig.suptitle("Full-grid context (read-only, from the completed formal run)")
    savefig(fig, "figure1_full_grid_context")


def fig2_under_grid():
    combined = pd.read_csv(RESULTS / "combined_mechanism_cell_level.csv")
    m = combined.groupby("r_used").agg(
        UMP_Q_F_unique_purity=("UMP_Q_F_unique_purity", "mean"),
        U4_Q_C_capture=("U4_Q_C_capture", "mean"),
    ).reset_index()
    under = m[m.r_used <= 5]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(under.r_used, under.UMP_Q_F_unique_purity, "o-", color="tab:red", label="U_MP factor-unique purity")
    ax1.set_xlabel("r_used"); ax1.set_ylabel("factor-unique purity", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(under.r_used, under.U4_Q_C_capture, "s--", color="tab:blue", label="U_4 capture of Q_C")
    ax2.set_ylabel("U_4 capture of Q_C", color="tab:blue")
    fig.suptitle("Under-grid (r<=5): factor purity vs community capture")
    savefig(fig, "figure2_under_grid_mechanism")


def fig3_over_grid():
    incr = pd.read_csv(RESULTS / "combined_incremental_space_long.csv")
    extra = incr[incr.space_type == "extra"]
    m1 = extra.groupby("r_used")["Qspace_Q_C_unique_excess_purity"].mean().reset_index()
    combined = pd.read_csv(RESULTS / "combined_mechanism_cell_level.csv")
    m2 = combined[combined.r_used >= 5].groupby("r_used")["U4_Q_C_capture"].mean().reset_index()
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(m1.r_used, m1.Qspace_Q_C_unique_excess_purity, "o-", color="tab:green",
              label="Q_extra_r community-unique excess purity")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("r_used"); ax1.set_ylabel("excess purity", color="tab:green")
    ax2 = ax1.twinx()
    ax2.plot(m2.r_used, m2.U4_Q_C_capture, "s--", color="tab:blue", label="U_4 capture of Q_C")
    ax2.set_ylabel("U_4 capture of Q_C", color="tab:blue")
    fig.suptitle("Over-grid (r>=5): incremental community capture vs residual capture")
    savefig(fig, "figure3_over_grid_mechanism")


def fig4_primary_observables():
    raw = pd.read_csv(RESULTS / "observable_full_grid_cell_level.csv")
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    for ax, metric in zip(axes, cfg.PRIMARY_OBSERVABLES):
        rep_r = raw.groupby(["replication", "r_used"])[metric].mean().reset_index()
        piv = rep_r.pivot(index="replication", columns="r_used", values=metric)
        for rep in piv.index:
            ax.plot(piv.columns, piv.loc[rep], color="gray", alpha=0.25, lw=0.8)
        mean = piv.mean(axis=0)
        sd = piv.std(axis=0, ddof=1)
        n = piv.shape[0]
        ci = 1.96 * sd / np.sqrt(n)
        ax.plot(mean.index, mean.values, "o-", color="tab:blue")
        ax.fill_between(mean.index, mean.values - ci, mean.values + ci, alpha=0.25, color="tab:blue")
        ax.set_title(metric, fontsize=9); ax.set_xlabel("r_used")
    fig.suptitle("Primary observables: individual replications (gray) + mean/95% interval (blue)")
    savefig(fig, "figure4_primary_observable_curves")


def fig5_mirror_contrasts():
    """Bug fix (2026-08-17 cleanup): the previous version computed per-bar
    Holm-significance colors but never passed them to barh(), so every bar
    rendered in matplotlib's default color regardless of significance. Also
    switched to one small-multiple panel per observable (metrics differ by
    orders of magnitude -- d_hat ~1-3 vs selected_excess_spectral_mass ~10-55
    -- so a single shared x-axis made the smaller-scale metrics unreadable;
    this changes layout only, not any plotted value)."""
    obs = pd.read_csv(RESULTS / "observable_feasibility_tests.csv")
    mirror = obs[obs.test_type == "mirror"]
    metrics = cfg.PRIMARY_OBSERVABLES
    pairs = ["r1_minus_r9", "r2_minus_r8", "r3_minus_r7", "r4_minus_r6"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(19, 3.5), sharey=False)
    y = np.arange(len(pairs))
    for ax, metric in zip(axes, metrics):
        sub = mirror[mirror.observable == metric].set_index("test")
        vals = [sub.loc[p, "mean"] if p in sub.index else np.nan for p in pairs]
        colors = ["tab:green" if (p in sub.index and sub.loc[p, "significant_holm"]) else "tab:red"
                  for p in pairs]
        ax.barh(y, vals, color=colors, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(pairs, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(metric, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:green"), plt.Rectangle((0, 0), 1, 1, color="tab:red")]
    fig.legend(handles, ["Holm-significant", "not Holm-significant"], loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Mirror-pair contrasts (r_lo - r_hi) by primary observable, colored by Holm significance")
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, "figure5_mirror_contrasts")


def fig6_mechanism_vs_observable_summary():
    """Bug fixes (2026-08-17 cleanup), numerical data unchanged:
    1. Title corrected: the oracle mechanism was tested and supported across
       the FULL r=1..9 grid in this pass (all 4 Holm families + all 4
       dose-response sequences), not only at r=3,7 (that was the prior,
       3-point Stage C pass this one extends).
    2. residual_variance_ratio -- the sole formally direction_capable
       observable -- is drawn with an amber, hatched bar (not unqualified
       green) and the historical-A-vs-operational-no-go distinction is shown
       directly in the title, per reports/OPERATIONAL_STAGE6_DECISION.md."""
    class_df = pd.read_csv(RESULTS / "observable_signal_classification.csv")
    gate = json.loads((RESULTS / "stage6_gate_decision.json").read_text())
    op = json.loads((RESULTS / "operational_stage6_decision.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {"direction_capable": "tab:green", "distance_only": "tab:orange",
              "not_a_usable_correction_signal": "tab:red"}
    y = np.arange(len(class_df))
    for i, row in class_df.iterrows():
        is_mechanical_caveat = (row.observable == op["qualifying_observable"]
                                 and row.classification == "direction_capable")
        if is_mechanical_caveat:
            ax.barh(i, 1, color="tab:orange", hatch="///", edgecolor="black", linewidth=0.6)
        else:
            ax.barh(i, 1, color=colors[row.classification])
        label = f"{row.observable} ({row.information_class})"
        if is_mechanical_caveat:
            label += "  [mechanical PCA-truncation artifact, not a genuine r-detector]"
        ax.text(0.02, i, label, va="center", fontsize=9)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title(
        "Observable feasibility classification -- oracle mechanism supported across the full\n"
        f"r=1,...,9 grid (all 4 Holm families + all 4 dose-response sequences, this pass)\n"
        f"historical formal screen: {gate['decision']} | operational decision: "
        f"{op['operational_decision']} (mechanical PCA variance trend)"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:green"),
               plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
               plt.Rectangle((0, 0), 1, 1, color="tab:orange", hatch="///", edgecolor="black"),
               plt.Rectangle((0, 0), 1, 1, color="tab:red")]
    labels = ["direction_capable (no caveat)", "distance_only",
              "direction_capable but mechanical (operational no-go)", "not a usable correction signal"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=7.5)
    savefig(fig, "figure6_mechanism_vs_observable_summary")


if __name__ == "__main__":
    fig1_context()
    fig2_under_grid()
    fig3_over_grid()
    fig4_primary_observables()
    fig5_mirror_contrasts()
    fig6_mechanism_vs_observable_summary()
    print("wrote 6 figures to", FIGDIR)
