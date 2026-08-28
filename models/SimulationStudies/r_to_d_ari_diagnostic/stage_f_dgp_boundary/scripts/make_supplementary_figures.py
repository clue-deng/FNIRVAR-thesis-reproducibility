"""
Stage F supplementary figures (additive; does not modify any frozen output).

Reads results/formal_anchor_summary.csv and writes:
  1. stage_f_structure_and_feasibility.pdf  - ARI panel + the three co-primary panels
  2. stage_f_eigengap_vs_feasibility.pdf    - achieved normalised eigengap vs feasibility

Style follows scripts/summarize_stage_f.py:make_figure so the new figures read as
siblings of figures/stage_f_primary_metrics.pdf.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANCHOR_ORDER = ["S0", "S1", "S2", "S3", "S4",
                "loading_scale_negative_gap", "loading_scale_positive_gap"]
LABELS = ["S0", "S1", "S2", "S3", "S4", "scale 0.20", "scale 0.60"]
BLUE, ORANGE = "#2455a4", "#c2621a"
GAP = "delta_over_lambda_max_Gamma_xi"


def _pick(df, metric):
    return df[df.metric.eq(metric)].set_index("anchor").loc[ANCHOR_ORDER]


def _asym(sub):
    return np.vstack([sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]])


def fig_structure_and_feasibility(summary, out):
    panels = [("ARI_at_r_true",         "Community recovery at true $r$"),
              ("fixed_point_indicator", "True $r$ is a fixed point"),
              ("basin_fraction",        "Basin fraction"),
              ("end_to_end_success",    "Released IC$_{p2}$ success")]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6), sharex=True)
    for ax, (metric, title) in zip(axes, panels):
        sub = _pick(summary, metric)
        x = np.arange(len(sub))
        ax.errorbar(x, sub["mean"], yerr=_asym(sub),
                    fmt="o", capsize=3, color=BLUE)
        ax.axvline(4.5, color="0.75", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xticks(x, LABELS, rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("World-level mean (95% t interval)")
    fig.suptitle("Stage F: structural signal and feedback feasibility across registered DGP anchors",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_eigengap(summary, out):
    gap = _pick(summary, GAP)
    fam_S = np.array([a.startswith("S") for a in ANCHOR_ORDER])
    panels = [("fixed_point_indicator", "True $r$ is a fixed point"),
              ("basin_fraction",        "Basin fraction")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        sub = _pick(summary, metric)
        for mask, colour, marker, name in [(fam_S, BLUE, "o", "Family S (separation)"),
                                           (~fam_S, ORANGE, "^", "Family E (loading scale)")]:
            ax.errorbar(gap["mean"][mask], sub["mean"][mask],
                        yerr=_asym(sub)[:, mask],
                        xerr=np.vstack([gap["mean"][mask] - gap["ci_low"][mask],
                                        gap["ci_high"][mask] - gap["mean"][mask]]),
                        fmt=marker, capsize=2, color=colour, markersize=7,
                        elinewidth=0.9, label=name)
        for xi, yi, lab, is_S in zip(gap["mean"], sub["mean"], LABELS, fam_S):
            # Family-E labels sit below-right so they do not cross the Delta=0 rule.
            ax.annotate(lab, (xi, yi), textcoords="offset points",
                        xytext=(7, 5) if is_S else (7, -14),
                        fontsize=8, color="0.25")
        ax.axvline(0.0, color="0.55", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Achieved $\Delta/\lambda_{\max}(\Gamma_\xi)$")
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.08, 1.15)
    axes[0].set_ylabel("World-level mean (95% t interval)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Stage F: achieved eigengap does not order feedback feasibility across families",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=Path("results/formal_anchor_summary.csv"))
    p.add_argument("--figure-dir", type=Path, required=True)
    a = p.parse_args()
    a.figure_dir.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(a.summary)
    fig_structure_and_feasibility(s, a.figure_dir / "stage_f_structure_and_feasibility.pdf")
    fig_eigengap(s, a.figure_dir / "stage_f_eigengap_vs_feasibility.pdf")
    print("wrote 2 figures to", a.figure_dir)
