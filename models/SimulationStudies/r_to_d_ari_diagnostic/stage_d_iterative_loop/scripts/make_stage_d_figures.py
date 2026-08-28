#!/usr/bin/env python3
"""
Stage D figures.

fig1 -- the transition map F(r) (identity line = fixed points), one panel per DGP.
fig2 -- basin heatmap: P(trajectory from r0 reaches r_true), sequential blue ramp.

Palette: categorical slots 1/2/3 and the blue sequential ramp from the validated
default (`dataviz/references/palette.md`); the three-slot categorical set passes
lightness, chroma, CVD and normal-vision checks. Series are direct-labelled as
well as coloured, so identity is never colour-alone.

USAGE: python3 make_stage_d_figures.py --results ../results --outdir ../results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SERIES = {"A_incremental": "#2a78d6", "B_absolute": "#eb6834", "C_criterion": "#1baf7a"}
LABEL = {"A_incremental": "A incremental", "B_absolute": "B absolute",
         "C_criterion": "C criterion"}
DGP_TITLE = {"strong_sbm": "strong SBM  ($p_{in}$=0.9, $p_{out}$=0.1)",
             "weak_sbm": "weak SBM  ($p_{in}$=0.6, $p_{out}$=0.4)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
})


def fig_transition(tmap: pd.DataFrame, out: Path, baing_branch="zero_fixed"):
    dgps = ["strong_sbm", "weak_sbm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, dgp in zip(axes, dgps):
        s = tmap[(tmap.dgp == dgp) & (tmap.K_branch == "primary_fixed_K")
                 & (tmap.baing_branch == baing_branch)]
        ax.plot([1, 9], [1, 9], color=GRID, lw=1.2, zorder=1)
        ax.text(3.2, 2.55, "$r'=r$", color=INK2, fontsize=7.5, ha="left", va="center")
        ax.axvline(5, color=GRID, lw=1, ls=(0, (2, 2)), zorder=0)
        for variant, colour in SERIES.items():
            g = s[s.variant == variant].sort_values("r_used")
            if not len(g):
                continue
            ax.plot(g.r_used, g.mean_r_next, color=colour, lw=2,
                    marker="o", ms=4.5, mec="white", mew=1.0, zorder=3,
                    label=LABEL[variant])
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("imposed factor count $r$")
        ax.set_title(DGP_TITLE[dgp], color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("next iterate  $r' = F(r)$")
    axes[0].set_yticks(range(1, 11))
    # direct labels at the right-hand end, in ink, colour carried by the mark
    s = tmap[(tmap.dgp == "weak_sbm") & (tmap.K_branch == "primary_fixed_K")
             & (tmap.baing_branch == baing_branch)]
    ends = []
    for variant in SERIES:
        g = s[s.variant == variant].sort_values("r_used")
        if len(g):
            ends.append([float(g.mean_r_next.iloc[-1]), LABEL[variant]])
    ends.sort()
    for i in range(1, len(ends)):          # stagger labels that would collide
        if ends[i][0] - ends[i - 1][0] < 0.45:
            ends[i][0] = ends[i - 1][0] + 0.45
    for y, lab in ends:
        axes[1].annotate(lab, (9.12, y), color=INK2, fontsize=7.5, va="center")
    axes[1].set_xlim(0.6, 11.6)
    axes[0].set_xlim(0.6, 9.4)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=7.5, loc="upper left",
                   labelcolor=INK2)
    fig.suptitle("Stage D: the Proposal §4.2 update map  "
                 "(mean over 20 worlds × 5 origins, $K=K_{true}$, zero-fixed Bai–Ng)",
                 fontsize=9, color=INK, y=1.04)
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig_basin(basin: pd.DataFrame, out: Path):
    variants = ["A_incremental", "B_absolute", "C_criterion"]
    branches = [("strong_sbm", "zero_fixed"), ("weak_sbm", "zero_fixed"),
                ("strong_sbm", "released")]
    titles = ["strong SBM, zero-fixed", "weak SBM, zero-fixed",
              "strong SBM, released (defective)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, (dgp, bb), title in zip(axes, branches, titles):
        M = np.full((len(variants), 9), np.nan)
        s = basin[(basin.dgp == dgp) & (basin.K_branch == "primary_fixed_K")
                  & (basin.baing_branch == bb)]
        for i, v in enumerate(variants):
            g = s[s.variant == v].set_index("r0")
            for j, r0 in enumerate(range(1, 10)):
                if r0 in g.index:
                    M[i, j] = g.loc[r0, "P_reach_r_true"]
        im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(9), [str(r) for r in range(1, 10)])
        ax.set_yticks(range(3), [LABEL[v] for v in variants])
        ax.set_xlabel("initial $r^{(0)}$")
        ax.set_title(title, color=INK)
        for i in range(3):
            for j in range(9):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.8,
                            color="white" if M[i, j] > 0.55 else INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("P(reach $r_{true}$)", color=INK2, fontsize=7.5)
    cb.outline.set_visible(False)
    fig.suptitle("Stage D: basin of attraction, $K=K_{true}$ "
                 "(cell values are also printed, so colour is never the only encoding)",
                 fontsize=9, color=INK, y=1.08)
    fig.savefig(out, format="pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("../results"))
    ap.add_argument("--outdir", type=Path, default=Path("../results/figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    tmap = pd.read_csv(args.results / "transition_map.csv")
    basin = pd.read_csv(args.results / "basin_summary.csv")
    fig_transition(tmap, args.outdir / "fig_stage_d_transition_map.pdf")
    fig_basin(basin, args.outdir / "fig_stage_d_basin.pdf")
    for f in sorted(args.outdir.glob("*.pdf")):
        print(f, f.stat().st_size)


if __name__ == "__main__":
    main()
