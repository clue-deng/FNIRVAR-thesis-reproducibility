#!/usr/bin/env python3
"""
Stage E-lite figures.

Palette: categorical slots 1-3 of the validated default data-viz palette
(#2a78d6 / #eb6834 / #1baf7a), which passes the lightness, chroma, CVD and
normal-vision checks. No dual-axis chart is used anywhere: quantities on
different scales are shown as small multiples, never on two y-scales.
Every series is labelled as well as coloured.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tdist  # noqa: E402

ROOT = HERE.parent
OUTPUT_ROOT = Path(os.environ.get("FNIRVAR_STAGE_E_OUTDIR", ROOT)).resolve()
OUT = OUTPUT_ROOT / "results" / "figures"
CFG = json.loads((ROOT / "configs" / "stage_e_lite_config.json").read_text())
R_TRUE = CFG["r_true"]

BR = {"robustness_K_equals_d_hat": "deployable pipeline  ($K=\\hat d$)",
      "primary_fixed_K": "controlled diagnostic  ($K=K_{true}$)"}
ORDER = ["robustness_K_equals_d_hat", "primary_fixed_K"]
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.bbox": "tight"})


def ci(x):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); sd = x.std(ddof=1) if n > 1 else 0.0
    h = tdist.t_ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 and sd > 0 else 0.0
    return m, m - h, m + h


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_curves(cell):
    w = cell.pivot_table(index=["replication", "branch", "r_used"], columns="split",
                         values=["MSPE", "ARI"]).reset_index()
    w.columns = ["replication", "branch", "r_used", "ARI_eval", "ARI_tune",
                 "MSPE_eval", "MSPE_tune"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6), sharex=True)
    for j, br in enumerate(ORDER):
        s = w[w.branch == br]
        for i, (col, colour, lab) in enumerate(
                [("MSPE_tune", C1, "tuning MSPE"), ("ARI_eval", C3, "evaluation ARI")]):
            ax = axes[i][j]
            rs = sorted(s.r_used.unique())
            mu, lo, hi = zip(*[ci(s.loc[s.r_used == r, col].values) for r in rs])
            ax.fill_between(rs, lo, hi, color=colour, alpha=0.18, lw=0)
            ax.plot(rs, mu, color=colour, lw=2, marker="o", ms=4, mec="white", mew=1)
            ax.axvline(R_TRUE, color=GRID, lw=1, ls=(0, (2, 2)), zorder=0)
            tidy(ax)
            ax.set_xticks(rs)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(BR[br], color=INK)
            if i == 1:
                ax.set_xlabel("imposed factor count $r$")
    axes[0][0].annotate("$r_{true}$", (R_TRUE, axes[0][0].get_ylim()[1]),
                        color=INK2, fontsize=7.5, ha="center", va="top")
    fig.suptitle("Stage E-lite: what the practitioner sees (top) versus what they get "
                 "(bottom)\nmean over 20 structural worlds, 95% t bands",
                 fontsize=9, color=INK, y=1.06)
    fig.savefig(OUT / "fig_stage_e_lite_curves.pdf", format="pdf")
    plt.close(fig)


def fig_selection(sel):
    rs = list(range(1, 10))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True)
    for ax, br in zip(axes, ORDER):
        s = sel[sel.branch == br]
        for k, (col, colour, lab) in enumerate(
                [("r_selected", C1, "selected by tuning MSPE"),
                 ("r_structure_oracle", C3, "structure oracle (ARI)")]):
            cnt = [int((s[col] == r).sum()) for r in rs]
            ax.bar(np.array(rs) + (k - 0.5) * 0.38, cnt, width=0.36, color=colour,
                   label=lab, zorder=3)
        ax.axvline(R_TRUE, color=GRID, lw=1, ls=(0, (2, 2)), zorder=0)
        ax.set_xticks(rs)
        ax.set_xlabel("$r$")
        ax.set_title(BR[br], color=INK)
        tidy(ax)
    axes[0].set_ylabel("number of worlds")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Stage E-lite: which $r$ the held-out criterion picks, "
                 "against which $r$ maximises community recovery",
                 fontsize=9, color=INK, y=1.06)
    fig.savefig(OUT / "fig_stage_e_lite_selection.pdf", format="pdf")
    plt.close(fig)


def fig_regret(sel):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for br, colour, mk in [(ORDER[0], C1, "o"), (ORDER[1], C2, "s")]:
        s = sel[sel.branch == br]
        ax.scatter(100 * s.predictive_regret_fraction, s.structural_regret,
                   s=34, color=colour, marker=mk, edgecolor="white", linewidth=0.8,
                   label=BR[br], zorder=3)
    ax.set_xlabel("predictive regret  (% of oracle evaluation MSPE)")
    ax.set_ylabel("structural regret  (evaluation ARI)")
    tidy(ax)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    ax.set_title("One point per structural world", color=INK)
    fig.suptitle("Stage E-lite: choosing by prediction costs little prediction "
                 "and much structure", fontsize=9, color=INK, y=1.02)
    fig.savefig(OUT / "fig_stage_e_lite_regret.pdf", format="pdf")
    plt.close(fig)


def fig_near_tie(tie):
    eps = sorted(tie.epsilon.unique())
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for br, colour, mk in [(ORDER[0], C1, "o"), (ORDER[1], C2, "s")]:
        s = tie[tie.branch == br]
        mu, lo, hi = zip(*[ci(s.loc[s.epsilon == e, "ARI_eval_range"].values) for e in eps])
        ax.fill_between([100 * e for e in eps], lo, hi, color=colour, alpha=0.16, lw=0)
        ax.plot([100 * e for e in eps], mu, color=colour, lw=2, marker=mk, ms=5,
                mec="white", mew=1, label=BR[br], zorder=3)
    ax.set_xlabel("tuning-MSPE tolerance $\\varepsilon$  (%)")
    ax.set_ylabel("evaluation-ARI range inside $S_\\varepsilon$")
    tidy(ax)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Stage E-lite: models indistinguishable by past forecast error\n"
                 "differ widely in future community recovery",
                 fontsize=9, color=INK, y=1.04)
    fig.savefig(OUT / "fig_stage_e_lite_near_tie.pdf", format="pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cell = pd.read_csv(OUTPUT_ROOT / "results" / "cell_split_metrics.csv")
    sel = pd.read_csv(OUTPUT_ROOT / "results" / "world_branch_selections.csv")
    tie = pd.read_csv(OUTPUT_ROOT / "results" / "near_tie_analysis.csv")
    fig_curves(cell); fig_selection(sel); fig_regret(sel); fig_near_tie(tie)
    for f in sorted(OUT.glob("*.pdf")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
