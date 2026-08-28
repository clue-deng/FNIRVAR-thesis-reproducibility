#!/usr/bin/env python3
"""
Stage E-lite summariser: world-level estimates and intervals.

The independent unit is the structural world (../DECISIONS.md E-06). The 499
forecast origins are serially dependent repeated measurements and are never
treated as independent samples; they are aggregated inside a world before any
interval is formed. Because this is a post-hoc re-analysis, the emphasis is on
estimates, intervals and effect sizes rather than p-values.

USAGE: python3 summarise_stage_e_lite.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tdist  # noqa: E402

ROOT = HERE.parent
OUTPUT_ROOT = Path(os.environ.get("FNIRVAR_STAGE_E_OUTDIR", ROOT)).resolve()
CFG = json.loads((ROOT / "configs" / "stage_e_lite_config.json").read_text())
R_TRUE = CFG["r_true"]


def mean_ci(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, mcse=np.nan, ci_low=np.nan, ci_high=np.nan)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    mcse = sd / np.sqrt(n) if n > 1 else 0.0
    half = tdist.t_ppf(0.975, n - 1) * mcse if (n > 1 and sd > 0) else 0.0
    return dict(n=n, mean=m, sd=sd, mcse=mcse, ci_low=m - half, ci_high=m + half)


def wilson(k: int, n: int, conf: float = 0.95) -> dict:
    """Wilson score interval for a proportion (worlds are the trials)."""
    if n == 0:
        return dict(k=0, n=0, p=np.nan, ci_low=np.nan, ci_high=np.nan)
    z = tdist.z_ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(k=int(k), n=int(n), p=float(p),
                ci_low=float(centre - half), ci_high=float(centre + half))


def main():
    sel = pd.read_csv(OUTPUT_ROOT / "results" / "world_branch_selections.csv")
    tie = pd.read_csv(OUTPUT_ROOT / "results" / "near_tie_analysis.csv")

    rows, dist_rows, tie_rows = [], [], []
    for br, s in sel.groupby("branch"):
        n = len(s)
        rec = {"branch": br, "n_worlds": n}
        for name, col in [
            ("predictive_regret", "predictive_regret"),
            ("predictive_regret_fraction", "predictive_regret_fraction"),
            ("structural_regret", "structural_regret"),
            ("structural_regret_fraction", "structural_regret_fraction"),
            ("ARI_eval_selected", "ARI_eval_selected"),
            ("ARI_eval_r_true", "ARI_eval_r_true"),
            ("ARI_eval_structure_oracle", "ARI_eval_structure_oracle"),
            ("MSPE_eval_selected", "MSPE_eval_selected"),
            ("MSPE_eval_r_true", "MSPE_eval_r_true"),
            ("MSPE_eval_prediction_oracle", "MSPE_eval_prediction_oracle"),
            ("ARI_eval_r_true_minus_selected", "ARI_eval_r_true_minus_selected"),
            ("MSPE_eval_selected_minus_r_true", "MSPE_eval_selected_minus_r_true"),
            ("abs_r_selected_minus_r_true", "abs_r_selected_minus_r_true"),
            ("spearman_tuneMSPE_evalMSPE", "spearman_tuneMSPE_evalMSPE"),
            ("spearman_tuneMSPE_evalARI", "spearman_tuneMSPE_evalARI"),
        ]:
            ci = mean_ci(s[col].values)
            for k, v in ci.items():
                if k != "n":
                    rec[f"{name}__{k}"] = v
        for name, col in [("selected_equals_r_true", "selected_equals_r_true"),
                          ("selected_within_one_of_r_true", "selected_within_one_of_r_true"),
                          ("selected_equals_prediction_oracle", "selected_equals_prediction_oracle"),
                          ("selected_equals_structure_oracle", "selected_equals_structure_oracle")]:
            w = wilson(int(s[col].sum()), n)
            rec[f"{name}__p"] = w["p"]
            rec[f"{name}__wilson_low"] = w["ci_low"]
            rec[f"{name}__wilson_high"] = w["ci_high"]
        rec["n_worlds_with_tuning_tie"] = int(s.tuning_min_tied.sum())
        rows.append(rec)

        for r, c in s.r_selected.value_counts().sort_index().items():
            dist_rows.append({"branch": br, "quantity": "r_selected", "r": int(r), "count": int(c)})
        for r, c in s.r_prediction_oracle.value_counts().sort_index().items():
            dist_rows.append({"branch": br, "quantity": "r_prediction_oracle", "r": int(r), "count": int(c)})
        for r, c in s.r_structure_oracle.value_counts().sort_index().items():
            dist_rows.append({"branch": br, "quantity": "r_structure_oracle", "r": int(r), "count": int(c)})

    for (br, eps), t in tie.groupby(["branch", "epsilon"]):
        rec = {"branch": br, "epsilon": eps, "n_worlds": len(t)}
        for name in ["n_in_set", "ARI_eval_range", "MSPE_eval_range"]:
            ci = mean_ci(t[name].values)
            rec[f"{name}__mean"] = ci["mean"]
            rec[f"{name}__ci_low"] = ci["ci_low"]
            rec[f"{name}__ci_high"] = ci["ci_high"]
        rec["r_min_overall"] = int(t.r_min.min())
        rec["r_max_overall"] = int(t.r_max.max())
        w = wilson(int(t.r_true_in_set.sum()), len(t))
        rec["pr_r_true_in_set"] = w["p"]
        rec["pr_r_true_in_set_wilson_low"] = w["ci_low"]
        rec["pr_r_true_in_set_wilson_high"] = w["ci_high"]
        w2 = wilson(int(t.structure_oracle_in_set.sum()), len(t))
        rec["pr_structure_oracle_in_set"] = w2["p"]
        tie_rows.append(rec)

    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "results" / "branch_summary.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUTPUT_ROOT / "results" / "selection_distributions.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(OUTPUT_ROOT / "results" / "near_tie_summary.csv", index=False)

    b = pd.DataFrame(rows).set_index("branch")
    for br in b.index:
        print(f"\n===== {br}  (n_worlds={int(b.loc[br,'n_worlds'])})")
        print("  Pr(r_sel = r_true)      %.2f  [%.2f, %.2f]" % (
            b.loc[br, "selected_equals_r_true__p"], b.loc[br, "selected_equals_r_true__wilson_low"],
            b.loc[br, "selected_equals_r_true__wilson_high"]))
        print("  predictive regret       %.5f [%.5f, %.5f]  (%.2f%% of oracle MSPE)" % (
            b.loc[br, "predictive_regret__mean"], b.loc[br, "predictive_regret__ci_low"],
            b.loc[br, "predictive_regret__ci_high"], 100 * b.loc[br, "predictive_regret_fraction__mean"]))
        print("  structural regret       %.4f  [%.4f, %.4f]  (%.1f%% of attainable ARI)" % (
            b.loc[br, "structural_regret__mean"], b.loc[br, "structural_regret__ci_low"],
            b.loc[br, "structural_regret__ci_high"], 100 * b.loc[br, "structural_regret_fraction__mean"]))
    print("\nnear-tie (tuning-defined) evaluation-ARI range:")
    print(pd.DataFrame(tie_rows)[["branch", "epsilon", "n_in_set__mean",
                                  "ARI_eval_range__mean", "ARI_eval_range__ci_low",
                                  "ARI_eval_range__ci_high", "pr_r_true_in_set"]].to_string(index=False))


if __name__ == "__main__":
    main()
