#!/usr/bin/env python3
"""
Stage E-lite runner: post-hoc held-out r-selection re-analysis.

Reads ONLY the 20 frozen origin-level CSVs of thesis_main_completed_20260816
(see scripts/sources.py for the manifest and its naming irregularity). Executes
no simulation, no factor adjustment, no NIRVAR, no GMM and no forecasting.

Research question (../DECISIONS.md E-01):
  Within the released FNIRVAR pipeline, where r is imposed and d_hat(r) is
  chosen by the released correlation-based MP rule, can past held-out
  forecasting MSPE select an r that also preserves downstream community
  structure?

USAGE: python3 run_stage_e_lite.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sources as S           # noqa: E402
import tdist                  # noqa: E402

ROOT = HERE.parent
OUTPUT_ROOT = Path(os.environ.get("FNIRVAR_STAGE_E_OUTDIR", ROOT)).resolve()
CFG = json.loads((ROOT / "configs" / "stage_e_lite_config.json").read_text())

R_TRUE = CFG["r_true"]
K_TRUE = CFG["K_true"]
N_TUNE = CFG["n_tuning_origins"]
EPSILONS = CFG["near_tie_epsilon"]


def sha256_file(p) -> str:
    return S.sha256(p)


def reconcile_with_frozen_summary(d: pd.DataFrame) -> dict:
    """Recompute the full-origin within-replication summary and compare."""
    w = pd.read_csv(S.WITHIN_SUMMARY)
    g = (d.groupby(["replication", "r_used", "branch"], as_index=False)
           .agg(num=("squared_error_sum", "sum"),
                den=("squared_error_denominator", "sum"),
                mean_ARI=("ARI", "mean"),
                mean_d_hat=("d_hat_package", "mean"),
                pr_d_eq_K=("d_hat_package", lambda s: float((s == K_TRUE).mean())),
                n=("ARI", "size")))
    g["MSPE"] = g.num / g.den
    m = g.merge(w, on=["replication", "r_used", "branch"], how="inner",
                suffixes=("_recomputed", "_frozen"))
    out = {
        "rows_matched": int(len(m)),
        "rows_expected": 360,
        "max_abs_diff_MSPE": float((m.MSPE - m.MSPE_complete).abs().max()),
        "max_abs_diff_mean_ARI": float((m.mean_ARI - m.mean_ARI_conditional).abs().max()),
        "max_abs_diff_mean_d_hat": float((m.mean_d_hat_recomputed - m.mean_d_hat_frozen).abs().max()),
        "max_abs_diff_Pr_d_eq_K": float((m.pr_d_eq_K - m.Pr_d_hat_equals_K_true_within).abs().max()),
        "max_abs_diff_numerator": float((m.num - m.MSPE_complete_numerator).abs().max()),
        "max_abs_diff_denominator": float((m.den - m.MSPE_complete_denominator).abs().max()),
    }
    out["reconciles"] = bool(
        out["rows_matched"] == 360
        and out["max_abs_diff_MSPE"] < 1e-9
        and out["max_abs_diff_mean_ARI"] < 1e-9
        and out["max_abs_diff_mean_d_hat"] < 1e-9
        and out["max_abs_diff_Pr_d_eq_K"] < 1e-12
        and out["max_abs_diff_numerator"] < 1e-6
        and out["max_abs_diff_denominator"] < 1e-9)
    return out


def main():
    t0 = time.perf_counter()
    (OUTPUT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "validation").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    st = tdist.self_test()
    if not st["all_pass"]:
        raise AssertionError("tdist self-test failed")

    manifest = S.assert_manifest()
    d = S.load_sources()

    # ---- origin split -----------------------------------------------------
    origins = sorted(d.forecast_origin_index.unique().tolist())
    if len(origins) != S.N_ORIGINS:
        raise AssertionError("origin count mismatch")
    tune_set, eval_set = set(origins[:N_TUNE]), set(origins[N_TUNE:])
    if tune_set & eval_set:
        raise AssertionError("tuning/evaluation origin sets overlap")
    # the split must be identical in every world x branch x r cell
    for key, g in d.groupby(["replication", "branch", "r_used"]):
        if sorted(g.forecast_origin_index.unique().tolist()) != origins:
            raise AssertionError(f"origin set differs in cell {key}")
    d = d.assign(split=np.where(d.forecast_origin_index.isin(tune_set), "tune", "eval"))

    recon = reconcile_with_frozen_summary(d)
    if not recon["reconciles"]:
        raise AssertionError(f"frozen-summary reconciliation failed: {recon}")

    # ---- cell-level split metrics ----------------------------------------
    cell = (d.groupby(["replication", "branch", "r_used", "split"], as_index=False)
              .agg(sse=("squared_error_sum", "sum"),
                   den=("squared_error_denominator", "sum"),
                   ARI=("ARI", "mean"),
                   d_hat=("d_hat_package", "mean"),
                   pr_d_hat_eq_K_true=("d_hat_package", lambda s: float((s == K_TRUE).mean())),
                   n_origins=("ARI", "size")))
    cell["MSPE"] = cell.sse / cell.den          # ratio of sums, never mean of ratios
    cell.to_csv(OUTPUT_ROOT / "results" / "cell_split_metrics.csv", index=False)

    wide = cell.pivot_table(index=["replication", "branch", "r_used"], columns="split",
                            values=["MSPE", "ARI", "d_hat", "pr_d_hat_eq_K_true",
                                    "n_origins"]).reset_index()
    wide.columns = ["replication", "branch", "r_used",
                    "ARI_eval", "ARI_tune", "MSPE_eval", "MSPE_tune",
                    "d_hat_eval", "d_hat_tune", "n_eval", "n_tune",
                    "prdK_eval", "prdK_tune"]

    # ---- selection, oracles, regrets, near-tie sets -----------------------
    sel_rows, tie_rows = [], []
    for (rep, br), s in wide.groupby(["replication", "branch"]):
        s = s.sort_values("r_used").reset_index(drop=True)
        if len(s) != len(CFG["r_grid"]):
            raise AssertionError(f"incomplete r grid for world {rep} branch {br}")
        get = lambda r, c: float(s.loc[s.r_used == r, c].iloc[0])  # noqa: E731

        tmin = float(s.MSPE_tune.min())
        cand = s.loc[np.isclose(s.MSPE_tune, tmin, rtol=0, atol=1e-12), "r_used"]
        r_sel = int(cand.min())                       # deterministic: smallest r
        tie = bool(len(cand) > 1)

        emin = float(s.MSPE_eval.min())
        r_po = int(s.loc[np.isclose(s.MSPE_eval, emin, rtol=0, atol=1e-12), "r_used"].min())
        po_tie = int((np.isclose(s.MSPE_eval, emin, rtol=0, atol=1e-12)).sum())
        amax = float(s.ARI_eval.max())
        r_so = int(s.loc[np.isclose(s.ARI_eval, amax, rtol=0, atol=1e-12), "r_used"].min())
        so_tie = int((np.isclose(s.ARI_eval, amax, rtol=0, atol=1e-12)).sum())

        row = {
            "replication": rep, "branch": br,
            "r_selected": r_sel, "tuning_min_tied": tie,
            "r_prediction_oracle": r_po, "n_prediction_oracle_ties": po_tie,
            "r_structure_oracle": r_so, "n_structure_oracle_ties": so_tie,
            "MSPE_eval_selected": get(r_sel, "MSPE_eval"),
            "MSPE_eval_prediction_oracle": emin,
            "MSPE_eval_r_true": get(R_TRUE, "MSPE_eval"),
            "ARI_eval_selected": get(r_sel, "ARI_eval"),
            "ARI_eval_structure_oracle": amax,
            "ARI_eval_r_true": get(R_TRUE, "ARI_eval"),
            "d_hat_eval_selected": get(r_sel, "d_hat_eval"),
            "pr_d_hat_eq_K_true_eval_selected": get(r_sel, "prdK_eval"),
            "abs_r_selected_minus_r_true": abs(r_sel - R_TRUE),
            "selected_equals_r_true": int(r_sel == R_TRUE),
            "selected_within_one_of_r_true": int(abs(r_sel - R_TRUE) <= 1),
            "selected_equals_prediction_oracle": int(r_sel == r_po),
            "selected_equals_structure_oracle": int(r_sel == r_so),
        }
        row["predictive_regret"] = row["MSPE_eval_selected"] - row["MSPE_eval_prediction_oracle"]
        row["structural_regret"] = row["ARI_eval_structure_oracle"] - row["ARI_eval_selected"]
        row["structural_regret_fraction"] = (
            row["structural_regret"] / row["ARI_eval_structure_oracle"]
            if row["ARI_eval_structure_oracle"] > 0 else np.nan)
        row["predictive_regret_fraction"] = (
            row["predictive_regret"] / row["MSPE_eval_prediction_oracle"])
        row["MSPE_eval_selected_minus_r_true"] = row["MSPE_eval_selected"] - row["MSPE_eval_r_true"]
        row["ARI_eval_r_true_minus_selected"] = row["ARI_eval_r_true"] - row["ARI_eval_selected"]
        # within-world Spearman across the nine r values (descriptive)
        row["spearman_tuneMSPE_evalMSPE"] = float(
            s[["MSPE_tune", "MSPE_eval"]].corr(method="spearman").iloc[0, 1])
        row["spearman_tuneMSPE_evalARI"] = float(
            s[["MSPE_tune", "ARI_eval"]].corr(method="spearman").iloc[0, 1])
        sel_rows.append(row)

        for eps in EPSILONS:                       # near-tie set from TUNING only
            m = s.MSPE_tune <= (1.0 + eps) * tmin
            sub = s[m]
            tie_rows.append({
                "replication": rep, "branch": br, "epsilon": eps,
                "n_in_set": int(m.sum()),
                "r_min": int(sub.r_used.min()), "r_max": int(sub.r_used.max()),
                "r_values": "|".join(str(int(v)) for v in sub.r_used),
                "MSPE_eval_min": float(sub.MSPE_eval.min()),
                "MSPE_eval_max": float(sub.MSPE_eval.max()),
                "MSPE_eval_range": float(sub.MSPE_eval.max() - sub.MSPE_eval.min()),
                "ARI_eval_min": float(sub.ARI_eval.min()),
                "ARI_eval_max": float(sub.ARI_eval.max()),
                "ARI_eval_range": float(sub.ARI_eval.max() - sub.ARI_eval.min()),
                "r_true_in_set": int(R_TRUE in set(sub.r_used)),
                "structure_oracle_in_set": int(r_so in set(sub.r_used)),
            })

    sel = pd.DataFrame(sel_rows).sort_values(["branch", "replication"])
    sel.to_csv(OUTPUT_ROOT / "results" / "world_branch_selections.csv", index=False)
    pd.DataFrame(tie_rows).sort_values(["branch", "epsilon", "replication"]).to_csv(
        OUTPUT_ROOT / "results" / "near_tie_analysis.csv", index=False)

    man = {
        "experiment": CFG["experiment"],
        "executed_simulation_code": False,
        "tdist_self_test": st,
        "source_manifest": manifest,
        "frozen_summary_reconciliation": recon,
        "n_source_rows": int(len(d)),
        "origin_split": {"n_tuning": len(tune_set), "n_evaluation": len(eval_set),
                         "tuning_origin_min": min(tune_set), "tuning_origin_max": max(tune_set),
                         "evaluation_origin_min": min(eval_set),
                         "evaluation_origin_max": max(eval_set),
                         "disjoint": True},
        "config_sha256": sha256_file(ROOT / "configs" / "stage_e_lite_config.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "sources_module_sha256": sha256_file(HERE / "sources.py"),
        "tdist_module_sha256": sha256_file(HERE / "tdist.py"),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (OUTPUT_ROOT / "validation" / "run_manifest.json").write_text(json.dumps(man, indent=2))

    lines = [f"{v['sha256']}  {v['path']}" for _, v in sorted(manifest["sources"].items(),
                                                              key=lambda kv: int(kv[0]))]
    (OUTPUT_ROOT / "SOURCE_HASHES.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: man[k] for k in
                      ["executed_simulation_code", "n_source_rows", "origin_split",
                       "frozen_summary_reconciliation", "elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
