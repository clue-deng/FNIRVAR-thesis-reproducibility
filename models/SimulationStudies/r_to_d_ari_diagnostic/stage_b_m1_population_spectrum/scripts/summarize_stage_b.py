#!/usr/bin/env python3
"""
Stage B/M1 summarizer: within-world and across-world summaries, the four
pre-specified Holm-corrected paired-contrast families, the d_hat frequency
table, and the interpretation decision-tree classification.

Reads only the frozen formal run's raw CSVs; writes only into results/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_b as cb  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Within-world / across-world summaries
# ---------------------------------------------------------------------------
def build_within_world_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Average path-level outcomes within each (world, branch, T). Branch E
    has a single deterministic replay per (world, T) -- n=1, not averaged
    across paths, reported as its own row with n_paths=1."""
    rows = []
    grp_cols = ["world_index", "branch", "T_window"]
    for keys, g in sample_df.groupby(grp_cols):
        world_index, branch, T = keys
        row = {"world_index": world_index, "branch": branch, "T_window": T, "n_paths": len(g),
               "mean_d_hat": g["d_hat"].mean(), "sd_d_hat": g["d_hat"].std(ddof=1) if len(g) > 1 else float("nan")}
        for j in range(1, 7):
            row[f"Pr_dhat_ge_{j}"] = g[f"indicator_dhat_ge_{j}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(grp_cols).reset_index(drop=True)


def build_across_world_summary(within_df: pd.DataFrame) -> pd.DataFrame:
    """Across the 20 (or 5, for qualification) world-level summaries, for
    each (branch, T): mean/SD/MCSE/95% t-CI of mean_d_hat and Pr(d_hat>=j)."""
    rows = []
    for (branch, T), g in within_df.groupby(["branch", "T_window"]):
        row = {"branch": branch, "T_window": T, "n_worlds": len(g)}
        for metric in ["mean_d_hat"] + [f"Pr_dhat_ge_{j}" for j in range(1, 7)]:
            t = cb.paired_t_test(g[metric].values)  # one-sample: mean/SD/MCSE/CI of the level itself
            row[f"{metric}_mean"] = t["mean"]
            row[f"{metric}_sd"] = t["sd"]
            row[f"{metric}_mcse"] = t["se"]
            row[f"{metric}_ci_low"] = t["ci_low"]
            row[f"{metric}_ci_high"] = t["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["branch", "T_window"]).reset_index(drop=True)


def build_population_summary(pop_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(1, cb.LEADING_EIGS + 1):
        t = cb.paired_t_test(pop_df[f"theta_{i}"].values)
        rows.append({"quantity": f"theta_{i}", "n_worlds": t["n"], "mean": t["mean"], "sd": t["sd"],
                     "mcse": t["se"], "ci_low": t["ci_low"], "ci_high": t["ci_high"]})
    for q in ["theta_2_minus_theta_3", "theta_4_minus_theta_5", "theta_3_over_theta_2", "theta_4_over_theta_2"]:
        t = cb.paired_t_test(pop_df[q].values)
        rows.append({"quantity": q, "n_worlds": t["n"], "mean": t["mean"], "sd": t["sd"],
                     "mcse": t["se"], "ci_low": t["ci_low"], "ci_high": t["ci_high"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired contrast families S, I, B, T
# ---------------------------------------------------------------------------
def paired_contrast(within_df: pd.DataFrame, metric: str, branch_a: str, branch_b: str, T: int):
    """delta = Pr_branch_a - Pr_branch_b, paired by world_index, at fixed T."""
    a = within_df[(within_df.branch == branch_a) & (within_df.T_window == T)].set_index("world_index")[metric]
    b = within_df[(within_df.branch == branch_b) & (within_df.T_window == T)].set_index("world_index")[metric]
    common = a.index.intersection(b.index)
    delta = (a.loc[common] - b.loc[common]).dropna()
    t = cb.paired_t_test(delta.values)
    t["deltas"] = json.dumps({int(k): float(v) for k, v in delta.items()})
    return t


def build_paired_contrasts(within_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Family S: VAR_stationary_start - IID, j in {3,4}, all T (10 tests)
    for j in [3, 4]:
        metric = f"Pr_dhat_ge_{j}"
        for T in cb.T_GRID:
            t = paired_contrast(within_df, metric, "var_stationary_start", "iid_marginal", T)
            rows.append({"family": "S", "test": f"j{j}_T{T}", "metric": metric,
                         "branch_a": "var_stationary_start", "branch_b": "iid_marginal", "T_window": T, **t})

    # Family I: VAR_zero_start - VAR_stationary_start, j in {3,4}, all T (10 tests)
    for j in [3, 4]:
        metric = f"Pr_dhat_ge_{j}"
        for T in cb.T_GRID:
            t = paired_contrast(within_df, metric, "var_zero_start", "var_stationary_start", T)
            rows.append({"family": "I", "test": f"j{j}_T{T}", "metric": metric,
                         "branch_a": "var_zero_start", "branch_b": "var_stationary_start", "T_window": T, **t})

    # Family B: VAR_burnin500 - VAR_stationary_start, j in {3,4}, all T (10 tests)
    for j in [3, 4]:
        metric = f"Pr_dhat_ge_{j}"
        for T in cb.T_GRID:
            t = paired_contrast(within_df, metric, "var_burnin_500", "var_stationary_start", T)
            rows.append({"family": "B", "test": f"j{j}_T{T}", "metric": metric,
                         "branch_a": "var_burnin_500", "branch_b": "var_stationary_start", "T_window": T, **t})

    # Family T: IID T=3000 - IID T=1500, j in {3,4} (2 tests). Paired WITHIN
    # world (same world, two T-values of the SAME nested-prefix path set).
    for j in [3, 4]:
        metric = f"Pr_dhat_ge_{j}"
        a = within_df[(within_df.branch == "iid_marginal") & (within_df.T_window == 3000)].set_index("world_index")[metric]
        b = within_df[(within_df.branch == "iid_marginal") & (within_df.T_window == 1500)].set_index("world_index")[metric]
        common = a.index.intersection(b.index)
        delta = (a.loc[common] - b.loc[common]).dropna()
        t = cb.paired_t_test(delta.values)
        t["deltas"] = json.dumps({int(k): float(v) for k, v in delta.items()})
        rows.append({"family": "T", "test": f"j{j}_T3000_minus_T1500", "metric": metric,
                     "branch_a": "iid_marginal(T=3000)", "branch_b": "iid_marginal(T=1500)",
                     "T_window": None, **t})

    df = pd.DataFrame(rows)
    for family in ["S", "I", "B", "T"]:
        mask = df.family == family
        df.loc[mask, "p_holm"] = cb.holm_adjust(df.loc[mask, "p_value"].values)
    df["holm_significant"] = df["p_holm"] < 0.05
    return df


def build_dhat_frequency_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (branch, T), g in sample_df.groupby(["branch", "T_window"]):
        counts = g["d_hat"].value_counts().sort_index()
        for d_hat_val, n in counts.items():
            rows.append({"branch": branch, "T_window": T, "d_hat": int(d_hat_val),
                        "n_observations": int(n), "n_total": len(g), "frequency": n / len(g)})
    return pd.DataFrame(rows).sort_values(["branch", "T_window", "d_hat"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interpretation decision tree (sec 10)
# ---------------------------------------------------------------------------
def interpretation_classification(pop_summary: pd.DataFrame, contrasts_df: pd.DataFrame,
                                   within_df: pd.DataFrame) -> dict:
    theta = {row["quantity"]: row["mean"] for _, row in pop_summary.iterrows()}
    weak_pop = (theta.get("theta_3", np.nan) < 0.5 * theta.get("theta_1", np.nan)
                and theta.get("theta_4", np.nan) < 0.5 * theta.get("theta_1", np.nan))

    iid_low_at_1500 = within_df[(within_df.branch == "iid_marginal") & (within_df.T_window <= 1500)]
    iid_freq_d2ish = float((iid_low_at_1500["mean_d_hat"].round() == 2).mean()) if len(iid_low_at_1500) else float("nan")

    family_t = contrasts_df[contrasts_df.family == "T"]
    detect_improves_with_T = bool((family_t["mean"] > 0).all() and (family_t["ci_low"] > 0).any())

    A_support = bool(weak_pop and iid_freq_d2ish > 0.5 and detect_improves_with_T)

    family_s = contrasts_df[contrasts_df.family == "S"]
    B_support = bool((family_s["holm_significant"]).any())

    family_i = contrasts_df[contrasts_df.family == "I"]
    family_b = contrasts_df[contrasts_df.family == "B"]
    i_sig = family_i["holm_significant"].any()
    # "burnin500 branch moves toward stationary-start behavior": compare |effect|
    # of burnin500-vs-stationary (family B) to |effect| of zero_start-vs-stationary
    # (family I), matched by (j, T) test key.
    merged = family_i.merge(family_b, on=["metric", "T_window"], suffixes=("_I", "_B"))
    moves_toward = bool((merged["mean_B"].abs() < merged["mean_I"].abs()).mean() > 0.5) if len(merged) else False
    C_support = bool(i_sig and moves_toward)

    supported = [k for k, v in {"A": A_support, "B": B_support, "C": C_support}.items() if v]
    if len(supported) == 0:
        decision = "E_undetermined"
    elif len(supported) == 1:
        decision = supported[0]
    else:
        decision = "D_mixed"

    return {
        "A_weak_population_finite_T_detectability": A_support,
        "A_evidence": {"weak_pop_theta3_theta4": weak_pop,
                      "iid_freq_dhat_round_eq_2_at_T_le_1500": iid_freq_d2ish,
                      "detectability_improves_with_T_family_T": detect_improves_with_T},
        "B_serial_dependence_distortion": B_support,
        "B_evidence": {"family_S_any_holm_significant": bool(B_support)},
        "C_initialization_transient": C_support,
        "C_evidence": {"family_I_any_holm_significant": bool(i_sig),
                      "burnin500_moves_toward_stationary_start": moves_toward},
        "decision": decision,
        "supported_explanations": supported,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pop_df = pd.read_csv(args.run_dir / "population_spectra.csv")
    sample_df = pd.read_csv(args.run_dir / "sample_spectra.csv")

    within_df = build_within_world_summary(sample_df)
    across_df = build_across_world_summary(within_df)
    pop_summary_df = build_population_summary(pop_df)
    contrasts_df = build_paired_contrasts(within_df)
    freq_df = build_dhat_frequency_table(sample_df)
    classification = interpretation_classification(pop_summary_df, contrasts_df, within_df)

    within_df.to_csv(args.out_dir / "within_world_summary.csv", index=False)
    across_df.to_csv(args.out_dir / "across_world_summary.csv", index=False)
    pop_summary_df.to_csv(args.out_dir / "population_summary.csv", index=False)
    contrasts_df.to_csv(args.out_dir / "paired_contrasts.csv", index=False)
    freq_df.to_csv(args.out_dir / "dhat_frequency_table.csv", index=False)
    (args.out_dir / "interpretation_classification.json").write_text(json.dumps(classification, indent=2, default=str))

    print("within_world_summary:", len(within_df), "rows")
    print("across_world_summary:", len(across_df), "rows")
    print("paired_contrasts:", len(contrasts_df), "rows")
    print("dhat_frequency_table:", len(freq_df), "rows")
    print("\nFamily-level Holm significance counts:")
    print(contrasts_df.groupby("family")["holm_significant"].sum())
    print("\nCLASSIFICATION:", json.dumps(classification, indent=2, default=str))


if __name__ == "__main__":
    main()
