#!/usr/bin/env python3
"""
Build the combined mechanism panel, Monte Carlo summaries, four Holm families,
four dose-response sequences, and the observable-feasibility classification /
Stage 6 gate decision for the Stage C full-grid extension.

Reads only frozen raw outputs (this experiment's own new_* CSVs and the frozen
Stage C results/*.csv); writes only into results/ and does not touch any raw
run directory.
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
import common_full_grid as cfg  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                     help="the formal run directory (new_*.csv live here)")
    ap.add_argument("--out-dir", required=True, type=Path)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 1. Combined mechanism / eigenvector / incremental panels
# ---------------------------------------------------------------------------
def build_combined_mechanism(new_cell: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_csv(cfg.STAGE_C_DIR / "results" / "cell_level_alignment.csv")
    frozen = frozen[frozen.r_used.isin(cfg.EXISTING_STAGE_C_R)].copy()
    frozen["source"] = "frozen_stage_c"
    new = new_cell.copy()
    new["source"] = "full_grid_extension"
    combined = pd.concat([new, frozen], ignore_index=True, sort=False)
    dup = combined.duplicated(["replication", "r_used", "forecast_origin_index"]).sum()
    assert dup == 0, f"combined mechanism panel has {dup} duplicate keys"
    return combined


def build_combined_eigenvector(new_eig: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_csv(cfg.STAGE_C_DIR / "results" / "eigenvector_level_alignment.csv")
    frozen["source"] = "frozen_stage_c"
    new = new_eig.rename(columns={
        "eigen_rank": "eig_rank",
        "sq_projection_Q_F": "sq_proj_Q_F", "sq_projection_Q_F_unique": "sq_proj_Q_F_unique",
        "sq_projection_Q_C": "sq_proj_Q_C", "sq_projection_Q_C_unique": "sq_proj_Q_C_unique",
        "sq_projection_Q_C_full": "sq_proj_Q_C_full", "sq_projection_Q_P4": "sq_proj_Q_P4",
    }).copy()
    new["source"] = "full_grid_extension"
    combined = pd.concat([new, frozen], ignore_index=True, sort=False)
    return combined


def build_combined_incremental(new_incr: pd.DataFrame) -> pd.DataFrame:
    """Reshape the frozen Stage C matched_incremental_space_alignment.csv (wide,
    Qmissing3_*/Qextra7_* columns) into the new long schema (Qspace_* columns +
    r_used + space_type) for r=3 and r=7, then concatenate with the new rows."""
    frozen = pd.read_csv(cfg.STAGE_C_DIR / "results" / "matched_incremental_space_alignment.csv")
    union_targets = ["Q_F", "Q_F_unique", "Q_C", "Q_C_unique", "Q_C_full", "Q_P4"]
    suffixes = [
        "dim_U", "dim_Q", "shared_energy", "purity", "capture",
        "expected_random_purity", "expected_random_capture", "excess_purity", "excess_capture",
        "largest_canonical_correlation", "smallest_principal_angle",
        "canonical_correlations", "principal_angles",
    ]
    rows = []
    for r, prefix, space_type in [(3, "Qmissing3", "missing"), (7, "Qextra7", "extra")]:
        for _, frow in frozen.iterrows():
            row = {
                "run_id": frow["run_id"], "replication": frow["replication"],
                "structural_seed": frow["structural_seed"],
                "forecast_origin_index": frow["forecast_origin_index"],
                "target_index": frow["target_index"], "r_used": r, "space_type": space_type,
                "rank_Q_Rr": frow[f"rank_Q_R{r}"], "rank_Q_R5": frow["rank_Q_R5"],
                "expected_rank": cfg.expected_incremental_rank(r),
                "rank_Qspace": frow[f"rank_Q_{'missing_3' if r == 3 else 'extra_7'}"],
                "rank_ok": frow["expected_ranks_ok"],
                "nesting_max_all_adjacent": max(frow["nesting_3_in_5"], frow["nesting_5_in_7"]),
                "nesting_ok": frow["nesting_ok"], "source": "frozen_stage_c",
            }
            for tname in union_targets:
                src_col_base = f"{prefix}_{tname}"
                for suf in suffixes:
                    src_col = f"{src_col_base}_{suf}"
                    dst_col = f"Qspace_{tname}_{suf}"
                    row[dst_col] = frow[src_col] if src_col in frow else float("nan")
            rows.append(row)
    frozen_long = pd.DataFrame(rows)
    new = new_incr.copy()
    new["source"] = "full_grid_extension"
    combined = pd.concat([new, frozen_long], ignore_index=True, sort=False)
    dup = combined.duplicated(["replication", "r_used", "forecast_origin_index", "space_type"]).sum()
    assert dup == 0, f"combined incremental panel has {dup} duplicate keys"
    return combined


# ---------------------------------------------------------------------------
# 2. Within-replication averaging (25 origins -> 1 value per replication x r)
# ---------------------------------------------------------------------------
def within_replication_mean(df: pd.DataFrame, metric: str, group_cols=("replication", "r_used")):
    g = df.groupby(list(group_cols))[metric]
    mean = g.mean()
    n_valid = g.apply(lambda s: s.notna().sum())
    n_total = g.size()
    out = pd.DataFrame({"mean": mean, "n_valid_origins": n_valid, "n_total_origins": n_total}).reset_index()
    return out


def mc_summary(rep_level_values: pd.Series):
    vals = rep_level_values.dropna().values
    n = len(vals)
    if n < 2:
        return {"n_valid_replications": n, "mean": float(vals.mean()) if n else float("nan"),
                "sd": float("nan"), "mcse": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    t = cfg.one_sample_t_interval(vals)
    return {"n_valid_replications": n, "mean": t["mean"], "sd": t["sd"], "mcse": t["se"],
            "ci_low": t["ci_low"], "ci_high": t["ci_high"]}


# ---------------------------------------------------------------------------
# 3. Holm families (U1, U2, O1, O2)
# ---------------------------------------------------------------------------
def paired_contrast_by_r(rep_r_means: pd.DataFrame, metric_col: str, r_test: int, r_ref: int = 5):
    a = rep_r_means[rep_r_means.r_used == r_test].set_index("replication")["mean"]
    b = rep_r_means[rep_r_means.r_used == r_ref].set_index("replication")["mean"]
    common = a.index.intersection(b.index)
    delta = (a.loc[common] - b.loc[common]).dropna()
    result = cfg.paired_t_test(delta.values)
    result["r_test"] = r_test
    result["r_ref"] = r_ref
    result["deltas"] = json.dumps({int(k): float(v) for k, v in delta.items()})
    return result


def level_test_by_r(rep_r_means: pd.DataFrame, metric_col: str, r_test: int):
    vals = rep_r_means[rep_r_means.r_used == r_test].set_index("replication")["mean"].dropna()
    result = cfg.one_sample_t_interval(vals.values)
    result["r_test"] = r_test
    result["deltas"] = json.dumps({int(k): float(v) for k, v in vals.items()})
    return result


def build_holm_families(combined_cell: pd.DataFrame, combined_incr: pd.DataFrame):
    rows = []
    umpq_f_unique = within_replication_mean(combined_cell, "UMP_Q_F_unique_purity")
    u4_qc_capture = within_replication_mean(combined_cell, "U4_Q_C_capture")
    qextra_qc_unique_excess = within_replication_mean(
        combined_incr[combined_incr.space_type == "extra"], "Qspace_Q_C_unique_excess_purity")

    # U1: under-grid U_MP factor-unique purity, r in {1,2,3,4} vs 5, sign required: positive
    for r in [1, 2, 3, 4]:
        res = paired_contrast_by_r(umpq_f_unique, "UMP_Q_F_unique_purity", r, 5)
        rows.append({"family": "U1", "test": f"r{r}_vs_r5", "metric": "UMP_Q_F_unique_purity",
                      "required_sign": "positive", **res})
    # U2: under-grid U_4 capture of Q_C, r in {1,2,3,4} vs 5, sign required: negative
    for r in [1, 2, 3, 4]:
        res = paired_contrast_by_r(u4_qc_capture, "U4_Q_C_capture", r, 5)
        rows.append({"family": "U2", "test": f"r{r}_vs_r5", "metric": "U4_Q_C_capture",
                      "required_sign": "negative", **res})
    # O1: over-grid Q_extra_r community-unique excess-purity LEVEL tests, r in {6,7,8,9}, sign: positive
    for r in [6, 7, 8, 9]:
        res = level_test_by_r(qextra_qc_unique_excess, "Qspace_Q_C_unique_excess_purity", r)
        rows.append({"family": "O1", "test": f"r{r}_level", "metric": "Qextra_r_Q_C_unique_excess_purity",
                      "required_sign": "positive", **res})
    # O2: over-grid U_4 capture of Q_C, r in {6,7,8,9} vs 5, sign required: negative
    for r in [6, 7, 8, 9]:
        res = paired_contrast_by_r(u4_qc_capture, "U4_Q_C_capture", r, 5)
        rows.append({"family": "O2", "test": f"r{r}_vs_r5", "metric": "U4_Q_C_capture",
                      "required_sign": "negative", **res})

    df = pd.DataFrame(rows)
    for family in ["U1", "U2", "O1", "O2"]:
        mask = df.family == family
        df.loc[mask, "p_holm"] = cfg.holm_adjust(df.loc[mask, "p_value"].values)

    def sign_ok(row):
        if row["required_sign"] == "positive":
            return row["ci_low"] > 0
        return row["ci_high"] < 0

    df["directional_ci_ok"] = df.apply(sign_ok, axis=1)
    df["holm_significant"] = df["p_holm"] < 0.05
    df["test_supported"] = df["directional_ci_ok"] & df["holm_significant"]

    family_decision = df.groupby("family")["test_supported"].all().rename("family_supported").reset_index()
    return df, family_decision


# ---------------------------------------------------------------------------
# 4. Dose-response: four ordered adjacent-step sequences
# ---------------------------------------------------------------------------
def adjacent_deltas(rep_r_means: pd.DataFrame, r_order):
    """delta_i(step k) = m_i(r_order[k+1]) - m_i(r_order[k]) for each replication i."""
    piv = rep_r_means.pivot(index="replication", columns="r_used", values="mean")
    steps = []
    for k in range(len(r_order) - 1):
        a, b = r_order[k], r_order[k + 1]
        if a in piv.columns and b in piv.columns:
            steps.append((a, b, (piv[b] - piv[a]).dropna()))
    return steps


def build_dose_response(combined_cell: pd.DataFrame, combined_incr: pd.DataFrame):
    umpq_f_unique = within_replication_mean(combined_cell, "UMP_Q_F_unique_purity")
    u4_qc_capture = within_replication_mean(combined_cell, "U4_Q_C_capture")

    # "over total community removal" needs an r=5 anchor with capture/shared_energy = 0
    extra = combined_incr[combined_incr.space_type == "extra"]
    qextra_capture = within_replication_mean(extra, "Qspace_Q_C_capture")
    reps = qextra_capture.replication.unique()
    anchor = pd.DataFrame({"replication": reps, "r_used": 5, "mean": 0.0,
                            "n_valid_origins": 25, "n_total_origins": 25})
    qextra_capture_with_anchor = pd.concat([qextra_capture, anchor], ignore_index=True)

    sequences = {
        "under_factor_contamination": {
            "table": umpq_f_unique, "order": [1, 2, 3, 4, 5],
            "expected_direction": "non_increasing",  # r1 >= r2 >= ... >= r5
            "metric": "UMP_Q_F_unique_purity",
        },
        "under_community_recovery": {
            "table": u4_qc_capture, "order": [1, 2, 3, 4, 5],
            "expected_direction": "non_decreasing",  # r1 <= r2 <= ... <= r5
            "metric": "U4_Q_C_capture",
        },
        "over_total_community_removal": {
            "table": qextra_capture_with_anchor, "order": [5, 6, 7, 8, 9],
            "expected_direction": "non_decreasing",  # r5(=0) <= r6 <= ... <= r9
            "metric": "Qextra_r_Q_C_capture_or_shared_energy",
        },
        "over_residual_community_geometry": {
            "table": u4_qc_capture, "order": [5, 6, 7, 8, 9],
            "expected_direction": "non_increasing",  # r5 >= r6 >= ... >= r9
            "metric": "U4_Q_C_capture",
        },
    }

    all_rows = []
    seq_decision = []
    for seq_name, spec in sequences.items():
        steps = adjacent_deltas(spec["table"], spec["order"])
        pvals = []
        step_rows = []
        for (a, b, delta) in steps:
            t = cfg.paired_t_test(delta.values)
            # required sign of (b - a) given expected_direction
            required_sign = "positive" if spec["expected_direction"] == "non_decreasing" else "negative"
            sign_ok = (t["ci_low"] > 0) if required_sign == "positive" else (t["ci_high"] < 0)
            n_pos = int((delta > 0).sum())
            n_neg = int((delta < 0).sum())
            step_rows.append({
                "sequence": seq_name, "metric": spec["metric"], "step": f"r{a}_to_r{b}",
                "required_sign": required_sign, "mean": t["mean"], "sd": t["sd"], "mcse": t["se"],
                "ci_low": t["ci_low"], "ci_high": t["ci_high"], "p_value": t["p_value"], "n": t["n"],
                "sign_ci_ok": sign_ok,
                "replication_sign_frequency_positive": n_pos, "replication_sign_frequency_negative": n_neg,
            })
            pvals.append(t["p_value"])
        p_holm = cfg.holm_adjust(pvals) if pvals else []
        for row, ph in zip(step_rows, p_holm):
            row["p_holm"] = ph
            row["step_supported"] = row["sign_ci_ok"] and (ph < 0.05)
        all_rows.extend(step_rows)

        monotone_supported = bool(step_rows) and all(r["step_supported"] for r in step_rows)
        breaking_steps = [r["step"] for r in step_rows if not r["step_supported"]]

        # fraction of replications satisfying the ENTIRE proposed order
        piv = spec["table"].pivot(index="replication", columns="r_used", values="mean")
        order = spec["order"]
        if all(c in piv.columns for c in order):
            if spec["expected_direction"] == "non_decreasing":
                sat = piv[order].apply(lambda row: all(row.iloc[k] <= row.iloc[k + 1] + 1e-12
                                                        for k in range(len(order) - 1)), axis=1)
            else:
                sat = piv[order].apply(lambda row: all(row.iloc[k] >= row.iloc[k + 1] - 1e-12
                                                        for k in range(len(order) - 1)), axis=1)
            frac_satisfying = float(sat.mean())
        else:
            frac_satisfying = float("nan")

        seq_decision.append({
            "sequence": seq_name, "metric": spec["metric"], "expected_direction": spec["expected_direction"],
            "monotone_supported": monotone_supported, "breaking_steps": json.dumps(breaking_steps),
            "fraction_replications_satisfying_full_order": frac_satisfying,
        })

    return pd.DataFrame(all_rows), pd.DataFrame(seq_decision)


# ---------------------------------------------------------------------------
# 5. Observable feasibility panel
# ---------------------------------------------------------------------------
def build_observable_feasibility(obs_df: pd.DataFrame):
    rep_means = {}
    for metric in cfg.PRIMARY_OBSERVABLES:
        rep_means[metric] = within_replication_mean(obs_df, metric)

    all_tests = []
    for metric in cfg.PRIMARY_OBSERVABLES:
        table = rep_means[metric]
        # under-side adjacent contrasts r=1..5
        under_steps = adjacent_deltas(table, [1, 2, 3, 4, 5])
        for a, b, delta in under_steps:
            t = cfg.paired_t_test(delta.values)
            all_tests.append({"observable": metric, "test_type": "under_adjacent", "test": f"r{a}_to_r{b}",
                               **t, "n_pos": int((delta > 0).sum()), "n_neg": int((delta < 0).sum()),
                               "n_total": len(delta)})
        # over-side adjacent contrasts r=5..9
        over_steps = adjacent_deltas(table, [5, 6, 7, 8, 9])
        for a, b, delta in over_steps:
            t = cfg.paired_t_test(delta.values)
            all_tests.append({"observable": metric, "test_type": "over_adjacent", "test": f"r{a}_to_r{b}",
                               **t, "n_pos": int((delta > 0).sum()), "n_neg": int((delta < 0).sum()),
                               "n_total": len(delta)})
        # mirror contrasts
        for r_lo, r_hi in [(1, 9), (2, 8), (3, 7), (4, 6)]:
            a = table[table.r_used == r_lo].set_index("replication")["mean"]
            b = table[table.r_used == r_hi].set_index("replication")["mean"]
            common = a.index.intersection(b.index)
            delta = (a.loc[common] - b.loc[common]).dropna()
            t = cfg.paired_t_test(delta.values)
            all_tests.append({"observable": metric, "test_type": "mirror", "test": f"r{r_lo}_minus_r{r_hi}",
                               **t, "n_pos": int((delta > 0).sum()), "n_neg": int((delta < 0).sum()),
                               "n_total": len(delta)})

    tests_df = pd.DataFrame(all_tests)
    tests_df["p_holm"] = cfg.holm_adjust(tests_df["p_value"].values)  # ONE Holm family across all primary tests
    tests_df["significant_holm"] = tests_df["p_holm"] < 0.05

    # classification
    classification_rows = []
    for metric in cfg.PRIMARY_OBSERVABLES:
        sub = tests_df[tests_df.observable == metric]
        under = sub[sub.test_type == "under_adjacent"]
        over = sub[sub.test_type == "over_adjacent"]
        mirror = sub[sub.test_type == "mirror"]

        def monotone_ok(block):
            # Consistent with the dose-response "step_supported" bar (sec 7): every
            # adjacent step must have the correct interval sign AND be Holm-adjusted
            # significant (using the single Holm family across ALL primary-observable
            # feasibility tests, sec 8) -- not just a raw, unadjusted CI.
            if len(block) == 0:
                return False, None
            signs = np.sign(block["mean"].values)
            consistent = len(set(signs[signs != 0])) <= 1 if any(signs != 0) else True
            ci_excludes_zero = ((block["ci_low"] > 0) | (block["ci_high"] < 0)).all()
            holm_sig = block["significant_holm"].all()
            return bool(consistent and ci_excludes_zero and holm_sig), (signs[0] if len(signs) else None)

        under_mono, under_dir = monotone_ok(under)
        over_mono, over_dir = monotone_ok(over)
        both_sides_monotone = under_mono and over_mono

        mirror_sig = mirror["significant_holm"].all() if len(mirror) else False
        mirror_signs = np.sign(mirror["mean"].values) if len(mirror) else np.array([])
        mirror_consistent_direction = (len(set(mirror_signs[mirror_signs != 0])) <= 1) if len(mirror_signs) else False
        mirror_all_significant_consistent = bool(mirror_sig and mirror_consistent_direction and len(mirror) == 4)

        sign_freq_ok = bool((mirror["n_pos"].combine(mirror["n_total"] - mirror["n_pos"], max) >= 16).all()) \
            if len(mirror) else False

        if both_sides_monotone and mirror_all_significant_consistent and sign_freq_ok:
            klass = "direction_capable"
        elif both_sides_monotone:
            klass = "distance_only"
        else:
            klass = "not_a_usable_correction_signal"

        classification_rows.append({
            "observable": metric,
            "information_class": "K_agnostic" if metric in cfg.K_AGNOSTIC_OBSERVABLES else "K_informed",
            "under_side_monotone": under_mono, "over_side_monotone": over_mono,
            "mirror_all_significant_consistent_direction": mirror_all_significant_consistent,
            "mirror_sign_frequency_ok_16_of_20": sign_freq_ok,
            "classification": klass,
        })

    class_df = pd.DataFrame(classification_rows)
    return tests_df, class_df


def stage6_gate_decision(class_df: pd.DataFrame) -> dict:
    capable = class_df[class_df.classification == "direction_capable"]
    distance_only = class_df[class_df.classification == "distance_only"]
    if len(capable) > 0:
        k_agnostic_capable = capable[capable.information_class == "K_agnostic"]
        decision = "A"
        conditional_on_k4 = len(k_agnostic_capable) == 0
    elif len(distance_only) > 0:
        decision = "B"
        conditional_on_k4 = False
    else:
        decision = "C"
        conditional_on_k4 = False
    return {"decision": decision, "conditional_on_known_K4": bool(conditional_on_k4),
            "direction_capable_observables": capable["observable"].tolist(),
            "distance_only_observables": distance_only["observable"].tolist()}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    new_cell = pd.read_csv(args.run_dir / "new_mechanism_cell_level.csv")
    new_eig = pd.read_csv(args.run_dir / "new_eigenvector_level.csv")
    new_incr = pd.read_csv(args.run_dir / "new_incremental_space_long.csv")
    obs_df = pd.read_csv(args.run_dir / "observable_full_grid_cell_level.csv")

    combined_cell = build_combined_mechanism(new_cell)
    combined_eig = build_combined_eigenvector(new_eig)
    combined_incr = build_combined_incremental(new_incr)

    combined_cell.to_csv(args.out_dir / "combined_mechanism_cell_level.csv", index=False)
    combined_eig.to_csv(args.out_dir / "combined_eigenvector_level.csv", index=False)
    combined_incr.to_csv(args.out_dir / "combined_incremental_space_long.csv", index=False)

    # within-replication summary (mechanism + observables), long format
    wr_rows = []
    for metric in ["UMP_Q_F_unique_purity", "U4_Q_C_capture", "UMP_Q_C_purity", "QR_Q_F_purity"]:
        t = within_replication_mean(combined_cell, metric)
        t["metric"] = metric
        t["series"] = "mechanism_cell"
        wr_rows.append(t)
    for metric in cfg.PRIMARY_OBSERVABLES:
        t = within_replication_mean(obs_df, metric)
        t["metric"] = metric
        t["series"] = "observable"
        wr_rows.append(t)
    within_df = pd.concat(wr_rows, ignore_index=True)
    within_df.to_csv(args.out_dir / "within_replication_summary.csv", index=False)

    holm_tests_df, family_decision_df = build_holm_families(combined_cell, combined_incr)
    holm_tests_df.to_csv(args.out_dir / "full_grid_pairwise_contrasts.csv", index=False)
    family_decision_df.to_csv(args.out_dir / "holm_family_decisions.csv", index=False)

    dose_steps_df, dose_seq_df = build_dose_response(combined_cell, combined_incr)
    dose_steps_df.to_csv(args.out_dir / "dose_response_adjacent_contrasts.csv", index=False)
    dose_seq_df.to_csv(args.out_dir / "dose_response_sequence_decisions.csv", index=False)

    obs_tests_df, obs_class_df = build_observable_feasibility(obs_df)
    obs_tests_df.to_csv(args.out_dir / "observable_feasibility_tests.csv", index=False)
    obs_class_df.to_csv(args.out_dir / "observable_signal_classification.csv", index=False)

    gate = stage6_gate_decision(obs_class_df)
    (args.out_dir / "stage6_gate_decision.json").write_text(json.dumps(gate, indent=2))

    print("combined_mechanism_cell_level.csv:", len(combined_cell), "rows")
    print("combined_eigenvector_level.csv:", len(combined_eig), "rows")
    print("combined_incremental_space_long.csv:", len(combined_incr), "rows")
    print("holm_family_decisions:")
    print(family_decision_df.to_string(index=False))
    print("dose_response_sequence_decisions:")
    print(dose_seq_df.to_string(index=False))
    print("observable_signal_classification:")
    print(obs_class_df.to_string(index=False))
    print("STAGE 6 GATE DECISION:", json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
