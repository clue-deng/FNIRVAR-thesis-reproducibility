#!/usr/bin/env python3
"""
Stage C subspace-mechanism diagnostic.

Reproduces the formal rolling window exactly for r_used in {3,5,7} at a fixed
set of forecast origins, on the same 20 structural worlds as the formal run.
Computes subspace-alignment metrics only -- no GMM, OLS, forecasting, or MSPE.
Imports the released package and the frozen formal `common.py` unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_c as cs  # noqa: E402

FORMAL_DIR = cs.FORMAL_DIR
K_TRUE = cs.K_TRUE
N = cs.N
L_F = cs.L_F
LOOKBACK_WINDOW = cs.LOOKBACK_WINDOW
FIRST_PREDICTION_DAY = cs.FIRST_PREDICTION_DAY
EMBEDDING_METHOD = cs.EMBEDDING_METHOD

TARGETS_U = ["Q_F", "Q_F_unique", "Q_C", "Q_C_unique", "Q_C_full", "Q_P4"]
TARGETS_QR = ["Q_F", "Q_C", "Q_C_unique", "Q_C_full", "Q_P4"]
TARGETS_QMISSING3 = ["Q_F", "Q_F_unique", "Q_C", "Q_C_unique", "Q_P4"]
TARGETS_QEXTRA7 = ["Q_C", "Q_C_unique", "Q_C_full", "Q_P4", "Q_F"]

METRIC_KEYS = [
    "dim_U", "dim_Q", "shared_energy", "purity", "capture",
    "expected_random_purity", "expected_random_capture",
    "excess_purity", "excess_capture",
    "largest_canonical_correlation", "smallest_principal_angle",
]


def flatten_comparison(prefix: str, cmp: dict) -> dict:
    out = {f"{prefix}_{k}": cmp[k] for k in METRIC_KEYS}
    out[f"{prefix}_canonical_correlations"] = json.dumps(
        [round(float(x), 10) for x in cmp["canonical_correlations"]]
    )
    out[f"{prefix}_principal_angles"] = json.dumps(
        [round(float(x), 10) for x in cmp["principal_angles"]]
    )
    return out


def formal_run_dir(replication: int) -> Path:
    if replication == 0:
        return FORMAL_DIR / "runs" / "qualification_final_rep0_20260816"
    return FORMAL_DIR / "runs" / f"formal_preferred_rep{replication:02d}_20260816"


def load_formal_manifest(replication: int) -> dict:
    return json.loads((formal_run_dir(replication) / "manifest_final.json").read_text())


def load_formal_primary_rows(replication: int, r_grid, origins) -> pd.DataFrame:
    df = pd.read_csv(formal_run_dir(replication) / "origin_results.csv")
    sel = df[
        (df.branch == "primary_fixed_K")
        & (df.r_used.isin(r_grid))
        & (df.forecast_origin_index.isin(origins))
    ].copy()
    return sel.set_index(["r_used", "forecast_origin_index"])


def git_status() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cs.REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cs.REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    return {"git_commit": commit, "git_status_porcelain": porcelain}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--replications", required=True, help="Comma-separated replication indices")
    p.add_argument("--r-grid", default="3,5,7")
    p.add_argument("--origins", required=True, help="Comma-separated forecast_origin_index values")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def process_replication(replication: int, r_grid, origins, config, log):
    seed = cs.structural_seed_for_index(replication, config["n_total_replications"])
    world = cs.generate_world(seed)

    formal_manifest = load_formal_manifest(replication)
    formal_rows = load_formal_primary_rows(replication, r_grid, origins)

    formal_seed_tree = formal_manifest.get("seed_tree", {})
    world_check = {
        "replication": replication,
        "structural_seed_match": world["structural_seed"] == formal_manifest["structural_seed"],
        "network_seed_match": world["network_seed"] == formal_seed_tree.get("network_seed"),
        "factor_seed_match": world["factor_seed"] == formal_seed_tree.get("factor_seed"),
        "gmm_seed_match": world["gmm_seed"] == formal_manifest.get("gmm_seed"),
        "phi_allclose_formal": bool(np.isclose(
            world["actual_spectral_radius"], formal_manifest["actual_spectral_radius"],
            rtol=1e-9, atol=1e-9,
        )),
        "actual_spectral_radius": world["actual_spectral_radius"],
        "formal_actual_spectral_radius": formal_manifest["actual_spectral_radius"],
    }

    Lambda = cs.reconstruct_lambda(world["factor_seed"])
    Lambda_hash = cs.sha256_of_array(Lambda)
    Q_F = cs.orth(Lambda)
    Z_full, Z_contrast = cs.community_targets(world["true_labels"])
    Q_C_full = cs.orth(Z_full)
    Q_C = cs.orth(Z_contrast)
    Q_F_unique = cs.project_out(Q_F, Q_C)
    Q_C_unique = cs.project_out(Q_C, Q_F)

    stat = cs.stationary_idiosyncratic_correlation(world["Phi"], world["Omega"])
    Q_P4 = stat["Q_P4"]

    baseline = {
        "replication": replication,
        "structural_seed": world["structural_seed"],
        "Lambda_sha256": Lambda_hash,
        "rank_Q_F": Q_F.shape[1],
        "rank_Q_C_full": Q_C_full.shape[1],
        "rank_Q_C": Q_C.shape[1],
        "rank_Q_F_unique": Q_F_unique.shape[1],
        "rank_Q_C_unique": Q_C_unique.shape[1],
        "rank_Q_P4": Q_P4.shape[1],
        "lyap_residual_rel": stat["lyap_residual_rel"],
        "sum_vs_lyap_close": stat["sum_vs_lyap_close"],
        "finite_sum_k_used": stat["k_used"],
        "finite_sum_last_ratio": stat["last_ratio"],
        "finite_sum_converged": stat["finite_sum_converged"],
        "corr_P_is_finite": stat["is_finite"],
        "corr_P_is_symmetric": stat["is_symmetric"],
        "corr_P_positive_diagonal": stat["positive_diagonal"],
        "corr_P_unit_diagonal": stat["unit_diagonal"],
        "corr_P_is_psd": stat["is_psd"],
        "corr_P_min_eig": stat["min_eig"],
    }
    # Baseline overlap between the four targets (registered design sec 4).
    targets = {"Q_F": Q_F, "Q_C": Q_C, "Q_C_full": Q_C_full, "Q_P4": Q_P4}
    for a in targets:
        for b in targets:
            if a >= b:
                continue
            c = cs.subspace_comparison(targets[a], targets[b])
            baseline[f"overlap_{a}_vs_{b}_purity"] = c["purity"]
            baseline[f"overlap_{a}_vs_{b}_capture"] = c["capture"]
            baseline[f"overlap_{a}_vs_{b}_largest_cc"] = c["largest_canonical_correlation"]

    X = world["X_full"][: cs.T_EVAL]

    cell_rows = []
    eig_rows = []
    matched_rows = []

    for origin_i in origins:
        target_i = FIRST_PREDICTION_DAY + 1 + origin_i
        window_start = FIRST_PREDICTION_DAY + origin_i - LOOKBACK_WINDOW
        window_stop = FIRST_PREDICTION_DAY + origin_i + 1
        window = X[window_start:window_stop]
        assert window.shape == (LOOKBACK_WINDOW + 1, N), window.shape

        Q_R = {}
        residual_by_r = {}
        eig_by_r = {}
        for r_used in r_grid:
            factor_model = cs.FactorAdjustment(window, r_used, L_F)
            residual = factor_model.get_idiosyncratic_component()
            estimated_loadings = factor_model.loadings()
            residual_by_r[r_used] = residual
            Q_R[r_used] = cs.orth(estimated_loadings)
            eig_by_r[r_used] = cs.mp_eigendecomposition(residual)

        # --- matched-group nesting / incremental spaces (uses r=3,5,7 jointly) ---
        matched_row = {
            "run_id": None, "replication": replication,
            "structural_seed": world["structural_seed"],
            "forecast_origin_index": origin_i, "target_index": target_i,
            "rank_Q_R3": Q_R[3].shape[1], "rank_Q_R5": Q_R[5].shape[1],
            "rank_Q_R7": Q_R[7].shape[1],
            "nesting_3_in_5": cs.nesting_residual_norm(Q_R[3], Q_R[5]),
            "nesting_5_in_7": cs.nesting_residual_norm(Q_R[5], Q_R[7]),
        }
        matched_row["nesting_ok"] = (
            matched_row["nesting_3_in_5"] <= 1e-8 and matched_row["nesting_5_in_7"] <= 1e-8
        )
        Q_missing_3 = cs.project_out(Q_R[5], Q_R[3])
        Q_extra_7 = cs.project_out(Q_R[7], Q_R[5])
        matched_row["rank_Q_missing_3"] = Q_missing_3.shape[1]
        matched_row["rank_Q_extra_7"] = Q_extra_7.shape[1]
        matched_row["expected_ranks_ok"] = (
            Q_missing_3.shape[1] == 2 and Q_extra_7.shape[1] == 2
        )
        for tname in TARGETS_QMISSING3:
            target = {"Q_F": Q_F, "Q_F_unique": Q_F_unique, "Q_C": Q_C,
                      "Q_C_unique": Q_C_unique, "Q_P4": Q_P4}[tname]
            c = cs.subspace_comparison(Q_missing_3, target)
            matched_row.update(flatten_comparison(f"Qmissing3_{tname}", c))
        for tname in TARGETS_QEXTRA7:
            target = {"Q_F": Q_F, "Q_C": Q_C, "Q_C_unique": Q_C_unique,
                      "Q_C_full": Q_C_full, "Q_P4": Q_P4}[tname]
            c = cs.subspace_comparison(Q_extra_7, target)
            matched_row.update(flatten_comparison(f"Qextra7_{tname}", c))
        matched_rows.append(matched_row)

        # --- per r_used cell-level rows ---
        for r_used in r_grid:
            residual = residual_by_r[r_used]
            eigd = eig_by_r[r_used]
            resid_hash = cs.sha256_of_array(residual)

            # Package d_hat cross-check (constructs NIRVAR only to read .d;
            # no gmm()/ols()/predict() call -- no GMM/OLS/forecasting is run).
            primary_model = cs.NIRVAR(
                Xi=residual, d=None, K=K_TRUE,
                embedding_method=EMBEDDING_METHOD, gmm_random_int=world["gmm_seed"],
            )
            d_hat_package = int(primary_model.d)

            key = (r_used, origin_i)
            if key in formal_rows.index:
                frow = formal_rows.loc[key]
                formal_residual_sha256 = frow["residual_sha256"]
                formal_mp_edge = float(frow["mp_edge"])
                formal_d_hat_package = int(frow["d_hat_package"])
                formal_d_hat_independent = int(frow["d_hat_independent"])
                formal_ARI = float(frow["ARI"])
                formal_sq_err_sum = float(frow["squared_error_sum"])
                formal_sq_err_denom = float(frow["squared_error_denominator"])
                formal_status = frow["origin_status"]
            else:
                formal_residual_sha256 = None
                formal_mp_edge = np.nan
                formal_d_hat_package = None
                formal_d_hat_independent = None
                formal_ARI = np.nan
                formal_sq_err_sum = np.nan
                formal_sq_err_denom = np.nan
                formal_status = None

            row = {
                "run_id": None, "replication": replication,
                "structural_seed": world["structural_seed"],
                "network_seed": world["network_seed"], "factor_seed": world["factor_seed"],
                "gmm_seed": world["gmm_seed"],
                "r_used": r_used, "forecast_origin_index": origin_i,
                "target_index": target_i, "T_window": window.shape[0],
                "residual_sha256": resid_hash,
                "mp_edge": eigd["mp_edge"],
                "leading_eigenvalues": json.dumps([round(float(x), 10) for x in eigd["leading_eigs"]]),
                "d_hat_independent": eigd["d_hat_independent"],
                "d_hat_package": d_hat_package,
                "rank_U_MP": eigd["U_MP"].shape[1],
                "rank_U_4": eigd["U_4"].shape[1],
                "rank_Q_R": Q_R[r_used].shape[1],
                "formal_residual_sha256": formal_residual_sha256,
                "formal_mp_edge": formal_mp_edge,
                "formal_d_hat_package": formal_d_hat_package,
                "formal_d_hat_independent": formal_d_hat_independent,
                "formal_ARI": formal_ARI,
                "formal_squared_error_sum": formal_sq_err_sum,
                "formal_squared_error_denominator": formal_sq_err_denom,
                "formal_origin_status": formal_status,
                "residual_hash_matches_formal": (resid_hash == formal_residual_sha256),
                "mp_edge_matches_formal": (
                    formal_mp_edge is not None and not np.isnan(formal_mp_edge)
                    and abs(eigd["mp_edge"] - formal_mp_edge) < 1e-9
                ),
                "d_hat_matches_formal": (d_hat_package == formal_d_hat_package
                                          and eigd["d_hat_independent"] == formal_d_hat_independent),
            }

            target_map = {
                "Q_F": Q_F, "Q_F_unique": Q_F_unique, "Q_C": Q_C,
                "Q_C_unique": Q_C_unique, "Q_C_full": Q_C_full, "Q_P4": Q_P4,
            }
            for tname in TARGETS_U:
                c_mp = cs.subspace_comparison(eigd["U_MP"], target_map[tname])
                row.update(flatten_comparison(f"UMP_{tname}", c_mp))
                c_4 = cs.subspace_comparison(eigd["U_4"], target_map[tname])
                row.update(flatten_comparison(f"U4_{tname}", c_4))
            for tname in TARGETS_QR:
                c_r = cs.subspace_comparison(Q_R[r_used], target_map[tname])
                row.update(flatten_comparison(f"QR_{tname}", c_r))

            cell_rows.append(row)

            # --- eigenvector-level rows (top 8) ---
            eigvals = eigd["eigvals_desc"]
            n_eig = min(8, eigvals.shape[0]) if eigvals.size else 0
            corr = None
            for rank_idx in range(n_eig):
                v = None
                # recompute eigenvector on demand from stored U_4/U_MP if within range,
                # else recompute full eigh once (cheap, N=100).
                if rank_idx < eigd["U_4"].shape[1]:
                    v = eigd["U_4"][:, rank_idx]
                else:
                    if corr is None:
                        corr = np.corrcoef(residual.T)
                    w, V = np.linalg.eigh(corr)
                    order = np.argsort(w)[::-1]
                    v = V[:, order][:, rank_idx]
                eig_rows.append({
                    "run_id": None, "replication": replication, "r_used": r_used,
                    "forecast_origin_index": origin_i, "eig_rank": rank_idx + 1,
                    "eigenvalue": float(eigvals[rank_idx]),
                    "exceeds_mp_edge": bool(eigvals[rank_idx] > eigd["mp_edge"]),
                    "sq_proj_Q_F": float(np.sum((Q_F.T @ v) ** 2)) if Q_F.shape[1] else 0.0,
                    "sq_proj_Q_F_unique": float(np.sum((Q_F_unique.T @ v) ** 2)) if Q_F_unique.shape[1] else 0.0,
                    "sq_proj_Q_C": float(np.sum((Q_C.T @ v) ** 2)) if Q_C.shape[1] else 0.0,
                    "sq_proj_Q_C_unique": float(np.sum((Q_C_unique.T @ v) ** 2)) if Q_C_unique.shape[1] else 0.0,
                    "sq_proj_Q_C_full": float(np.sum((Q_C_full.T @ v) ** 2)) if Q_C_full.shape[1] else 0.0,
                    "sq_proj_Q_P4": float(np.sum((Q_P4.T @ v) ** 2)) if Q_P4.shape[1] else 0.0,
                })

    return world_check, baseline, cell_rows, matched_rows, eig_rows


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    r_grid = [int(x) for x in args.r_grid.split(",")]
    origins = [int(x) for x in args.origins.split(",")]
    replications = [int(x) for x in args.replications.split(",")]

    if args.outdir.exists():
        raise SystemExit(f"refusing to overwrite existing run dir: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=False)

    script_sha256 = sha256_file(Path(__file__))
    common_sha256 = sha256_file(HERE / "common_stage_c.py")
    config_sha256 = sha256_file(args.config)
    (args.outdir / "config_used.json").write_text(args.config.read_text())

    started = {
        "run_id": args.run_id, "started_epoch": time.time(),
        "replications": replications, "r_grid": r_grid, "origins": origins,
        "script_sha256": script_sha256, "common_sha256": common_sha256,
        "config_sha256": config_sha256, **git_status(),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "cpu_count": os.cpu_count(),
        "env_threads": {
            k: os.environ.get(k) for k in
            ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
        },
    }
    (args.outdir / "manifest_started.json").write_text(json.dumps(started, indent=2, default=str))

    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"Stage C run {args.run_id}: {len(replications)} replications, r_grid={r_grid}, "
        f"{len(origins)} origins each -> {len(replications) * len(r_grid) * len(origins)} cells")

    t0 = time.time()
    all_cell_rows, all_matched_rows, all_eig_rows = [], [], []
    all_world_checks, all_baselines = [], []
    errors = []
    for idx, replication in enumerate(replications):
        try:
            world_check, baseline, cell_rows, matched_rows, eig_rows = process_replication(
                replication, r_grid, origins, config, log
            )
            for r in cell_rows:
                r["run_id"] = args.run_id
            for r in matched_rows:
                r["run_id"] = args.run_id
            for r in eig_rows:
                r["run_id"] = args.run_id
            all_world_checks.append(world_check)
            all_baselines.append(baseline)
            all_cell_rows.extend(cell_rows)
            all_matched_rows.extend(matched_rows)
            all_eig_rows.extend(eig_rows)
            log(f"replication {replication} done ({idx + 1}/{len(replications)}), "
                f"elapsed={time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            errors.append({"replication": replication, "error": str(exc), "traceback": tb})
            log(f"replication {replication} FAILED: {exc}\n{tb}")

    cell_df = pd.DataFrame(all_cell_rows)
    matched_df = pd.DataFrame(all_matched_rows)
    eig_df = pd.DataFrame(all_eig_rows)
    world_check_df = pd.DataFrame(all_world_checks)
    baseline_df = pd.DataFrame(all_baselines)

    cell_path = args.outdir / "cell_level_alignment.csv"
    matched_path = args.outdir / "matched_incremental_space_alignment.csv"
    eig_path = args.outdir / "eigenvector_level_alignment.csv"
    world_check_path = args.outdir / "world_check.csv"
    baseline_path = args.outdir / "baseline_overlap.csv"
    cell_df.to_csv(cell_path, index=False)
    matched_df.to_csv(matched_path, index=False)
    eig_df.to_csv(eig_path, index=False)
    world_check_df.to_csv(world_check_path, index=False)
    baseline_df.to_csv(baseline_path, index=False)

    (args.outdir / "stdout_stderr.log").write_text("\n".join(log_lines))

    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory = {"peak_rss_raw": peak_rss_raw, "unit_note": "platform-dependent (KB on Linux, bytes on macOS)"}
    (args.outdir / "peak_memory.json").write_text(json.dumps(peak_memory, indent=2))

    final = dict(started)
    final.update({
        "finished_epoch": time.time(),
        "elapsed_seconds": time.time() - t0,
        "status": "completed" if not errors else "failed",
        "n_replications_planned": len(replications),
        "n_replications_completed": len(replications) - len(errors),
        "errors": errors,
        "expected_cell_rows": len(replications) * len(r_grid) * len(origins),
        "actual_cell_rows": len(cell_df),
        "expected_matched_rows": len(replications) * len(origins),
        "actual_matched_rows": len(matched_df),
        "expected_eig_rows": len(replications) * len(r_grid) * len(origins) * 8,
        "actual_eig_rows": len(eig_df),
        "cell_level_alignment_sha256": sha256_file(cell_path),
        "matched_incremental_space_alignment_sha256": sha256_file(matched_path),
        "eigenvector_level_alignment_sha256": sha256_file(eig_path),
        "world_check_sha256": sha256_file(world_check_path),
        "baseline_overlap_sha256": sha256_file(baseline_path),
        "peak_rss_raw": peak_rss_raw,
    })
    (args.outdir / "manifest_final.json").write_text(json.dumps(final, indent=2, default=str))
    log(f"Stage C run {args.run_id} finished: status={final['status']}, "
        f"cells={len(cell_df)}/{final['expected_cell_rows']}, "
        f"elapsed={final['elapsed_seconds']:.1f}s")
    (args.outdir / "stdout_stderr.log").write_text("\n".join(log_lines))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
