#!/usr/bin/env python3
"""
Stage C full-grid extension runner (r_used = 1,...,9).

Usage:
  python3 run_full_grid_stage_c.py --mode qualification --run-id RUNID \
      --replications 0 --origins 0,249,498 --out-dir ../runs/RUNID
  python3 run_full_grid_stage_c.py --mode formal --run-id RUNID \
      --replications 0-19 --origins 25 --out-dir ../runs/RUNID
  python3 run_full_grid_stage_c.py --mode formal --run-id RUNID \
      --replications 0,5,12 --origins 3 --out-dir ../runs/RUNID   # subset rerun

Computes, for EVERY r in 1..9 on every sampled (replication, origin) cell:
  - the residual, its hash, MP edge, package/independent d_hat (formal
    cross-check available for all r, since the formal thesis_main run already
    covers r_used=1..9)
  - Q_Rr (estimated removed-loading space) and its nesting against adjacent r
  - the observable Stage-6-feasibility panel (all 9 r; written for all 9)
  - full mechanism alignment metrics (U_MP/U_4/Q_R vs Q_F/Q_C/... family) --
    written to new_mechanism_cell_level.csv / new_eigenvector_level.csv only
    for r in NEW_R (1,2,4,6,8,9); r in {3,5,7} mechanism metrics are computed
    too (needed for the qualification cross-check against frozen Stage C
    values) but not re-written as "new" rows in formal mode.
  - generalized Q_missing_r / Q_extra_r for every r != 5, written to
    new_incremental_space_long.csv only for r in NEW_R.

Sets the one-thread BLAS/OMP environment BEFORE importing NumPy (must happen
at process start, so this script re-execs itself with the env vars set if they
are not already present -- see the __main__ guard at the bottom).
"""
from __future__ import annotations

import os
import sys

_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}
if any(os.environ.get(k) != v for k, v in _THREAD_ENV.items()):
    os.environ.update(_THREAD_ENV)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_full_grid as cfg  # noqa: E402

METRIC_SUFFIXES = [
    "dim_U", "dim_Q", "shared_energy", "purity", "capture",
    "expected_random_purity", "expected_random_capture",
    "excess_purity", "excess_capture",
    "largest_canonical_correlation", "smallest_principal_angle",
    "canonical_correlations", "principal_angles",
]


def flatten_comparison(prefix, comp):
    out = {}
    for suf in METRIC_SUFFIXES:
        v = comp[suf]
        if isinstance(v, list):
            v = json.dumps(v)
        out[f"{prefix}_{suf}"] = v
    return out


def formal_raw_rows(replication: int) -> pd.DataFrame:
    d = (
        cfg.FORMAL_DIR / "runs" / "qualification_final_rep0_20260816"
        if replication == 0
        else cfg.FORMAL_DIR / "runs" / f"formal_preferred_rep{replication:02d}_20260816"
    )
    return pd.read_csv(d / "origin_results.csv")


def formal_manifest(replication: int) -> dict:
    d = (
        cfg.FORMAL_DIR / "runs" / "qualification_final_rep0_20260816"
        if replication == 0
        else cfg.FORMAL_DIR / "runs" / f"formal_preferred_rep{replication:02d}_20260816"
    )
    return json.loads((d / "manifest_final.json").read_text())


def build_world_context(replication: int) -> dict:
    seed = cfg.structural_seed_for_index(replication, cfg.N_STRUCTURAL_REPLICATIONS)
    world = cfg.generate_world(seed)
    fm = formal_manifest(replication)

    net_seed_children = np.random.SeedSequence(seed).spawn(3)
    network_seed = int(net_seed_children[0].generate_state(1, dtype=np.uint32)[0])
    factor_seed = int(net_seed_children[1].generate_state(1, dtype=np.uint32)[0])
    gmm_seed = world["gmm_seed"]

    fm_rows = formal_raw_rows(replication)
    fm_primary_first = fm_rows[fm_rows.branch == "primary_fixed_K"].iloc[0]
    world_checks = {
        "replication": replication,
        "structural_seed_match": bool(seed == fm_primary_first["structural_seed"]),
        "network_seed_match": bool(network_seed == fm_primary_first["network_seed"]),
        "factor_seed_match": bool(factor_seed == fm_primary_first["factor_seed"]),
        "gmm_seed_match": bool(gmm_seed == fm_primary_first["gmm_seed"]),
        "actual_spectral_radius_reconstructed": world["actual_spectral_radius"],
        "actual_spectral_radius_formal": float(fm["actual_spectral_radius"]),
        "phi_sha256_reconstructed": cfg.sha256_of_array(world["Phi"]),
        "phi_sha256_formal": fm.get("phi_sha256"),
        "phi_bit_identical": cfg.sha256_of_array(world["Phi"]) == fm.get("phi_sha256"),
        "omega_sha256_reconstructed": cfg.sha256_of_array(world["Omega"]),
        "omega_sha256_formal": fm.get("omega_sha256"),
        "omega_bit_identical": cfg.sha256_of_array(world["Omega"]) == fm.get("omega_sha256"),
        "is_stationary": world["is_stationary"],
    }

    Lambda = cfg.reconstruct_lambda(factor_seed)
    Q_F = cfg.orth(Lambda)
    Z_full, Z_contrast = cfg.community_targets(world["true_labels"])
    Q_C_full = cfg.orth(Z_full)
    Q_C = cfg.orth(Z_contrast)
    Q_F_unique = cfg.project_out(Q_F, Q_C)
    Q_C_unique = cfg.project_out(Q_C, Q_F)
    stat = cfg.stationary_idiosyncratic_correlation(world["Phi"], world["Omega"])
    Q_P4 = stat["Q_P4"]

    targets = {
        "Q_F": Q_F, "Q_C_full": Q_C_full, "Q_C": Q_C,
        "Q_F_unique": Q_F_unique, "Q_C_unique": Q_C_unique, "Q_P4": Q_P4,
    }
    baseline_row = {
        "replication": replication, "structural_seed": seed,
        "rank_Q_F": Q_F.shape[1], "rank_Q_C_full": Q_C_full.shape[1],
        "rank_Q_C": Q_C.shape[1], "rank_Q_F_unique": Q_F_unique.shape[1],
        "rank_Q_C_unique": Q_C_unique.shape[1], "rank_Q_P4": Q_P4.shape[1],
        "lyap_residual_rel": stat["lyap_residual_rel"],
        "sum_vs_lyap_close": stat["sum_vs_lyap_close"],
        "corr_P_is_finite": stat["is_finite"], "corr_P_is_symmetric": stat["is_symmetric"],
        "corr_P_positive_diagonal": stat["positive_diagonal"],
        "corr_P_unit_diagonal": stat["unit_diagonal"], "corr_P_is_psd": stat["is_psd"],
    }
    return {
        "replication": replication, "structural_seed": seed,
        "network_seed": network_seed, "factor_seed": factor_seed, "gmm_seed": gmm_seed,
        "world": world, "targets": targets, "world_check": world_checks,
        "baseline_row": baseline_row, "fm_rows": fm_rows,
    }


def compute_cell(ctx: dict, t: int) -> dict:
    """All 9 r values for one (replication, forecast_origin_index=t) cell."""
    X = ctx["world"]["X_full"][: cfg.T_EVAL]
    window = X[cfg.FIRST_PREDICTION_DAY + t - cfg.LOOKBACK_WINDOW: cfg.FIRST_PREDICTION_DAY + t + 1]
    target_index = cfg.FIRST_PREDICTION_DAY + t + 1

    per_r = {}
    for r in cfg.R_GRID:
        fa = cfg.FactorAdjustment(window, r, cfg.L_F)
        residual = fa.get_idiosyncratic_component()
        loadings_est = fa.loadings()
        Q_Rr = cfg.orth(loadings_est)
        mp = cfg.mp_eigendecomposition(residual)
        pkg_model = cfg.NIRVAR(Xi=residual, d=None, K=cfg.K_TRUE,
                                embedding_method=cfg.EMBEDDING_METHOD,
                                gmm_random_int=ctx["gmm_seed"])
        d_hat_package = int(pkg_model.d)
        obs = cfg.compute_observables(mp["eigvals_desc"], mp["mp_edge"],
                                       mp["d_hat_independent"], residual, window)
        per_r[r] = {
            "residual": residual, "Q_Rr": Q_Rr, "mp": mp,
            "d_hat_package": d_hat_package, "observables": obs,
        }

    nesting_adjacent = {
        r: cfg.nesting_residual_norm(per_r[r]["Q_Rr"], per_r[r + 1]["Q_Rr"])
        for r in range(1, 9)
    }

    incremental = {}
    for r in [1, 2, 3, 4]:
        Qm = cfg.generalized_missing_space(per_r[r]["Q_Rr"], per_r[5]["Q_Rr"])
        incremental[r] = {"space_type": "missing", "Q": Qm,
                           "expected_rank": cfg.expected_incremental_rank(r)}
    for r in [6, 7, 8, 9]:
        Qe = cfg.generalized_extra_space(per_r[r]["Q_Rr"], per_r[5]["Q_Rr"])
        incremental[r] = {"space_type": "extra", "Q": Qe,
                           "expected_rank": cfg.expected_incremental_rank(r)}

    return {"t": t, "target_index": target_index, "T_window": window.shape[0],
            "per_r": per_r, "nesting_adjacent": nesting_adjacent, "incremental": incremental}


def mechanism_row(ctx, cell, r, run_id):
    per_r = cell["per_r"][r]
    residual, Q_Rr, mp = per_r["residual"], per_r["Q_Rr"], per_r["mp"]
    fm_rows = ctx["fm_rows"]
    fm_primary = fm_rows[(fm_rows.branch == "primary_fixed_K") & (fm_rows.r_used == r)
                          & (fm_rows.forecast_origin_index == cell["t"])]
    row = {
        "run_id": run_id, "replication": ctx["replication"], "structural_seed": ctx["structural_seed"],
        "network_seed": ctx["network_seed"], "factor_seed": ctx["factor_seed"], "gmm_seed": ctx["gmm_seed"],
        "r_used": r, "forecast_origin_index": cell["t"], "target_index": cell["target_index"],
        "T_window": cell["T_window"],
        "residual_sha256": cfg.sha256_of_array(residual),
        "mp_edge": mp["mp_edge"], "leading_eigenvalues": json.dumps(mp["leading_eigs"]),
        "d_hat_independent": mp["d_hat_independent"], "d_hat_package": per_r["d_hat_package"],
        "rank_U_MP": mp["U_MP"].shape[1], "rank_U_4": mp["U_4"].shape[1], "rank_Q_R": Q_Rr.shape[1],
    }
    if len(fm_primary):
        f = fm_primary.iloc[0]
        row.update({
            "formal_residual_sha256": f["residual_sha256"], "formal_mp_edge": f["mp_edge"],
            "formal_d_hat_package": f["d_hat_package"], "formal_d_hat_independent": f["d_hat_independent"],
            "formal_ARI": f["ARI"], "formal_squared_error_sum": f["squared_error_sum"],
            "formal_squared_error_denominator": f["squared_error_denominator"],
            "formal_origin_status": f["origin_status"],
            "residual_hash_matches_formal": row["residual_sha256"] == f["residual_sha256"],
            "mp_edge_matches_formal": bool(np.isclose(row["mp_edge"], f["mp_edge"], rtol=1e-10, atol=1e-10)),
            "d_hat_matches_formal": row["d_hat_package"] == f["d_hat_package"],
        })
    else:
        row.update({k: None for k in [
            "formal_residual_sha256", "formal_mp_edge", "formal_d_hat_package",
            "formal_d_hat_independent", "formal_ARI", "formal_squared_error_sum",
            "formal_squared_error_denominator", "formal_origin_status",
            "residual_hash_matches_formal", "mp_edge_matches_formal", "d_hat_matches_formal"]})

    for tname in cfg.MECHANISM_TARGETS_FOR_UMP_U4:
        Q = ctx["targets"][tname]
        row.update(flatten_comparison(f"UMP_{tname}", cfg.subspace_comparison(mp["U_MP"], Q)))
        row.update(flatten_comparison(f"U4_{tname}", cfg.subspace_comparison(mp["U_4"], Q)))
    for tname in cfg.MECHANISM_TARGETS_FOR_QR:
        Q = ctx["targets"][tname]
        row.update(flatten_comparison(f"QR_{tname}", cfg.subspace_comparison(Q_Rr, Q)))
    return row


def eigenvector_rows(ctx, cell, r, run_id):
    per_r = cell["per_r"][r]
    mp = per_r["mp"]
    rows = []
    eigvals = mp["eigvals_desc"]
    corr = np.corrcoef(per_r["residual"].T)
    eigvals_full, eigvecs_full = np.linalg.eigh(corr)
    order = np.argsort(eigvals_full)[::-1]
    eigvecs_desc = eigvecs_full[:, order]
    for rank in range(min(cfg.LEADING_EIGS, len(eigvals))):
        v = eigvecs_desc[:, rank:rank + 1]
        row = {
            "run_id": run_id, "replication": ctx["replication"], "r_used": r,
            "forecast_origin_index": cell["t"], "eigen_rank": rank + 1,
            "eigenvalue": float(eigvals[rank]), "exceeds_mp_edge": bool(eigvals[rank] > mp["mp_edge"]),
        }
        for tname in cfg.MECHANISM_TARGETS_FOR_UMP_U4:
            Q = ctx["targets"][tname]
            if Q.shape[1] > 0:
                proj = float(np.sum((v.T @ Q) ** 2))
            else:
                proj = float("nan")
            row[f"sq_projection_{tname}"] = proj
        rows.append(row)
    return rows


def incremental_row(ctx, cell, r, run_id):
    info = cell["incremental"][r]
    Q = info["Q"]
    space_type = info["space_type"]
    targets_list = cfg.MISSING_TARGETS if space_type == "missing" else cfg.EXTRA_TARGETS
    row = {
        "run_id": run_id, "replication": ctx["replication"], "structural_seed": ctx["structural_seed"],
        "forecast_origin_index": cell["t"], "target_index": cell["target_index"],
        "r_used": r, "space_type": space_type,
        "rank_Q_Rr": cell["per_r"][r]["Q_Rr"].shape[1], "rank_Q_R5": cell["per_r"][5]["Q_Rr"].shape[1],
        "expected_rank": info["expected_rank"], "rank_Qspace": Q.shape[1],
        "rank_ok": Q.shape[1] == info["expected_rank"],
        "nesting_max_all_adjacent": max(cell["nesting_adjacent"].values()),
        "nesting_ok": max(cell["nesting_adjacent"].values()) <= cfg.NESTING_TOL,
    }
    union_targets = ["Q_F", "Q_F_unique", "Q_C", "Q_C_unique", "Q_C_full", "Q_P4"]
    for tname in union_targets:
        if tname in targets_list:
            Q_t = ctx["targets"][tname]
            comp = cfg.subspace_comparison(Q, Q_t)
        else:
            comp = {k: (float("nan") if k not in ("canonical_correlations", "principal_angles")
                        else []) for k in METRIC_SUFFIXES}
            comp["dim_U"] = Q.shape[1]
            comp["dim_Q"] = 0
        row.update(flatten_comparison(f"Qspace_{tname}", comp))
    return row


def observable_row(ctx, cell, r, run_id):
    per_r = cell["per_r"][r]
    obs = per_r["observables"]
    fm_rows = ctx["fm_rows"]
    row = {
        "run_id": run_id, "replication": ctx["replication"], "structural_seed": ctx["structural_seed"],
        "network_seed": ctx["network_seed"], "factor_seed": ctx["factor_seed"], "gmm_seed": ctx["gmm_seed"],
        "r_used": r, "forecast_origin_index": cell["t"], "target_index": cell["target_index"],
        "T_window": cell["T_window"], "residual_sha256": cfg.sha256_of_array(per_r["residual"]),
        "mp_edge": per_r["mp"]["mp_edge"], "d_hat_independent": per_r["mp"]["d_hat_independent"],
        "d_hat_package": per_r["d_hat_package"],
    }
    obs_out = dict(obs)
    obs_out["top12_eigenvalues"] = json.dumps(obs_out["top12_eigenvalues"])
    row.update(obs_out)
    for branch in ["primary_fixed_K", "robustness_K_equals_d_hat"]:
        sub = fm_rows[(fm_rows.branch == branch) & (fm_rows.r_used == r)
                       & (fm_rows.forecast_origin_index == cell["t"])]
        if len(sub):
            f = sub.iloc[0]
            row[f"{branch}_ARI"] = f["ARI"]
            row[f"{branch}_squared_error_sum"] = f["squared_error_sum"]
            row[f"{branch}_squared_error_denominator"] = f["squared_error_denominator"]
            row[f"{branch}_origin_status"] = f["origin_status"]
        else:
            row[f"{branch}_ARI"] = None
            row[f"{branch}_squared_error_sum"] = None
            row[f"{branch}_squared_error_denominator"] = None
            row[f"{branch}_origin_status"] = None
    return row


def parse_replications(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_origins(spec: str):
    if spec.strip() == "25":
        return list(cfg.ORIGINS_25)
    return sorted(set(int(x.strip()) for x in spec.split(",")))


def git_status():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=cfg.REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    return commit, porcelain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["qualification", "formal"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--replications", required=True, help='e.g. "0" or "0-19" or "0,5,12"')
    ap.add_argument("--origins", required=True, help='e.g. "0,249,498" or "25" for all 25')
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    replications = parse_replications(args.replications)
    origins = parse_origins(args.origins)

    t0 = time.time()
    errors = []
    mechanism_rows_out, eig_rows_out, incr_rows_out, obs_rows_out = [], [], [], []
    world_check_rows, baseline_rows = [], []

    for rep in replications:
        try:
            ctx = build_world_context(rep)
        except Exception as e:  # pragma: no cover
            errors.append({"replication": rep, "stage": "world", "error": str(e)})
            continue
        world_check_rows.append(ctx["world_check"])
        baseline_rows.append(ctx["baseline_row"])
        if not ctx["world"]["is_stationary"]:
            errors.append({"replication": rep, "stage": "world", "error": "unstable DGP"})
            continue

        for t in origins:
            try:
                cell = compute_cell(ctx, t)
            except Exception as e:  # pragma: no cover
                errors.append({"replication": rep, "origin": t, "stage": "cell", "error": str(e)})
                continue
            for r in cfg.R_GRID:
                obs_rows_out.append(observable_row(ctx, cell, r, args.run_id))
                if r in cfg.NEW_R:
                    mechanism_rows_out.append(mechanism_row(ctx, cell, r, args.run_id))
                    eig_rows_out.extend(eigenvector_rows(ctx, cell, r, args.run_id))
                if r in cell["incremental"] and r in cfg.NEW_R:
                    incr_rows_out.append(incremental_row(ctx, cell, r, args.run_id))

    cell_df = pd.DataFrame(mechanism_rows_out)
    eig_df = pd.DataFrame(eig_rows_out)
    incr_df = pd.DataFrame(incr_rows_out)
    obs_df = pd.DataFrame(obs_rows_out)
    world_check_df = pd.DataFrame(world_check_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    cell_df.to_csv(args.out_dir / "new_mechanism_cell_level.csv", index=False)
    eig_df.to_csv(args.out_dir / "new_eigenvector_level.csv", index=False)
    incr_df.to_csv(args.out_dir / "new_incremental_space_long.csv", index=False)
    obs_df.to_csv(args.out_dir / "observable_full_grid_cell_level.csv", index=False)
    world_check_df.to_csv(args.out_dir / "world_check.csv", index=False)
    baseline_df.to_csv(args.out_dir / "baseline_overlap.csv", index=False)
    (args.out_dir / "errors.json").write_text(json.dumps(errors, indent=2))

    expected_new_cells = len(replications) * len(cfg.NEW_R) * len(origins)
    expected_obs_cells = len(replications) * len(cfg.R_GRID) * len(origins)
    expected_incr_cells = len(replications) * len(cfg.NEW_R) * len(origins)
    expected_eig_rows = expected_new_cells * cfg.LEADING_EIGS

    config_path = args.out_dir / "config_used.json"
    config_path.write_text(json.dumps({
        "R_GRID": cfg.R_GRID, "NEW_R": cfg.NEW_R, "EXISTING_STAGE_C_R": cfg.EXISTING_STAGE_C_R,
        "N_STRUCTURAL_REPLICATIONS": cfg.N_STRUCTURAL_REPLICATIONS, "ORIGINS_25": cfg.ORIGINS_25,
        "NESTING_TOL": cfg.NESTING_TOL, "MASTER_SEED": cfg.MASTER_SEED,
    }, indent=2))

    commit, porcelain = git_status()
    manifest = {
        "config_sha256": cfg.sha256_of_file(config_path),
        "run_id": args.run_id, "mode": args.mode, "replications": replications, "origins": origins,
        "status": "completed" if not errors else "completed_with_errors",
        "errors": errors,
        "expected_mechanism_rows": expected_new_cells, "actual_mechanism_rows": len(cell_df),
        "expected_eig_rows": expected_eig_rows, "actual_eig_rows": len(eig_df),
        "expected_incremental_rows": expected_incr_cells, "actual_incremental_rows": len(incr_df),
        "expected_observable_rows": expected_obs_cells, "actual_observable_rows": len(obs_df),
        "script_sha256": cfg.sha256_of_file(HERE / "run_full_grid_stage_c.py"),
        "common_sha256": cfg.sha256_of_file(HERE / "common_full_grid.py"),
        "common_stage_c_sha256": cfg.sha256_of_file(cfg.STAGE_C_DIR / "scripts" / "common_stage_c.py"),
        "common_formal_sha256": cfg.sha256_of_file(cfg.FORMAL_DIR / "scripts" / "common.py"),
        "git_commit": commit, "git_status_porcelain": porcelain,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": __import__("scipy").__version__, "scikit_learn": __import__("sklearn").__version__,
        "pandas": pd.__version__, "cpu_count": os.cpu_count(),
        "new_mechanism_cell_level_sha256": cfg.sha256_of_file(args.out_dir / "new_mechanism_cell_level.csv"),
        "new_eigenvector_level_sha256": cfg.sha256_of_file(args.out_dir / "new_eigenvector_level.csv"),
        "new_incremental_space_long_sha256": cfg.sha256_of_file(args.out_dir / "new_incremental_space_long.csv"),
        "observable_full_grid_cell_level_sha256": cfg.sha256_of_file(args.out_dir / "observable_full_grid_cell_level.csv"),
        "world_check_sha256": cfg.sha256_of_file(args.out_dir / "world_check.csv"),
        "baseline_overlap_sha256": cfg.sha256_of_file(args.out_dir / "baseline_overlap.csv"),
        "elapsed_seconds": time.time() - t0,
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.out_dir / "manifest_final.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "git_status_porcelain"}, indent=2, default=str))


if __name__ == "__main__":
    main()
