#!/usr/bin/env python3
"""Standalone imposed-r rolling FNIRVAR experiment; released package is read-only."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    EMBEDDING_METHOD,
    FIRST_PREDICTION_DAY,
    K_TRUE,
    L_F,
    LOOKBACK_WINDOW,
    N,
    N_FORECAST_ORIGINS,
    REPO_ROOT,
    T_EVAL,
    FactorAdjustment,
    NIRVAR,
    generate_world,
    independent_mp_check,
    sha256_of_array,
    sha256_of_file,
    structural_seed_for_index,
)

BRANCHES = ("primary_fixed_K", "robustness_K_equals_d_hat")
LEADING_EIGS = 8

FIELDS = [
    "run_id", "replication", "structural_seed", "network_seed", "factor_seed",
    "gmm_seed", "script_sha256", "common_sha256", "config_sha256", "r_true",
    "r_used", "branch", "T", "N", "lookback_window",
    "forecast_origin_index", "target_index", "n_origins_total",
    "nominal_spectral_radius_multiplier", "actual_spectral_radius",
    "is_stationary", "residual_sha256", "mp_edge", "leading_eigenvalues",
    "d_hat_package", "d_hat_independent", "d_embedding", "K_gmm", "ARI",
    "squared_error_sum", "squared_error_denominator", "origin_status",
    "error_type", "error_message", "elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--replication", type=int, required=True)
    p.add_argument("--n-total-replications", type=int, default=20)
    p.add_argument("--r-grid", required=True, help="Comma-separated integers")
    p.add_argument("--origin-limit", type=int, default=N_FORECAST_ORIGINS)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--config", required=True, type=Path)
    return p.parse_args()


def nan_row_base(args, world, r_used, branch, origin_i, target_i):
    return {
        "run_id": args.run_id,
        "replication": args.replication,
        "structural_seed": world["structural_seed"],
        "network_seed": world["network_seed"],
        "factor_seed": world["factor_seed"],
        "gmm_seed": world["gmm_seed"],
        "script_sha256": world["script_sha256"],
        "common_sha256": world["common_sha256"],
        "config_sha256": world["config_sha256"],
        "r_true": 5,
        "r_used": r_used,
        "branch": branch,
        "T": T_EVAL,
        "N": N,
        "lookback_window": LOOKBACK_WINDOW,
        "forecast_origin_index": origin_i,
        "target_index": target_i,
        "n_origins_total": N_FORECAST_ORIGINS,
        "nominal_spectral_radius_multiplier": world["nominal_spectral_radius_multiplier"],
        "actual_spectral_radius": world["actual_spectral_radius"],
        "is_stationary": world["is_stationary"],
        "residual_sha256": "",
        "mp_edge": np.nan,
        "leading_eigenvalues": "[]",
        "d_hat_package": "",
        "d_hat_independent": "",
        "d_embedding": "",
        "K_gmm": K_TRUE if branch == "primary_fixed_K" else "",
        "ARI": np.nan,
        "squared_error_sum": np.nan,
        "squared_error_denominator": N,
        "origin_status": "unprocessed",
        "error_type": "",
        "error_message": "",
        "elapsed_seconds": np.nan,
    }


def branch_result(model, residual, common_prediction, target, true_labels):
    similarity, labels = model.gmm()
    ari = float(adjusted_rand_score(true_labels, labels))
    phi_hat = model.ols_parameters(similarity)
    xi_prediction = phi_hat @ residual[-1, :]
    prediction = common_prediction + xi_prediction
    if prediction.shape != (N,) or target.shape != (N,):
        raise ValueError(f"shape mismatch prediction={prediction.shape}, target={target.shape}")
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("non-finite prediction")
    squared_error_sum = float(np.sum((target - prediction) ** 2))
    return ari, squared_error_sum


def append_failure(rows, args, world, r_used, origin_i, target_i, status, exc):
    for branch in BRANCHES:
        row = nan_row_base(args, world, r_used, branch, origin_i, target_i)
        row["origin_status"] = status
        row["error_type"] = type(exc).__name__ if exc is not None else status
        row["error_message"] = str(exc)[:1000] if exc is not None else status
        rows.append(row)


def apply_common_values(rows, values):
    """Attach computed residual/MP diagnostics to failure rows when available."""
    for row in rows:
        row.update(values)


def run(args: argparse.Namespace):
    r_grid = [int(x) for x in args.r_grid.split(",") if x.strip()]
    if not r_grid or any(r < 1 for r in r_grid):
        raise ValueError("r-grid must contain positive integers")
    if args.origin_limit < 1 or args.origin_limit > N_FORECAST_ORIGINS:
        raise ValueError(f"origin-limit must be 1..{N_FORECAST_ORIGINS}")

    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    args.outdir.mkdir(parents=True, exist_ok=False)
    config_copy = args.outdir / "config_used.json"
    shutil.copy2(args.config, config_copy)
    script_hash = sha256_of_file(Path(__file__))
    common_hash = sha256_of_file(HERE / "common.py")
    config_hash = sha256_of_file(config_copy)
    config = json.loads(config_copy.read_text())
    expected_config = {
        "master_seed": 20260727,
        "n_structural_replications": args.n_total_replications,
        "N": N,
        "T_evaluation": T_EVAL,
        "r_true": 5,
        "K_true": K_TRUE,
        "l_F": L_F,
        "lookback_window": LOOKBACK_WINDOW,
        "first_prediction_day": FIRST_PREDICTION_DAY,
        "n_forecast_origins": N_FORECAST_ORIGINS,
        "embedding_method": EMBEDDING_METHOD,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"config/constants mismatch: {mismatches}")

    log_path = args.outdir / "stdout_stderr.log"

    def log(message):
        rendered = str(message)
        print(rendered, flush=True)
        with log_path.open("a") as log_handle:
            log_handle.write(rendered + "\n")

    structural_seed = structural_seed_for_index(args.replication, args.n_total_replications)
    generation_stdout = io.StringIO()
    with contextlib.redirect_stdout(generation_stdout):
        world = generate_world(structural_seed)
    if generation_stdout.getvalue().strip():
        log(generation_stdout.getvalue().strip())
    world.update({
        "script_sha256": script_hash,
        "common_sha256": common_hash,
        "config_sha256": config_hash,
    })
    X = world["X_full"][:T_EVAL]
    true_labels = world["true_labels"]

    if (REPO_ROOT / ".git").exists():
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(REPO_ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            git_commit = "unavailable"
            git_status = f"unavailable: {exc}"
    else:
        git_commit = "unavailable"
        git_status = "unavailable: public bundle is not initialised as a Git repository"

    manifest = {
        "run_id": args.run_id,
        "replication": args.replication,
        "structural_seed": structural_seed,
        "seed_tree": {
            "master_seed": 20260727,
            "structural_seed": structural_seed,
            "network_seed": world["network_seed"],
            "factor_seed": world["factor_seed"],
            "gmm_seed": world["gmm_seed"],
        },
        "gmm_seed": world["gmm_seed"],
        "r_grid": r_grid,
        "branches": list(BRANCHES),
        "origin_limit": args.origin_limit,
        "n_origins_canonical": N_FORECAST_ORIGINS,
        "expected_rows": len(r_grid) * len(BRANCHES) * args.origin_limit,
        "actual_spectral_radius": world["actual_spectral_radius"],
        "is_stationary": world["is_stationary"],
        "phi_sha256": sha256_of_array(world["Phi"]),
        "omega_sha256": sha256_of_array(world["Omega"]),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "git_commit": git_commit,
        "git_status_porcelain": git_status,
        "script_sha256": script_hash,
        "common_sha256": common_hash,
        "config_used_sha256": config_hash,
        "started_epoch": time.time(),
    }
    (args.outdir / "manifest_started.json").write_text(json.dumps(manifest, indent=2))

    if not world["is_stationary"]:
        manifest["status"] = "invalid_unstable_DGP"
        (args.outdir / "manifest_final.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    csv_path = args.outdir / "origin_results.csv"
    total_start = time.perf_counter()
    row_count = 0
    status_counts = {}
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for r_used in r_grid:
            for origin_i in range(args.origin_limit):
                cell_start = time.perf_counter()
                target_i = FIRST_PREDICTION_DAY + 1 + origin_i
                window_start = FIRST_PREDICTION_DAY + origin_i - LOOKBACK_WINDOW
                window_stop = FIRST_PREDICTION_DAY + origin_i + 1
                window = X[window_start:window_stop]
                target = X[target_i]
                rows = []
                try:
                    if window.shape != (LOOKBACK_WINDOW + 1, N):
                        raise ValueError(f"window shape {window.shape}")
                    factor_model = FactorAdjustment(window, r_used, L_F)
                    residual = factor_model.get_idiosyncratic_component()
                    common_prediction = factor_model.predict_common_component()[:, 0]
                    residual_hash = sha256_of_array(residual)
                except Exception as exc:
                    append_failure(rows, args, world, r_used, origin_i, target_i,
                                   "factor_adjustment_failure", exc)
                else:
                    mp = independent_mp_check(residual)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            primary = NIRVAR(
                                Xi=residual,
                                d=None,
                                K=K_TRUE,
                                embedding_method=EMBEDDING_METHOD,
                                gmm_random_int=world["gmm_seed"],
                            )
                        d_hat = int(primary.d)
                    except Exception as exc:
                        append_failure(rows, args, world, r_used, origin_i, target_i,
                                       "mp_package_failure", exc)
                    else:
                        common_values = {
                            "residual_sha256": residual_hash,
                            "mp_edge": mp["mp_edge"],
                            "leading_eigenvalues": json.dumps(mp["leading_eigs"]),
                            "d_hat_package": d_hat,
                            "d_hat_independent": mp["d_hat_independent"],
                        }
                        if not mp["corr_finite"] or not mp["corr_symmetric"]:
                            append_failure(rows, args, world, r_used, origin_i, target_i,
                                           "invalid_correlation", None)
                            apply_common_values(rows, common_values)
                        elif d_hat != mp["d_hat_independent"]:
                            append_failure(rows, args, world, r_used, origin_i, target_i,
                                           "mp_count_mismatch", None)
                            apply_common_values(rows, common_values)
                        elif d_hat == 0:
                            for branch in BRANCHES:
                                row = nan_row_base(args, world, r_used, branch, origin_i, target_i)
                                row.update(common_values)
                                row["d_embedding"] = 0
                                row["K_gmm"] = K_TRUE if branch == "primary_fixed_K" else 0
                                row["origin_status"] = "d_hat_zero"
                                row["error_type"] = "d_hat_zero"
                                row["error_message"] = "MP selected zero dimensions; no fallback used"
                                rows.append(row)
                        else:
                            for branch in BRANCHES:
                                row = nan_row_base(args, world, r_used, branch, origin_i, target_i)
                                row.update(common_values)
                                row["d_embedding"] = d_hat
                                if branch == "primary_fixed_K":
                                    model = primary
                                    row["K_gmm"] = K_TRUE
                                else:
                                    with contextlib.redirect_stdout(io.StringIO()):
                                        model = NIRVAR(
                                            Xi=residual,
                                            d=d_hat,
                                            K=d_hat,
                                            embedding_method=EMBEDDING_METHOD,
                                            gmm_random_int=world["gmm_seed"],
                                        )
                                    row["K_gmm"] = d_hat
                                try:
                                    ari, sq_sum = branch_result(
                                        model, residual, common_prediction, target, true_labels
                                    )
                                    row["ARI"] = ari
                                    row["squared_error_sum"] = sq_sum
                                    row["origin_status"] = "ok"
                                except Exception as exc:
                                    row["origin_status"] = "branch_estimation_failure"
                                    row["error_type"] = type(exc).__name__
                                    row["error_message"] = str(exc)[:1000]
                                rows.append(row)

                elapsed = time.perf_counter() - cell_start
                for row in rows:
                    row["elapsed_seconds"] = elapsed
                    writer.writerow(row)
                    row_count += 1
                    s = row["origin_status"]
                    status_counts[s] = status_counts.get(s, 0) + 1
                handle.flush()
                completed_cells = r_grid.index(r_used) * args.origin_limit + origin_i + 1
                total_cells = len(r_grid) * args.origin_limit
                if args.progress_every > 0 and (
                    completed_cells % args.progress_every == 0 or completed_cells == total_cells
                ):
                    elapsed_so_far = time.perf_counter() - total_start
                    log(
                        f"progress {completed_cells}/{total_cells} r-origin cells "
                        f"({100 * completed_cells / total_cells:.1f}%), "
                        f"elapsed={elapsed_so_far:.1f}s"
                    )

    elapsed_total = time.perf_counter() - total_start
    manifest.update({
        "status": "completed",
        "actual_rows": row_count,
        "status_counts": status_counts,
        "elapsed_seconds": elapsed_total,
        "seconds_per_origin_r_cell": elapsed_total / (len(r_grid) * args.origin_limit),
        "origin_results_sha256": sha256_of_file(csv_path),
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "finished_epoch": time.time(),
    })
    if row_count != manifest["expected_rows"]:
        manifest["status"] = "row_count_mismatch"
    (args.outdir / "manifest_final.json").write_text(json.dumps(manifest, indent=2))
    (args.outdir / "peak_memory.json").write_text(json.dumps({
        "ru_maxrss_raw": manifest["peak_rss_raw"],
        "platform_note": "macOS reports ru_maxrss in bytes; preserve raw value for auditability",
    }, indent=2))
    log(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    run(parse_args())
