#!/usr/bin/env python3
"""Run the provenance-limited pinned-d/pinned-K repository reference cell."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng

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
    R_TRUE,
    T_EVAL,
    FactorAdjustment,
    GenerateFNIRVAR,
    GenerateNIRVAR,
    NIRVAR,
    loadings,
    sha256_of_array,
    sha256_of_file,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--origin-limit", type=int, default=N_FORECAST_ORIGINS)
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.origin_limit <= N_FORECAST_ORIGINS:
        raise ValueError(f"origin-limit must be 1..{N_FORECAST_ORIGINS}")
    args.outdir.mkdir(parents=True, exist_ok=False)

    # Preserve the released scaffold's single Generator and draw order,
    # including its unused phi_dist draw.
    rs = default_rng(seed=4436)
    Lambda = 0.4 * loadings(N, R_TRUE, 0.1, rs)
    _unused_phi_dist = rs.normal(1, 1, size=(N, N))
    network = GenerateNIRVAR(
        random_state=rs,
        T=T_EVAL,
        B=K_TRUE,
        N=N,
        Q=1,
        p_in=0.9,
        p_out=0.1,
        phi_distribution=None,
        multiplier=0.9,
        global_noise=1,
        symmetrize_phi=False,
    )
    xi = network.generate()[:, :, 0]
    factors = GenerateFNIRVAR(
        l_F=L_F,
        T=T_EVAL,
        r=R_TRUE,
        q=5,
        rho_F=0.7,
        random_state=rs,
        P=None,
        N0=None,
    )
    X = factors.generate_data(Lambda=Lambda, xi=xi)
    Phi = network.phi_coefficients[:, 0, :, 0]
    actual_rho = float(np.max(np.abs(np.linalg.eigvals(Phi))))

    manifest = {
        "cell_label": "reconstructed_repository_reference_cell",
        "provenance_warning": (
            "No preserved yaml uniquely matches the stored historical FNIRVAR outputs; "
            "this is a reconstructed reference, not an exact historical reproduction."
        ),
        "seed": 4436,
        "r_used": 5,
        "d": 4,
        "K": 4,
        "origin_limit": args.origin_limit,
        "n_origins_canonical": N_FORECAST_ORIGINS,
        "actual_spectral_radius": actual_rho,
        "is_stationary": actual_rho < 1,
        "phi_sha256": sha256_of_array(Phi),
        "x_sha256": sha256_of_array(X),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "script_sha256": sha256_of_file(Path(__file__)),
        "common_sha256": sha256_of_file(HERE / "common.py"),
        "started_epoch": time.time(),
    }
    (args.outdir / "manifest_started.json").write_text(json.dumps(manifest, indent=2))
    if actual_rho >= 1:
        manifest["status"] = "invalid_unstable_DGP"
        (args.outdir / "manifest_final.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        return

    rows = []
    started = time.perf_counter()
    for origin_i in range(args.origin_limit):
        target_i = FIRST_PREDICTION_DAY + 1 + origin_i
        start = FIRST_PREDICTION_DAY + origin_i - LOOKBACK_WINDOW
        stop = FIRST_PREDICTION_DAY + origin_i + 1
        window = X[start:stop]
        target = X[target_i]
        row = {
            "forecast_origin_index": origin_i,
            "target_index": target_i,
            "squared_error_sum": np.nan,
            "squared_error_denominator": N,
            "origin_status": "unprocessed",
            "error_type": "",
            "error_message": "",
        }
        try:
            factor_model = FactorAdjustment(window, 5, L_F)
            residual = factor_model.get_idiosyncratic_component()
            common_prediction = factor_model.predict_common_component()[:, 0]
            # Suppress package print-spam only; no algorithmic call is changed.
            with contextlib.redirect_stdout(io.StringIO()):
                model = NIRVAR(
                    Xi=residual,
                    embedding_method=EMBEDDING_METHOD,
                    d=K_TRUE,
                    K=K_TRUE,
                )
                xi_prediction = model.predict_idiosyncratic_component()
            prediction = common_prediction + xi_prediction
            row["squared_error_sum"] = float(np.sum((target - prediction) ** 2))
            row["origin_status"] = "ok"
        except Exception as exc:
            row["origin_status"] = "estimation_failure"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)[:1000]
        rows.append(row)
        if args.progress_every > 0 and (
            (origin_i + 1) % args.progress_every == 0 or origin_i + 1 == args.origin_limit
        ):
            print(
                f"reference progress {origin_i + 1}/{args.origin_limit}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    csv_path = args.outdir / "origin_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["origin_status"] == "ok"]
    complete = len(ok) == len(rows)
    mspe = (
        float(sum(r["squared_error_sum"] for r in ok) / (N * len(ok)))
        if complete and ok
        else np.nan
    )
    manifest.update({
        "status": "completed" if complete else "completed_with_failures",
        "n_successful": len(ok),
        "n_failed": len(rows) - len(ok),
        "MSPE_complete": mspe,
        "elapsed_seconds": time.perf_counter() - started,
        "origin_results_sha256": sha256_of_file(csv_path),
        "finished_epoch": time.time(),
    })
    (args.outdir / "manifest_final.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
