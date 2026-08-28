#!/usr/bin/env python3
"""
Stage D runner: build the frozen transition map F(r) = r_next for the
Proposal-Sec-4.2 iterative feedback procedure.

One row per
    (dgp, replication, origin, K_branch, r_used, variant, baing_branch).

Trajectories are NOT simulated here: because the GMM seed is frozen per world
and does not vary with the iteration index (DECISIONS.md D-06), F is a total
deterministic function on R_GRID and every trajectory is a table walk. That walk
is done in `summarize_stage_d.py`.

USAGE:
    python3 run_stage_d.py --run-id <id> --outdir <dir> --config <json>
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_d as C  # noqa: E402

FIELDS = [
    "run_id", "dgp", "p_in", "p_out", "replication", "structural_seed", "gmm_seed",
    "phi_sha256", "omega_sha256", "actual_spectral_radius", "is_stationary",
    "origin_index", "window_start", "window_stop", "K_branch", "r_used",
    "d_hat", "ARI", "variant", "baing_branch", "r_next", "argmin_index",
    "ic_tie", "released_pkg_r_hat", "released_pkg_ok", "released_pkg_error",
    "recon_matches_released_pkg", "status", "error_message",
]


def run_world(task):
    dgp_name, p_in, p_out, rep, n_total, origins, run_id = task
    seed = C.structural_seed_for_index(rep, n_total)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            world = C.generate_world(seed, p_in=p_in, p_out=p_out)
    except Exception as exc:
        row = dict.fromkeys(FIELDS, "")
        row.update({"run_id": run_id, "dgp": dgp_name, "p_in": p_in, "p_out": p_out,
                    "replication": rep, "structural_seed": seed,
                    "status": "world_generation_failure",
                    "error_message": f"{type(exc).__name__}: {exc}"[:300]})
        return [row]
    if world.get("generation_status") != "ok":
        row = dict.fromkeys(FIELDS, "")
        generation_status = world.get("generation_status", "unknown_generation_failure")
        row_status = ("invalid_unstable_DGP" if generation_status == "unstable_DGP"
                      else "excluded_" + generation_status)
        row.update({"run_id": run_id, "dgp": dgp_name, "p_in": p_in, "p_out": p_out,
                    "replication": rep, "structural_seed": seed,
                    "gmm_seed": world["gmm_seed"],
                    "actual_spectral_radius": world["actual_spectral_radius"],
                    "is_stationary": world["is_stationary"],
                    "status": row_status,
                    "error_message": generation_status})
        return [row]
    X = world["X_full"][: C.T_EVAL]
    true_labels = world["true_labels"]
    base = {
        "run_id": run_id, "dgp": dgp_name, "p_in": p_in, "p_out": p_out,
        "replication": rep, "structural_seed": seed, "gmm_seed": world["gmm_seed"],
        "phi_sha256": world["phi_sha256"], "omega_sha256": world["omega_sha256"],
        "actual_spectral_radius": world["actual_spectral_radius"],
        "is_stationary": world["is_stationary"],
    }
    rows = []
    if not world["is_stationary"]:
        row = dict.fromkeys(FIELDS, "")
        row.update(base)
        row["status"] = "invalid_unstable_DGP"
        return [row]

    for origin_i in origins:
        ws = C.FIRST_PREDICTION_DAY + origin_i - C.LOOKBACK_WINDOW
        wstop = C.FIRST_PREDICTION_DAY + origin_i + 1
        window = X[ws:wstop]
        for K_branch in C.K_BRANCHES:
            for r_used in C.R_GRID:
                stub = dict(base)
                stub.update({
                    "origin_index": origin_i, "window_start": ws, "window_stop": wstop,
                    "K_branch": K_branch, "r_used": r_used,
                })
                try:
                    fa = C.FactorAdjustment(window, r_used, C.L_F)
                    xi = fa.get_idiosyncratic_component()
                    d_hat, labels, phi_hat, st = C.within_block_phi(
                        xi, K_branch, world["gmm_seed"]
                    )
                except Exception as exc:
                    for variant in C.VARIANTS:
                        for bb in C.BAING_BRANCHES:
                            row = dict.fromkeys(FIELDS, "")
                            row.update(stub)
                            row.update({"variant": variant, "baing_branch": bb,
                                        "status": "clustering_failure",
                                        "error_message": f"{type(exc).__name__}: {exc}"[:300]})
                            rows.append(row)
                    continue

                if st != "ok":
                    for variant in C.VARIANTS:
                        for bb in C.BAING_BRANCHES:
                            row = dict.fromkeys(FIELDS, "")
                            row.update(stub)
                            row.update({"d_hat": d_hat, "variant": variant,
                                        "baing_branch": bb,
                                        "status": "clustering_failure",
                                        "error_message": st})
                            rows.append(row)
                    continue

                ari = float(adjusted_rand_score(true_labels, labels))
                panels = C.variant_panels(window, xi, phi_hat)
                ic_by_variant = {
                    "A_incremental": C.baing_ic_array(panels["A_incremental"]),
                    "B_absolute": C.baing_ic_array(panels["B_absolute"]),
                    "C_criterion": C.variant_c_ic(window, phi_hat),
                }
                for variant, IC in ic_by_variant.items():
                    dec = C.decide_from_ic(IC)
                    if variant == "C_criterion":
                        pkg = {"ok": None, "r_hat": None, "error": "no released equivalent"}
                        matches = ""
                    else:
                        pkg = C.released_baing_call(panels[variant])
                        matches = (pkg["r_hat"] == dec["released"]) if pkg["ok"] else False
                    for bb in C.BAING_BRANCHES:
                        k = dec[bb] if bb == "released" else dec["zero_fixed"]
                        if variant == "A_incremental":
                            r_next = r_used + k
                        else:
                            r_next = k
                        row = dict.fromkeys(FIELDS, "")
                        row.update(stub)
                        row.update({
                            "d_hat": d_hat, "ARI": ari, "variant": variant,
                            "baing_branch": bb, "r_next": r_next,
                            "argmin_index": dec["argmin_index"], "ic_tie": dec["tie"],
                            "released_pkg_r_hat": pkg["r_hat"],
                            "released_pkg_ok": pkg["ok"],
                            "released_pkg_error": pkg["error"],
                            "recon_matches_released_pkg": matches,
                            "status": "ok", "error_message": "",
                        })
                        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copy2(args.config, args.outdir / "config_used.json")
    cfg = json.loads((args.outdir / "config_used.json").read_text())

    n_total = cfg["n_structural_replications"]
    origins = cfg["origins"]
    dgps = cfg["dgps"]

    tasks = [
        (name, spec["p_in"], spec["p_out"], rep, n_total, origins, args.run_id)
        for name, spec in dgps.items()
        for rep in range(n_total)
    ]

    manifest = {
        "run_id": args.run_id,
        "config_sha256": C.sha256_of_file(args.outdir / "config_used.json"),
        "script_sha256": C.sha256_of_file(Path(__file__)),
        "common_sha256": C.sha256_of_file(HERE / "common_stage_d.py"),
        "n_tasks": len(tasks),
        "expected_rows": len(tasks) * len(origins) * len(C.K_BRANCHES)
        * len(C.R_GRID) * len(C.VARIANTS) * len(C.BAING_BRANCHES),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "started_epoch": time.time(),
    }
    (args.outdir / "manifest_started.json").write_text(json.dumps(manifest, indent=2))

    csv_path = args.outdir / "transition_cells.csv"
    t0 = time.perf_counter()
    n_rows = 0
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        if args.workers == 1:
            results = map(run_world, tasks)
            for i, rows in enumerate(results, 1):
                for row in rows:
                    writer.writerow(row)
                    n_rows += 1
                fh.flush()
                print(f"[{i}/{len(tasks)}] rows={n_rows} elapsed={time.perf_counter()-t0:.0f}s",
                      flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for i, rows in enumerate(ex.map(run_world, tasks), 1):
                    for row in rows:
                        writer.writerow(row)
                        n_rows += 1
                    fh.flush()
                    print(f"[{i}/{len(tasks)}] rows={n_rows} elapsed={time.perf_counter()-t0:.0f}s",
                          flush=True)

    manifest.update({
        "status": "completed" if n_rows == manifest["expected_rows"] else "row_count_mismatch",
        "actual_rows": n_rows,
        "elapsed_seconds": time.perf_counter() - t0,
        "transition_cells_sha256": C.sha256_of_file(csv_path),
        "finished_epoch": time.time(),
    })
    (args.outdir / "manifest_final.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
