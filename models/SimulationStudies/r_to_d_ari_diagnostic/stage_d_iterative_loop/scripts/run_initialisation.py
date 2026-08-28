#!/usr/bin/env python3
"""
Stage D companion: where does the Proposal-Sec-4.2 loop actually START?

Proposal Sec 4.2 initialises at `r_hat_0` from the Bai-Ng criterion applied to
the raw observed window. The transition map alone cannot say whether the loop
works in practice -- that depends on whether the realised initialisation lands
in a basin that reaches r_true. This script records, per (dgp, world, origin):

  * `r0_released`   -- released `baing(window, kmax, jj=2)` (the repository default,
                       algebraically Bai and Ng (2002) IC_p2);
  * `r0_zero_fixed` -- same IC array, zero-factor indexing repaired;
  * `r0_ER`, `r0_GR` -- released eigenvalue-ratio / growth-ratio estimators, as
                       robustness alternatives the FNIRVAR authors also ship.

USAGE: python3 run_initialisation.py --run-id <id> --outdir <dir> --config <json>
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_d as C  # noqa: E402
from train import ER, GR  # noqa: E402

FIELDS = ["run_id", "dgp", "replication", "structural_seed",
          "actual_spectral_radius", "is_stationary", "origin_index",
          "r0_released", "r0_zero_fixed", "r0_argmin_index", "r0_ic_tie",
          "r0_ER", "r0_GR", "status", "error_message"]


def run_world(task):
    dgp_name, p_in, p_out, rep, n_total, origins, run_id = task
    seed = C.structural_seed_for_index(rep, n_total)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            world = C.generate_world(seed, p_in=p_in, p_out=p_out)
    except Exception as exc:
        row = dict.fromkeys(FIELDS, "")
        row.update({"run_id": run_id, "dgp": dgp_name, "replication": rep,
                    "structural_seed": seed, "status": "world_excluded",
                    "error_message": f"{type(exc).__name__}: {exc}"[:200]})
        return [row]
    if world.get("generation_status") != "ok" or not world.get("is_stationary", False):
        generation_status = world.get("generation_status", "unknown_generation_failure")
        row = dict.fromkeys(FIELDS, "")
        row.update({"run_id": run_id, "dgp": dgp_name, "replication": rep,
                    "structural_seed": seed,
                    "actual_spectral_radius": world.get("actual_spectral_radius", ""),
                    "is_stationary": world.get("is_stationary", False),
                    "status": ("invalid_unstable_DGP"
                               if generation_status == "unstable_DGP"
                               else "world_excluded"),
                    "error_message": generation_status})
        return [row]
    X = world["X_full"][: C.T_EVAL]
    rows = []
    for origin_i in origins:
        ws = C.FIRST_PREDICTION_DAY + origin_i - C.LOOKBACK_WINDOW
        window = X[ws: C.FIRST_PREDICTION_DAY + origin_i + 1]
        row = dict.fromkeys(FIELDS, "")
        row.update({"run_id": run_id, "dgp": dgp_name, "replication": rep,
                    "structural_seed": seed,
                    "actual_spectral_radius": world["actual_spectral_radius"],
                    "is_stationary": world["is_stationary"],
                    "origin_index": origin_i})
        try:
            IC = C.baing_ic_array(window)
            dec = C.decide_from_ic(IC)
            pkg = C.released_baing_call(window)
            assert (not pkg["ok"]) or pkg["r_hat"] == dec["released"], "recon mismatch"
            with contextlib.redirect_stdout(io.StringIO()):
                r_er = int(ER(window, C.KMAX_BAING))
                r_gr = int(GR(window, C.KMAX_BAING))
            row.update({"r0_released": dec["released"], "r0_zero_fixed": dec["zero_fixed"],
                        "r0_argmin_index": dec["argmin_index"], "r0_ic_tie": dec["tie"],
                        "r0_ER": r_er, "r0_GR": r_gr, "status": "ok"})
        except Exception as exc:
            row.update({"status": "initialisation_failure",
                        "error_message": f"{type(exc).__name__}: {exc}"[:300]})
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    args = p.parse_args()
    # Run directories are immutable.  Refuse to overwrite an earlier run.
    args.outdir.mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copy2(args.config, args.outdir / "config_used.json")
    cfg = json.loads((args.outdir / "config_used.json").read_text())
    tasks = [(name, s["p_in"], s["p_out"], rep, cfg["n_structural_replications"],
              cfg["origins"], args.run_id)
             for name, s in cfg["dgps"].items()
             for rep in range(cfg["n_structural_replications"])]

    csv_path = args.outdir / "initialisation.csv"
    t0 = time.perf_counter()
    n = 0
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, rows in enumerate(ex.map(run_world, tasks), 1):
                for r in rows:
                    w.writerow(r)
                    n += 1
                fh.flush()
                print(f"[{i}/{len(tasks)}] rows={n} elapsed={time.perf_counter()-t0:.0f}s",
                      flush=True)
    status_counts = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    (args.outdir / "manifest_final.json").write_text(json.dumps({
        "run_id": args.run_id, "actual_rows": n,
        "nominal_rows": len(tasks) * len(cfg["origins"]),
        "status_counts": status_counts,
        "config_sha256": C.sha256_of_file(args.outdir / "config_used.json"),
        "script_sha256": C.sha256_of_file(Path(__file__)),
        "common_sha256": C.sha256_of_file(HERE / "common_stage_d.py"),
        "initialisation_sha256": C.sha256_of_file(csv_path),
        "elapsed_seconds": time.perf_counter() - t0,
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
    }, indent=2))
    print("done", n)


if __name__ == "__main__":
    main()
