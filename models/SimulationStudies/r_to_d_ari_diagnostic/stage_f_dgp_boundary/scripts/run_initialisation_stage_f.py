#!/usr/bin/env python3
"""
Stage F initialisation runner -- reachability from the released factor-count
initialisers, adapted from the frozen Stage-D procedure
(stage_d_iterative_loop/scripts/executed_20260819/run_initialisation.py) without
modifying the released package.

The registered estimand includes reachability, so the transition map alone is not
sufficient: `run_stage_f.py` produces F(r); this produces the realised r0 that the
loop would actually start from, per anchor/world/origin.

Portable: repository root from pathlib.Path(__file__); paths via argparse.

USAGE:
  python3 run_initialisation_stage_f.py --run-id formal_reproduction_init \
      --outdir ../runs/formal_reproduction_init --config ../configs/stage_f_config.json
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_f as C  # noqa: E402
from train import ER, GR  # noqa: E402  (released estimators, unmodified)

FIELDS = ["run_id", "anchor", "family", "p_in", "p_out", "loading_scale",
          "loading_sigma", "replication", "structural_seed", "origin_index",
          "r0_released", "r0_zero_fixed", "r0_argmin_index", "r0_ic_tie",
          "r0_ER", "r0_GR", "recon_matches_released_pkg",
          "actual_spectral_radius", "is_stationary", "delta",
          "delta_over_lambda_max_Gamma_xi", "status", "error_message"]


def run_anchor_world(task):
    anchor, rep, origins, run_id = task
    a = C.ANCHORS[anchor]
    seed = C.structural_seed_for_index(rep)
    base = {"run_id": run_id, "anchor": anchor, "family": a["family"],
            "p_in": a["p_in"], "p_out": a["p_out"],
            "loading_scale": a["loading_scale"], "loading_sigma": C.LOADING_SIGMA,
            "replication": rep, "structural_seed": seed}
    try:
        world = C.generate_world(seed, a["p_in"], a["p_out"], a["loading_scale"])
    except Exception as exc:
        row = dict.fromkeys(FIELDS, "")
        row.update(base, status="world_generation_failure",
                   error_message=f"{type(exc).__name__}: {exc}"[:300])
        return [row]
    if world.get("generation_status") != "ok":
        row = dict.fromkeys(FIELDS, "")
        row.update(base, status=world["generation_status"],
                   actual_spectral_radius=world.get("actual_spectral_radius", ""),
                   is_stationary=world.get("is_stationary", ""))
        return [row]

    X = world["X_full"][:C.T_EVAL]
    rows = []
    for origin_i in origins:
        ws = C.FIRST_PREDICTION_DAY + origin_i - C.LOOKBACK_WINDOW
        window = X[ws:C.FIRST_PREDICTION_DAY + origin_i + 1]
        row = dict.fromkeys(FIELDS, "")
        row.update(base, origin_index=origin_i,
                   actual_spectral_radius=world["actual_spectral_radius"],
                   is_stationary=world["is_stationary"], delta=world["delta"],
                   delta_over_lambda_max_Gamma_xi=world["delta_over_lambda_max_Gamma_xi"])
        try:
            IC = C.baing_ic_array(window)
            dec = C.decide_from_ic(IC)
            pkg = C.released_baing_call(window)
            with contextlib.redirect_stdout(io.StringIO()):
                r_er, r_gr = int(ER(window, C.KMAX_BAING)), int(GR(window, C.KMAX_BAING))
            row.update(r0_released=dec["released"], r0_zero_fixed=dec["zero_fixed"],
                       r0_argmin_index=dec["argmin_index"], r0_ic_tie=dec["tie"],
                       r0_ER=r_er, r0_GR=r_gr,
                       recon_matches_released_pkg=(pkg["r_hat"] == dec["released"])
                       if pkg["ok"] else False,
                       status="ok")
        except Exception as exc:
            row.update(status="initialisation_failure",
                       error_message=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    args = p.parse_args()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.outdir / "config_used.json")
    cfg = json.loads((args.outdir / "config_used.json").read_text())
    tasks = [(anc, rep, cfg["origins"], args.run_id)
             for anc in cfg["anchors"] for rep in cfg["replications"]]

    csv_path = args.outdir / "initialisation.csv"
    t0 = time.perf_counter()
    n = 0
    with open(csv_path, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, rows in enumerate(ex.map(run_anchor_world, tasks), 1):
                for r in rows:
                    w.writerow(r)
                    n += 1
                fh.flush()
                print(f"[{i}/{len(tasks)}] rows={n} elapsed={time.perf_counter()-t0:.0f}s",
                      flush=True)
    man = {"run_id": args.run_id, "actual_rows": n,
           "anchors": cfg["anchors"], "replications": cfg["replications"],
           "origins": cfg["origins"],
           "script_sha256": C.sha256_of_file(Path(__file__)),
           "common_sha256": C.sha256_of_file(HERE / "common_stage_f.py"),
           "initialisation_sha256": C.sha256_of_file(csv_path),
           "elapsed_seconds": time.perf_counter() - t0,
           "python": sys.version, "platform": platform.platform(),
           "numpy": np.__version__, "status": "completed"}
    (args.outdir / "manifest_final.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: man[k] for k in ["actual_rows", "elapsed_seconds", "platform"]},
                     indent=2))


if __name__ == "__main__":
    main()
