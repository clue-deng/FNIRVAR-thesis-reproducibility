#!/usr/bin/env python3
"""
Stage F runner: transition map F(r) across the seven approved DGP anchors.

Portable: the repository root is derived from pathlib.Path(__file__); all paths
come through argparse; the caller's working directory is irrelevant.

USAGE:
  python3 run_stage_f.py --run-id formal_reproduction \
      --outdir ../runs/formal_reproduction --config ../configs/stage_f_config.json
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
import sklearn
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_f as C  # noqa: E402

FIELDS = [
    "run_id", "anchor", "family", "p_in", "p_out", "separation", "loading_scale",
    "loading_sigma", "replication", "structural_seed", "network_seed", "factor_seed",
    "gmm_seed", "phi_sha256", "omega_sha256", "lambda_sha256",
    "actual_spectral_radius", "is_stationary", "nominal_density",
    "finite_n_expected_offdiag_density", "realised_offdiag_density",
    "realised_within_block_density", "realised_between_block_density",
    "delta", "delta_over_lambda_max_Gamma_xi", "delta_sign",
    "lambda_max_Gamma_xi", "lambda_min_nonzero_Gamma_chi",
    "origin_index", "K_branch", "r_used", "residual_sha256", "mp_edge",
    "leading_eigenvalues", "d_hat", "d_hat_independent", "ARI", "K_gmm",
    "variant", "baing_branch", "ic_values", "argmin_index", "ic_tie", "r_next",
    "released_pkg_r_hat", "recon_matches_released_pkg", "status", "error_message",
]


def _stub(world, anchor, name, extra=None):
    row = dict.fromkeys(FIELDS, "")
    a = C.ANCHORS[anchor]
    row.update({"anchor": anchor, "family": a["family"], "p_in": a["p_in"],
                "p_out": a["p_out"], "separation": a["separation"],
                "loading_scale": a["loading_scale"], "loading_sigma": C.LOADING_SIGMA})
    for k in ("structural_seed", "network_seed", "factor_seed", "gmm_seed",
              "phi_sha256", "omega_sha256", "lambda_sha256", "actual_spectral_radius",
              "is_stationary", "nominal_density", "finite_n_expected_offdiag_density",
              "realised_offdiag_density", "realised_within_block_density",
              "realised_between_block_density", "delta",
              "delta_over_lambda_max_Gamma_xi", "delta_sign", "lambda_max_Gamma_xi",
              "lambda_min_nonzero_Gamma_chi"):
        if k in world:
            row[k] = world[k]
    row["run_id"] = name
    if extra:
        row.update(extra)
    return row


def run_anchor_world(task):
    anchor, rep, origins, run_id = task
    a = C.ANCHORS[anchor]
    seed = C.structural_seed_for_index(rep)
    try:
        world = C.generate_world(seed, a["p_in"], a["p_out"], a["loading_scale"])
    except Exception as exc:
        return [_stub({"structural_seed": seed}, anchor, run_id,
                      {"replication": rep, "status": "world_generation_failure",
                       "error_message": f"{type(exc).__name__}: {exc}"[:300]})]
    if world.get("generation_status") != "ok":
        return [_stub(world, anchor, run_id,
                      {"replication": rep, "status": world["generation_status"]})]

    X = world["X_full"][:C.T_EVAL]
    rows = []
    for origin_i in origins:
        ws = C.FIRST_PREDICTION_DAY + origin_i - C.LOOKBACK_WINDOW
        window = X[ws:C.FIRST_PREDICTION_DAY + origin_i + 1]
        for K_branch in C.K_BRANCHES:
            for r_used in C.R_GRID:
                base = _stub(world, anchor, run_id,
                             {"replication": rep, "origin_index": origin_i,
                              "K_branch": K_branch, "r_used": r_used})
                try:
                    fa = C.FactorAdjustment(window, r_used, C.L_F)
                    xi = fa.get_idiosyncratic_component()
                    mp = C.independent_mp_check(xi)
                    d_hat, labels, phi_hat, st = C.within_block_phi(
                        xi, K_branch, world["gmm_seed"])
                except Exception as exc:
                    for v in C.VARIANTS:
                        for bb in C.BAING_BRANCHES:
                            rows.append({**base, "variant": v, "baing_branch": bb,
                                         "status": "clustering_failure",
                                         "error_message": f"{type(exc).__name__}: {exc}"[:300]})
                    continue
                base.update({"residual_sha256": C.sha256_of_array(xi),
                             "mp_edge": mp["mp_edge"],
                             "leading_eigenvalues": json.dumps(
                                 [round(x, 12) for x in mp["leading_eigs"]]),
                             "d_hat": d_hat, "d_hat_independent": mp["d_hat_independent"]})
                if st != "ok":
                    for v in C.VARIANTS:
                        for bb in C.BAING_BRANCHES:
                            rows.append({**base, "variant": v, "baing_branch": bb,
                                         "status": "clustering_failure", "error_message": st})
                    continue
                ari = float(adjusted_rand_score(world["true_labels"], labels))
                panels = C.variant_panels(window, xi, phi_hat)
                ic = {"A_incremental": C.baing_ic_array(panels["A_incremental"]),
                      "B_absolute": C.baing_ic_array(panels["B_absolute"]),
                      "C_criterion": C.variant_c_ic(window, phi_hat)}
                K_gmm = C.K_TRUE if K_branch == "primary_fixed_K" else d_hat
                for v, IC in ic.items():
                    dec = C.decide_from_ic(IC)
                    if v == "C_criterion":
                        pkg, matches = {"ok": None, "r_hat": None}, ""
                    else:
                        pkg = C.released_baing_call(panels[v])
                        matches = (pkg["r_hat"] == dec["released"]) if pkg["ok"] else False
                    for bb in C.BAING_BRANCHES:
                        k = dec[bb] if bb == "released" else dec["zero_fixed"]
                        r_next = r_used + k if v == "A_incremental" else k
                        rows.append({**base, "ARI": ari, "K_gmm": K_gmm, "variant": v,
                                     "baing_branch": bb,
                                     "ic_values": json.dumps([round(float(x), 12) for x in IC]),
                                     "argmin_index": dec["argmin_index"],
                                     "ic_tie": dec["tie"], "r_next": r_next,
                                     "released_pkg_r_hat": pkg["r_hat"],
                                     "recon_matches_released_pkg": matches,
                                     "status": "ok", "error_message": ""})
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
    anchors, reps, origins = cfg["anchors"], cfg["replications"], cfg["origins"]

    tasks = [(anc, rep, origins, args.run_id) for anc in anchors for rep in reps]
    man = {
        "run_id": args.run_id, "anchors": anchors, "replications": reps,
        "origins": origins,
        "expected_rows_if_all_valid": len(tasks) * len(origins) * len(C.K_BRANCHES)
        * len(C.R_GRID) * len(C.VARIANTS) * len(C.BAING_BRANCHES),
        "config_sha256": C.sha256_of_file(args.outdir / "config_used.json"),
        "script_sha256": C.sha256_of_file(Path(__file__)),
        "common_sha256": C.sha256_of_file(HERE / "common_stage_f.py"),
        "reused_stage_d_sha256": C.sha256_of_file(HERE / "reused_stage_d.py"),
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "started_epoch": time.time(),
    }
    (args.outdir / "manifest_started.json").write_text(json.dumps(man, indent=2))

    csv_path = args.outdir / "transition_cells.csv"
    t0 = time.perf_counter()
    n = 0
    with open(csv_path, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        if args.workers == 1:
            results = map(run_anchor_world, tasks)
            for i, rows in enumerate(results, 1):
                for row in rows:
                    w.writerow(row)
                    n += 1
                fh.flush()
                print(f"[{i}/{len(tasks)}] rows={n} elapsed={time.perf_counter()-t0:.0f}s",
                      flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for i, rows in enumerate(ex.map(run_anchor_world, tasks), 1):
                    for row in rows:
                        w.writerow(row)
                        n += 1
                    fh.flush()
                    print(f"[{i}/{len(tasks)}] rows={n} elapsed={time.perf_counter()-t0:.0f}s",
                          flush=True)

    man.update({"actual_rows": n, "elapsed_seconds": time.perf_counter() - t0,
                "transition_cells_sha256": C.sha256_of_file(csv_path),
                "peak_rss_kb": __import__("resource").getrusage(
                    __import__("resource").RUSAGE_SELF).ru_maxrss,
                "finished_epoch": time.time(), "status": "completed"})
    (args.outdir / "manifest_final.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: man[k] for k in
                      ["actual_rows", "expected_rows_if_all_valid", "elapsed_seconds",
                       "peak_rss_kb", "platform"]}, indent=2))


if __name__ == "__main__":
    main()
