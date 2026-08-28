#!/usr/bin/env python3
"""
Stage B/M1 runner: population spectrum + branches A-E sample MP computation.

Usage:
  python3 run_stage_b.py --mode qualification --run-id RUNID --worlds 0-4 \
      --out-dir ../runs/RUNID
  python3 run_stage_b.py --mode formal --run-id RUNID --worlds 0-19 \
      --out-dir ../runs/RUNID

Sets the one-thread BLAS/OMP environment before importing NumPy (re-execs
itself if not already set).
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
import common_stage_b as cb  # noqa: E402


def parse_worlds(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def m1_manifest_for_index(idx: int):
    """Only defined for idx in 0..4 -- the 5 frozen M1 worlds."""
    dirs = sorted(cb.M1_DIR.glob("replicate*_seed*"))
    if idx >= len(dirs):
        return None
    d = dirs[idx]
    manifest = json.loads((d / "manifest.json").read_text())
    return {
        "dir": d, "manifest": manifest,
        "phi": np.load(d / "phi.npy"), "omega": np.load(d / "omega.npy"),
    }


def world_check_row(idx: int, world: dict) -> dict:
    row = {
        "world_index": idx, "structural_seed": world["structural_seed"],
        "actual_spectral_radius": world["actual_spectral_radius"],
        "nominal_spectral_radius_multiplier": world["nominal_spectral_radius_multiplier"],
        "is_stationary": world["is_stationary"],
        "phi_sha256": cb.sha256_of_array(world["Phi"]),
        "omega_sha256": cb.sha256_of_array(world["Omega"]),
    }
    m1 = m1_manifest_for_index(idx)
    if m1 is not None:
        row["m1_structural_seed"] = m1["manifest"]["structural_seed"]
        row["m1_structural_seed_match"] = (m1["manifest"]["structural_seed"] == world["structural_seed"])
        row["m1_phi_bit_identical"] = bool(np.array_equal(world["Phi"], m1["phi"]))
        row["m1_omega_bit_identical"] = bool(np.array_equal(world["Omega"], m1["omega"]))
        row["m1_phi_allclose"] = bool(np.allclose(world["Phi"], m1["phi"]))
        row["m1_omega_allclose"] = bool(np.allclose(world["Omega"], m1["omega"]))
    else:
        row["m1_structural_seed"] = None
        row["m1_structural_seed_match"] = None
        row["m1_phi_bit_identical"] = None
        row["m1_omega_bit_identical"] = None
        row["m1_phi_allclose"] = None
        row["m1_omega_allclose"] = None
    return row


def formal_manifest_check_row(idx: int, world: dict) -> dict:
    """Cross-check against thesis_main_completed_20260816 formal-run manifests
    where available (all 20 worlds)."""
    d = (
        cb.FORMAL_DIR / "runs" / "qualification_final_rep0_20260816"
        if idx == 0
        else cb.FORMAL_DIR / "runs" / f"formal_preferred_rep{idx:02d}_20260816"
    )
    mpath = d / "manifest_final.json"
    if not mpath.exists():
        return {"formal_run_manifest_found": False}
    fm = json.loads(mpath.read_text())
    return {
        "formal_run_manifest_found": True,
        "formal_actual_spectral_radius": fm.get("actual_spectral_radius"),
        "formal_radius_close": bool(np.isclose(world["actual_spectral_radius"],
                                                fm.get("actual_spectral_radius", np.nan),
                                                rtol=1e-9, atol=1e-9)),
    }


def process_world(idx: int, run_id: str, prelim_df: pd.DataFrame):
    seed = cb.structural_seed_for_index(idx, cb.N_STRUCTURAL_REPLICATIONS)
    world = cb.generate_world(seed)
    wc_row = world_check_row(idx, world)
    wc_row.update(formal_manifest_check_row(idx, world))

    sample_rows = []
    subset_rows = []
    e_crosscheck_rows = []
    pop_row_out = None

    if not world["is_stationary"]:
        wc_row["status"] = "invalid_unstable_DGP"
        return wc_row, None, sample_rows, subset_rows, e_crosscheck_rows
    wc_row["status"] = "ok"

    pop_row = cb.population_spectrum_row(world["Phi"], world["Omega"])
    Sigma_xi = pop_row.pop("Sigma_xi")
    pop_row_out = {"world_index": idx, "structural_seed": seed, "run_id": run_id, **pop_row}

    X_A = cb.generate_branch_A(seed, Sigma_xi)
    X_B, X_C, X_D = cb.generate_branch_BCD(seed, world["Phi"], world["Omega"], Sigma_xi)
    branch_data = {
        "iid_marginal": X_A, "var_stationary_start": X_B,
        "var_zero_start": X_C, "var_burnin_500": X_D,
    }

    for branch_name, X in branch_data.items():
        for p in range(cb.N_PATHS):
            for T in cb.T_GRID:
                row = cb.sample_mp_row(X[p, :T, :])
                row.update({"run_id": run_id, "world_index": idx, "structural_seed": seed,
                            "branch": branch_name, "path": p})
                sample_rows.append(row)
                if p == 0:  # deterministic subset: path 0 of every (world, branch, T)
                    pkg = cb.package_dhat(X[p, :T, :], world["gmm_seed"])
                    subset_rows.append({"run_id": run_id, "world_index": idx, "branch": branch_name,
                                         "path": p, "T_window": T, "d_hat_independent": row["d_hat"],
                                         "d_hat_package": pkg, "match": pkg == row["d_hat"]})

    # Branch E: released replay validation -- uses generate_world()'s own xi_full,
    # which already performs exactly the "reconstruct GenerateNIRVAR + call
    # generate() once" operation this branch specifies (see DECISIONS.md /
    # IMPLEMENTATION_AUDIT.md for why no second, redundant reconstruction is done).
    xi_full = world["xi_full"]
    for T in cb.T_GRID:
        row = cb.sample_mp_row(xi_full[:T, :])
        row.update({"run_id": run_id, "world_index": idx, "structural_seed": seed,
                    "branch": "released_replay_validation", "path": None})
        sample_rows.append(row)
        pkg = cb.package_dhat(xi_full[:T, :], world["gmm_seed"])
        subset_rows.append({"run_id": run_id, "world_index": idx, "branch": "released_replay_validation",
                             "path": None, "T_window": T, "d_hat_independent": row["d_hat"],
                             "d_hat_package": pkg, "match": pkg == row["d_hat"]})

        # NOTE: use prelim_df["T"] via bracket indexing, NOT prelim_df.T --
        # the latter is pandas' DataFrame.transpose property, not the column
        # named "T", and silently produces a nonsense (all-NaN) filter result.
        prelim_match = prelim_df[(prelim_df.structural_seed == seed) & (prelim_df["T"] == T)]
        if len(prelim_match):
            pm = prelim_match.iloc[0]
            eig_match = [row["leading_eigenvalues"][j] for j in range(min(2, len(row["leading_eigenvalues"])))]
            e_crosscheck_rows.append({
                "world_index": idx, "structural_seed": seed, "T": T,
                "my_d_hat": row["d_hat"], "prelim_d_hat": int(pm["d_hat"]),
                "d_hat_match": row["d_hat"] == int(pm["d_hat"]),
                "my_mp_edge": row["mp_edge"], "prelim_mp_edge": float(pm["MP_edge"]),
                "mp_edge_abs_diff": abs(row["mp_edge"] - float(pm["MP_edge"])),
                "my_eig_1": eig_match[0] if eig_match else float("nan"), "prelim_eig_1": float(pm["eig_1"]),
                "eig_1_abs_diff": abs((eig_match[0] if eig_match else float("nan")) - float(pm["eig_1"])),
                "my_eig_2": eig_match[1] if len(eig_match) > 1 else float("nan"), "prelim_eig_2": float(pm["eig_2"]),
                "eig_2_abs_diff": abs((eig_match[1] if len(eig_match) > 1 else float("nan")) - float(pm["eig_2"])),
            })

    return wc_row, pop_row_out, sample_rows, subset_rows, e_crosscheck_rows


def git_status():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cb.REPO_ROOT,
                             capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=cb.REPO_ROOT,
                                capture_output=True, text=True).stdout.rstrip("\n")
    return commit, porcelain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["qualification", "formal"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--worlds", required=True, help='e.g. "0-4" or "0-19"')
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    worlds = parse_worlds(args.worlds)
    prelim_df = pd.read_csv(cb.PRELIM_DIR / "preliminary_results.csv")
    prelim_df = prelim_df[(prelim_df.experiment == "paired_T_true_residual")
                           & (prelim_df.branch == "mp_fixed_k")]

    t0 = time.time()
    world_check_rows, pop_rows, sample_rows_all, subset_rows_all, e_cross_all = [], [], [], [], []
    errors = []

    for idx in worlds:
        try:
            wc, pop, samples, subset, ecross = process_world(idx, args.run_id, prelim_df)
        except Exception as e:  # pragma: no cover
            errors.append({"world_index": idx, "error": str(e)})
            continue
        world_check_rows.append(wc)
        if pop is not None:
            pop_rows.append(pop)
        sample_rows_all.extend(samples)
        subset_rows_all.extend(subset)
        e_cross_all.extend(ecross)

    world_check_df = pd.DataFrame(world_check_rows)
    pop_df = pd.DataFrame(pop_rows)
    # leading_eigenvalues is a list -> JSON-encode for CSV storage
    sample_df = pd.DataFrame(sample_rows_all)
    if len(sample_df):
        sample_df["leading_eigenvalues"] = sample_df["leading_eigenvalues"].apply(json.dumps)
    subset_df = pd.DataFrame(subset_rows_all)
    ecross_df = pd.DataFrame(e_cross_all)

    world_check_df.to_csv(args.out_dir / "world_check.csv", index=False)
    pop_df.to_csv(args.out_dir / "population_spectra.csv", index=False)
    sample_df.to_csv(args.out_dir / "sample_spectra.csv", index=False)
    subset_df.to_csv(args.out_dir / "package_dhat_subset_check.csv", index=False)
    ecross_df.to_csv(args.out_dir / "branch_e_preliminary_crosscheck.csv", index=False)
    (args.out_dir / "errors.json").write_text(json.dumps(errors, indent=2))

    n_stationary = int((world_check_df["status"] == "ok").sum())
    n_invalid = int((world_check_df["status"] == "invalid_unstable_DGP").sum())
    expected_sample_rows = n_stationary * (len(cb.BRANCHES_50PATH) * cb.N_PATHS * len(cb.T_GRID)
                                            + len(cb.T_GRID))

    config_path = args.out_dir / "config_used.json"
    config_path.write_text(json.dumps({
        "N": cb.N, "K_TRUE": cb.K_TRUE, "R_TRUE": cb.R_TRUE, "MASTER_SEED": cb.MASTER_SEED,
        "T_GRID": cb.T_GRID, "T_MAX": cb.T_MAX, "BURNIN": cb.BURNIN, "N_PATHS": cb.N_PATHS,
        "N_STRUCTURAL_REPLICATIONS": cb.N_STRUCTURAL_REPLICATIONS,
        "BRANCHES_50PATH": cb.BRANCHES_50PATH, "worlds": worlds,
    }, indent=2))

    commit, porcelain = git_status()
    manifest = {
        "run_id": args.run_id, "mode": args.mode, "worlds": worlds,
        "status": "completed" if not errors else "completed_with_errors", "errors": errors,
        "n_worlds_planned": len(worlds), "n_worlds_stationary": n_stationary,
        "n_worlds_invalid_unstable": n_invalid,
        "expected_sample_rows": expected_sample_rows, "actual_sample_rows": len(sample_df),
        "script_sha256": cb.sha256_of_file(HERE / "run_stage_b.py"),
        "common_sha256": cb.sha256_of_file(HERE / "common_stage_b.py"),
        "common_stage_c_sha256": cb.sha256_of_file(cb.STAGE_C_DIR / "scripts" / "common_stage_c.py"),
        "common_formal_sha256": cb.sha256_of_file(cb.FORMAL_DIR / "scripts" / "common.py"),
        "config_sha256": cb.sha256_of_file(config_path),
        "git_commit": commit, "git_status_porcelain": porcelain,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": __import__("scipy").__version__, "scikit_learn": __import__("sklearn").__version__,
        "pandas": pd.__version__, "cpu_count": os.cpu_count(),
        "world_check_sha256": cb.sha256_of_file(args.out_dir / "world_check.csv"),
        "population_spectra_sha256": cb.sha256_of_file(args.out_dir / "population_spectra.csv"),
        "sample_spectra_sha256": cb.sha256_of_file(args.out_dir / "sample_spectra.csv"),
        "package_dhat_subset_check_sha256": cb.sha256_of_file(args.out_dir / "package_dhat_subset_check.csv"),
        "branch_e_preliminary_crosscheck_sha256": cb.sha256_of_file(args.out_dir / "branch_e_preliminary_crosscheck.csv"),
        "elapsed_seconds": time.time() - t0,
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.out_dir / "manifest_final.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("git_status_porcelain", "errors")},
                      indent=2, default=str))


if __name__ == "__main__":
    main()
