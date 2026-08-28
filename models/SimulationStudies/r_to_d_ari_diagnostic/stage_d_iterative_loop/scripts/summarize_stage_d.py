#!/usr/bin/env python3
"""
Stage D summariser: walk the frozen transition map into trajectories, classify
stop states, and do replication-level inference.

Inference rules inherited from thesis_main (DECISIONS.md D-08):
  * the independent Monte Carlo unit is the STRUCTURAL WORLD;
  * origins are repeated measurements inside a world and are averaged first;
  * across-world summaries use Student-t intervals with n_worlds - 1 df.

USAGE: python3 summarize_stage_d.py --run <runs/formal_...> --outdir <results/>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_stage_d as C  # noqa: E402

CELL_KEYS = ["dgp", "replication", "origin_index", "K_branch", "variant", "baing_branch"]


def mean_ci(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "mcse": np.nan,
                "ci_low": np.nan, "ci_high": np.nan}
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    mcse = sd / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and sd > 0:
        half = stats.t.ppf(0.975, n - 1) * mcse
    else:
        half = 0.0
    return {"n": n, "mean": m, "sd": sd, "mcse": mcse,
            "ci_low": m - half, "ci_high": m + half}


def build_tables(df: pd.DataFrame) -> dict:
    tables = {}
    for key, g in df.groupby(CELL_KEYS, sort=False):
        F = {}
        for _, row in g.iterrows():
            r = int(row["r_used"])
            if row["status"] != "ok" or pd.isna(row["r_next"]):
                F[r] = {"next": None, "status": str(row["status"])}
            else:
                F[r] = {"next": int(row["r_next"]), "status": "ok"}
        tables[key] = F
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--init-run", type=Path, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.run / "transition_cells.csv")
    tables = build_tables(df)

    # ---------------- trajectory level ----------------
    traj_rows = []
    for key, F in tables.items():
        dgp, rep, origin, K_branch, variant, baing_branch = key
        for r0 in C.R_GRID:
            out = C.iterate_from_table(F, r0)
            traj_rows.append({
                "dgp": dgp, "replication": rep, "origin_index": origin,
                "K_branch": K_branch, "variant": variant, "baing_branch": baing_branch,
                "r0": r0, "stop_state": out["stop_state"], "n_iter": out["n_iter"],
                "terminal_r": out["terminal_r"],
                "reached_r_true": int(out["stop_state"] == "correct_fixed_point"),
                "trajectory": "->".join(str(v) for v in out["trajectory"]),
            })
    traj = pd.DataFrame(traj_rows)
    traj.to_csv(args.outdir / "trajectories.csv", index=False)

    # ---------------- within-world, then across-world ----------------
    group = ["dgp", "K_branch", "variant", "baing_branch", "r0"]
    within = (traj.groupby(group + ["replication"], as_index=False)
                  .agg(reached=("reached_r_true", "mean"),
                       n_iter=("n_iter", "mean")))
    within.to_csv(args.outdir / "within_world_summary.csv", index=False)

    across_rows = []
    for key, g in within.groupby(group, sort=False):
        ci = mean_ci(g["reached"].values)
        it = mean_ci(g["n_iter"].values)
        across_rows.append(dict(zip(group, key), n_worlds=ci["n"],
                                P_reach_r_true=ci["mean"], sd=ci["sd"], mcse=ci["mcse"],
                                ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                                mean_n_iter=it["mean"]))
    across = pd.DataFrame(across_rows)
    across.to_csv(args.outdir / "basin_summary.csv", index=False)

    # ---------------- stop-state distribution ----------------
    stop = (traj.groupby(["dgp", "K_branch", "variant", "baing_branch", "r0", "stop_state"])
                .size().rename("count").reset_index())
    stop.to_csv(args.outdir / "stop_state_counts.csv", index=False)

    # ---------------- mean transition map ----------------
    ok = df[df["status"] == "ok"].copy()
    tmap = (ok.groupby(["dgp", "K_branch", "variant", "baing_branch", "r_used"])
              .agg(mean_r_next=("r_next", "mean"),
                   modal_r_next=("r_next", lambda s: int(s.mode().iloc[0])),
                   p_next_eq_r_true=("r_next", lambda s: float((s == C.R_TRUE).mean())),
                   mean_d_hat=("d_hat", "mean"), mean_ARI=("ARI", "mean"), n=("r_next", "size"))
              .reset_index())
    tmap.to_csv(args.outdir / "transition_map.csv", index=False)

    # ---------------- validation gates ----------------
    non_c = df[(df["variant"] != "C_criterion") & (df["status"] == "ok")]
    stable_worlds = (df[df["status"] == "ok"][["dgp", "replication"]]
                     .drop_duplicates().sort_values(["dgp", "replication"]))
    excluded_worlds = (df[df["status"] != "ok"]
                       [["dgp", "replication", "status"]].drop_duplicates())
    gates = {
        "rows_total": int(len(df)),
        "all_status_ok": bool((df["status"] == "ok").all()),
        "status_counts": df["status"].value_counts().to_dict(),
        "all_included_worlds_stationary": bool(
            df.loc[df["status"] == "ok", "is_stationary"]
              .astype(str).str.lower().eq("true").all()),
        "n_stable_worlds_per_dgp": {k: int(v) for k, v in
                                    stable_worlds.groupby("dgp")["replication"].nunique().items()},
        "excluded_worlds": excluded_worlds.to_dict(orient="records"),
        "reconstruction_rows_checked": int(len(non_c)),
        "recon_matches_released_baing_rate":
            float(non_c["recon_matches_released_pkg"].astype(str).str.lower().eq("true").mean()),
        "n_ic_ties": int(df["ic_tie"].astype(str).str.lower().eq("true").sum()),
        "transition_map_total_on_grid": bool(
            all(set(F.keys()) == set(C.R_GRID) for F in tables.values())),
        "n_transition_tables": len(tables),
    }
    if args.init_run is not None and (args.init_run / "initialisation.csv").is_file():
        ini = pd.read_csv(args.init_run / "initialisation.csv")
        ini.to_csv(args.outdir / "initialisation.csv", index=False)
        gates["initialisation_rows"] = int(len(ini))
        gates["initialisation_status_counts"] = ini["status"].value_counts().to_dict()

    # ---------------- end-to-end: compose realised initialiser with the basin ----
    if args.init_run is not None and (args.init_run / "initialisation.csv").is_file():
        ini = pd.read_csv(args.init_run / "initialisation.csv")
        ini_ok = ini[ini["status"] == "ok"].copy()
        matched = ini_ok.merge(stable_worlds.assign(_stable_world=True),
                               on=["dgp", "replication"], how="left",
                               validate="many_to_one")
        unmatched = matched[matched["_stable_world"].isna()].copy()
        ini = matched[matched["_stable_world"].eq(True)].drop(columns="_stable_world")
        ini.to_csv(args.outdir / "initialisation_stable_worlds.csv", index=False)
        gates["initialisation_ok_rows"] = int(len(ini_ok))
        gates["initialisation_matched_stable_rows"] = int(len(ini))
        gates["initialisation_unmatched_ok_rows"] = int(len(unmatched))
        gates["initialisation_unmatched_worlds"] = (
            unmatched[["dgp", "replication"]].drop_duplicates().to_dict(orient="records"))
        e2e_rows = []
        for col in ["r0_released", "r0_zero_fixed", "r0_ER", "r0_GR"]:
            sub = ini[["dgp", "replication", "origin_index", col]].rename(columns={col: "r0"})
            sub["r0"] = sub["r0"].astype(int)
            j = traj.merge(sub, on=["dgp", "replication", "origin_index", "r0"], how="inner")
            j["initialiser"] = col.replace("r0_", "")
            e2e_rows.append(j)
        e2e = pd.concat(e2e_rows, ignore_index=True)
        e2e.to_csv(args.outdir / "end_to_end_trajectories.csv", index=False)
        expected_e2e_rows = len(ini) * 4 * len(C.K_BRANCHES) * len(C.VARIANTS) * len(C.BAING_BRANCHES)
        gates["end_to_end_expected_rows"] = int(expected_e2e_rows)
        gates["end_to_end_row_count_matches"] = bool(len(e2e) == expected_e2e_rows)
        g2 = ["dgp", "K_branch", "variant", "baing_branch", "initialiser"]
        w2 = (e2e.groupby(g2 + ["replication"], as_index=False)
                  .agg(reached=("reached_r_true", "mean"),
                       mean_r0=("r0", "mean"), n_iter=("n_iter", "mean")))
        rows2 = []
        for key, g in w2.groupby(g2, sort=False):
            ci = mean_ci(g["reached"].values)
            rows2.append(dict(zip(g2, key), n_worlds=ci["n"], P_reach_r_true=ci["mean"],
                              sd=ci["sd"], mcse=ci["mcse"], ci_low=ci["ci_low"],
                              ci_high=ci["ci_high"], mean_r0=float(g["mean_r0"].mean())))
        pd.DataFrame(rows2).to_csv(args.outdir / "end_to_end_summary.csv", index=False)
        gates["end_to_end_rows"] = int(len(e2e))
        gates["initialiser_r0_distribution"] = {
            dgp: {c: {int(k): int(v) for k, v in
                      ini[ini.dgp == dgp][c].value_counts().sort_index().items()}
                  for c in ["r0_released", "r0_zero_fixed", "r0_ER", "r0_GR"]}
            for dgp in sorted(ini.dgp.unique())}

    (args.outdir / "STAGE_D_VALIDATION.json").write_text(json.dumps(gates, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
