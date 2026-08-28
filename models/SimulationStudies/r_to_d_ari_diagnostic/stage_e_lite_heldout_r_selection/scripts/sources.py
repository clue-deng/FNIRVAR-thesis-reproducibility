#!/usr/bin/env python3
"""
Frozen source manifest for Stage E-lite.

IMPORTANT NAMING IRREGULARITY. A single `formal_preferred_rep*` glob matches
only replications 1-19 and silently omits replication 0. Replication 0 of the
published formal grid lives under a `qualification_*` run id, and there are TWO
rep-0 directories with different content. The published formal summary used
run_id `qualification_final_20260816`.

This module hard-codes the exact 20 sources and provides the assertions that
prove the mapping against the frozen within-replication summary.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = (REPO_ROOT / "models" / "SimulationStudies" / "r_to_d_ari_diagnostic"
            / "thesis_main_completed_20260816")
RUNS = SRC_ROOT / "runs"
WITHIN_SUMMARY = SRC_ROOT / "formal_results_20260816" / "within_replication_summary.csv"

REP0_PATH = RUNS / "qualification_final_rep0_20260816" / "origin_results.csv"
REP0_RUN_ID = "qualification_final_20260816"
REP1_19_RUN_ID = "formal_preferred_20260816"

FORBIDDEN = [
    RUNS / "qualification_preferred_rep0_20260816" / "origin_results.csv",
]

N_WORLDS = 20
R_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BRANCHES = ["primary_fixed_K", "robustness_K_equals_d_hat"]
N_ORIGINS = 499
ROWS_PER_WORLD = 8982


def source_paths() -> dict:
    paths = {0: REP0_PATH}
    for rep in range(1, 20):
        paths[rep] = RUNS / f"formal_preferred_rep{rep:02d}_20260816" / "origin_results.csv"
    return paths


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_manifest() -> dict:
    """Fail closed on any manifest inconsistency. Returns the audit record."""
    rec = {"checks": {}, "sources": {}}

    w = pd.read_csv(WITHIN_SUMMARY)
    rec["checks"]["within_summary_rows_360"] = (len(w) == 360)
    rec["checks"]["within_summary_replications_0_19"] = (
        sorted(w.replication.unique().tolist()) == list(range(20)))
    rec["checks"]["rep0_run_id_is_qualification_final"] = (
        w.loc[w.replication == 0, "run_id"].unique().tolist() == [REP0_RUN_ID])
    rec["checks"]["rep1_19_run_id_is_formal_preferred"] = (
        w.loc[w.replication != 0, "run_id"].unique().tolist() == [REP1_19_RUN_ID])

    paths = source_paths()
    rec["checks"]["all_20_sources_exist"] = all(p.is_file() for p in paths.values())
    rec["checks"]["forbidden_sources_not_used"] = all(
        Path(p) not in set(paths.values()) for p in FORBIDDEN)

    for rep, p in sorted(paths.items()):
        rec["sources"][str(rep)] = {
            "path": str(p.relative_to(REPO_ROOT)),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
    if not all(rec["checks"].values()):
        failed = [k for k, v in rec["checks"].items() if not v]
        raise AssertionError(f"source manifest checks failed: {failed}")
    return rec


def load_sources() -> pd.DataFrame:
    frames = []
    for rep, p in sorted(source_paths().items()):
        d = pd.read_csv(p)
        present = sorted(d.replication.unique().tolist())
        if present != [rep]:
            raise AssertionError(f"{p} contains replications {present}, expected [{rep}]")
        if len(d) != ROWS_PER_WORLD:
            raise AssertionError(f"{p} has {len(d)} rows, expected {ROWS_PER_WORLD}")
        if sorted(d.r_used.unique().tolist()) != R_GRID:
            raise AssertionError(f"{p} r_used mismatch")
        if sorted(d.branch.unique().tolist()) != sorted(BRANCHES):
            raise AssertionError(f"{p} branch mismatch")
        if d.forecast_origin_index.nunique() != N_ORIGINS:
            raise AssertionError(f"{p} origin count mismatch")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    if sorted(out.replication.unique().tolist()) != list(range(N_WORLDS)):
        raise AssertionError("combined source does not contain exactly worlds 0..19")
    return out
