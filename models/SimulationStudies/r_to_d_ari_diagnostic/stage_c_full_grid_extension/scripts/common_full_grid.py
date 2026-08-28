#!/usr/bin/env python3
"""
Shared helpers for the Stage C full-grid extension (r_used = 1,...,9).

Imports, WITHOUT modification:
  - stage_c_subspace_alignment/scripts/common_stage_c.py
      (which itself imports thesis_main_completed_20260816/scripts/common.py
       and the released fnirvar package, also unmodified)

Adds only what the three-point Stage C run did not need:
  - generalized incremental spaces Q_missing_r / Q_extra_r for any r != 5
    (built directly from the already-imported project_out()/nesting_residual_norm())
  - the observable Stage-6-feasibility panel (K-agnostic + K-informed formulas)
  - a from-scratch Holm-Bonferroni step-down correction, with its own unit test
    (no new dependency added solely for this)

Does not modify fnirvar/modeling/, mspe_factor_performance, preliminary_suite,
thesis_main_completed_20260816/, or stage_c_subspace_alignment/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[5] if len(_HERE.parents) > 5 else _HERE.parent
REPO_ROOT = Path(os.environ.get("FNIRVAR_REPO_ROOT", _DEFAULT_ROOT))
STAGE_C_DIR = (
    REPO_ROOT / "models/SimulationStudies/r_to_d_ari_diagnostic/stage_c_subspace_alignment"
)
FORMAL_DIR = (
    REPO_ROOT / "models/SimulationStudies/r_to_d_ari_diagnostic/thesis_main_completed_20260816"
)
sys.path.insert(0, str(STAGE_C_DIR / "scripts"))
# Frozen Stage C helper, imported unchanged.
from common_stage_c import (  # noqa: E402
    FORMAL_DIR as _FORMAL_DIR_CHECK,  # sanity: same path resolved from within common_stage_c
    LEADING_EIGS,
    ORTH_RTOL,
    EMBEDDING_METHOD,
    FIRST_PREDICTION_DAY,
    K_TRUE,
    L_F,
    LOADING_SCALE,
    LOADING_SIGMA,
    LOOKBACK_WINDOW,
    MASTER_SEED,
    N,
    N_FORECAST_ORIGINS,
    R_TRUE,
    T_EVAL,
    FactorAdjustment,
    NIRVAR,
    generate_world,
    independent_mp_check,
    loadings,
    sha256_of_array,
    sha256_of_file,
    structural_seed_for_index,
    orth,
    reconstruct_lambda,
    community_targets,
    finite_sum_stationary_covariance,
    stationary_idiosyncratic_correlation,
    subspace_comparison,
    mp_eigendecomposition,
    nesting_residual_norm,
    project_out,
)

assert _FORMAL_DIR_CHECK == FORMAL_DIR, "FORMAL_DIR path mismatch between modules"

R_GRID = list(range(1, 10))          # 1..9
EXISTING_STAGE_C_R = [3, 5, 7]
NEW_R = [1, 2, 4, 6, 8, 9]
ORIGINS_25 = [
    0, 20, 41, 62, 83, 103, 124, 145, 166, 186,
    207, 228, 249, 269, 290, 311, 332, 352, 373,
    394, 415, 435, 456, 477, 498,
]
ORIGINS_QUALIFICATION = [0, 249, 498]
N_STRUCTURAL_REPLICATIONS = 20
NESTING_TOL = 1e-8

MECHANISM_TARGETS_FOR_UMP_U4 = ["Q_F", "Q_F_unique", "Q_C", "Q_C_unique", "Q_C_full", "Q_P4"]
MECHANISM_TARGETS_FOR_QR = ["Q_F", "Q_C", "Q_C_unique", "Q_C_full", "Q_P4"]
MISSING_TARGETS = ["Q_F", "Q_F_unique", "Q_C", "Q_C_unique", "Q_P4"]
EXTRA_TARGETS = ["Q_C", "Q_C_unique", "Q_C_full", "Q_P4", "Q_F"]


# ---------------------------------------------------------------------------
# Generalized incremental spaces (r != 5)
# ---------------------------------------------------------------------------
def generalized_missing_space(Q_Rr: np.ndarray, Q_R5: np.ndarray) -> np.ndarray:
    """Q_missing_r = orth((I - P_Rr) @ Q_R5), for under-specified r < 5.
    Expected rank 5-r."""
    return project_out(Q_target=Q_R5, Q_remove=Q_Rr)


def generalized_extra_space(Q_Rr: np.ndarray, Q_R5: np.ndarray) -> np.ndarray:
    """Q_extra_r = orth((I - P_R5) @ Q_Rr), for over-specified r > 5.
    Expected rank r-5."""
    return project_out(Q_target=Q_Rr, Q_remove=Q_R5)


def expected_incremental_rank(r: int) -> int:
    if r < 5:
        return 5 - r
    if r > 5:
        return r - 5
    return 0


# ---------------------------------------------------------------------------
# Observable Stage-6-feasibility panel
# ---------------------------------------------------------------------------
def compute_observables(eigvals_desc: np.ndarray, edge: float, d_hat: int,
                         residual: np.ndarray, window: np.ndarray) -> dict:
    """
    K-blind and K-informed observable quantities computable from the candidate
    data, the estimated residual, candidate r, and the residual spectrum only.
    Never uses Q_F, Q_C, Q_P4, true labels, ARI, MSPE, or r_true.
    """
    eigvals_desc = np.asarray(eigvals_desc, dtype=float)
    above_edge = eigvals_desc[eigvals_desc > edge]
    selected_excess_spectral_mass = float(np.sum(above_edge - edge)) if above_edge.size else 0.0

    def eig_or_nan(idx_1based):
        idx = idx_1based - 1
        return float(eigvals_desc[idx]) if 0 <= idx < len(eigvals_desc) else float("nan")

    lambda1 = eig_or_nan(1)
    lambda4 = eig_or_nan(4)
    lambda5 = eig_or_nan(5)
    lambda1_over_edge = lambda1 / edge if edge > 0 else float("nan")
    lambda4_over_edge = lambda4 / edge if edge > 0 and not np.isnan(lambda4) else float("nan")
    lambda5_over_edge = lambda5 / edge if edge > 0 and not np.isnan(lambda5) else float("nan")
    gap_4_5 = (lambda4 - lambda5) if not (np.isnan(lambda4) or np.isnan(lambda5)) else float("nan")

    if d_hat is not None and d_hat > 0 and d_hat < len(eigvals_desc):
        boundary_eigengap = float(eigvals_desc[d_hat - 1] - eigvals_desc[d_hat])
    else:
        boundary_eigengap = float("nan")

    if d_hat is not None and d_hat > 0:
        smallest_selected_margin = float(eigvals_desc[d_hat - 1] - edge)
    else:
        smallest_selected_margin = float("nan")

    if d_hat is not None and d_hat < len(eigvals_desc):
        first_rejected_margin = float(edge - eigvals_desc[d_hat])
    else:
        first_rejected_margin = float("nan")

    window_centered = window - window.mean(axis=0, keepdims=True)
    denom = float(np.sum(window_centered ** 2))
    resid_ss = float(np.sum(residual ** 2))
    residual_variance_ratio = resid_ss / denom if denom > 0 else float("nan")
    removed_variance_fraction = (1.0 - residual_variance_ratio
                                  if not np.isnan(residual_variance_ratio) else float("nan"))

    return {
        "d_hat": int(d_hat) if d_hat is not None else None,
        "selected_excess_spectral_mass": selected_excess_spectral_mass,
        "lambda1_over_edge": lambda1_over_edge,
        "lambda4_over_edge": lambda4_over_edge,
        "lambda5_over_edge": lambda5_over_edge,
        "gap_4_5": gap_4_5,
        "boundary_eigengap": boundary_eigengap,
        "smallest_selected_margin": smallest_selected_margin,
        "first_rejected_margin": first_rejected_margin,
        "residual_variance_ratio": residual_variance_ratio,
        "removed_variance_fraction": removed_variance_fraction,
        "top12_eigenvalues": eigvals_desc[:12].tolist(),
    }


PRIMARY_OBSERVABLES = [
    "d_hat", "selected_excess_spectral_mass", "gap_4_5",
    "lambda4_over_edge", "residual_variance_ratio",
]
K_AGNOSTIC_OBSERVABLES = ["d_hat", "selected_excess_spectral_mass", "residual_variance_ratio"]
K_INFORMED_OBSERVABLES = ["gap_4_5", "lambda4_over_edge"]


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction (implemented directly; no new dependency)
# ---------------------------------------------------------------------------
def holm_adjust(pvalues) -> list:
    """
    Holm (1979) step-down adjustment. Returns adjusted p-values in the ORIGINAL
    input order (standard behaviour matching e.g. R's p.adjust(method="holm")
    and statsmodels' multipletests).
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * ranked[i]
        running_max = max(running_max, val)
        adjusted_ranked[i] = min(running_max, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def _holm_unit_test() -> bool:
    """
    Fixed toy vector, hand-checkable. p = [0.01, 0.02, 0.03, 0.20], m=4.
    Sorted: 0.01,0.02,0.03,0.20 -> multipliers 4,3,2,1
      -> raw: 0.04, 0.06, 0.06, 0.20
      -> step-down max-so-far (already monotone here): 0.04, 0.06, 0.06, 0.20
    Expected adjusted (in original order) = [0.04, 0.06, 0.06, 0.20].
    """
    toy = [0.01, 0.02, 0.03, 0.20]
    expected = [0.04, 0.06, 0.06, 0.20]
    got = holm_adjust(toy)
    ok = all(abs(a - b) < 1e-12 for a, b in zip(got, expected))
    # Second toy vector exercising the running-max (non-monotone raw) case:
    # p = [0.20, 0.01], m=2 -> sorted 0.01,0.20 -> mult 2,1 -> raw 0.02,0.20
    # running max stays 0.02 then 0.20 (already increasing) -> [0.20,0.02] in
    # original order.
    toy2 = [0.20, 0.01]
    expected2 = [0.20, 0.02]
    got2 = holm_adjust(toy2)
    ok2 = all(abs(a - b) < 1e-12 for a, b in zip(got2, expected2))
    # Third: enforce running-max actually matters: p=[0.05, 0.04, 0.5], m=3
    # sorted 0.04,0.05,0.5 -> mult 3,2,1 -> raw 0.12,0.10,0.5
    # running max: 0.12, max(0.12,0.10)=0.12, max(0.12,0.5)=0.5 -> [0.12,0.12,0.5]
    # mapped back to original order [0.05,0.04,0.5] -> idx for 0.04 is rank0 (0.12),
    # idx for 0.05 is rank1 (0.12), idx for 0.5 is rank2 (0.5)
    # original order [0.05,0.04,0.5] -> [0.12, 0.12, 0.5]
    toy3 = [0.05, 0.04, 0.5]
    expected3 = [0.12, 0.12, 0.5]
    got3 = holm_adjust(toy3)
    ok3 = all(abs(a - b) < 1e-12 for a, b in zip(got3, expected3))
    return ok and ok2 and ok3


def paired_t_test(delta: np.ndarray):
    """Two-sided one-sample t-test of delta against 0. Returns (mean, sd, se,
    t_stat, p_value, ci_low, ci_high, n)."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    if n < 2:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd / np.sqrt(n)
    t_stat = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if se > 0 else (0.0 if mean != 0 else 1.0)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return {"mean": mean, "sd": sd, "se": se, "t_stat": float(t_stat), "p_value": p_value,
            "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se, "n": n}


def one_sample_t_interval(values: np.ndarray):
    """95% t interval + two-sided p-value against 0 for a level (not a paired
    difference) series."""
    return paired_t_test(values)
