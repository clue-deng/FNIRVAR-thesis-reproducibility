#!/usr/bin/env python3
"""
Stage B/M1 shared helpers: seed tree, branch A-E data generators, population
(Lyapunov) and sample (MP) spectrum calculations.

Imports, WITHOUT modification:
  - stage_c_subspace_alignment/scripts/common_stage_c.py
      (stationary_idiosyncratic_correlation, mp_eigendecomposition, orth,
       sha256_of_array/file, and re-exports of the frozen formal helper)
  - thesis_main_completed_20260816/scripts/common.py (via common_stage_c.py)
      (structural_seed_for_index, generate_world, N, K_TRUE, R_TRUE,
       MASTER_SEED, FactorAdjustment, NIRVAR, EMBEDDING_METHOD)

Does not modify fnirvar/modeling/, preliminary_suite/, thesis_main/,
thesis_main_completed_20260816/, stage_c_subspace_alignment/, or
stage_c_full_grid_extension/.
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
DIAG_ROOT = REPO_ROOT / "models/SimulationStudies/r_to_d_ari_diagnostic"
STAGE_C_DIR = DIAG_ROOT / "stage_c_subspace_alignment"
FORMAL_DIR = DIAG_ROOT / "thesis_main_completed_20260816"
PRELIM_DIR = DIAG_ROOT / "preliminary_suite"
M1_DIR = PRELIM_DIR / "mechanism_diagnostics" / "common_inputs" / "M1"

sys.path.insert(0, str(STAGE_C_DIR / "scripts"))
# Frozen Stage C helper, imported unchanged.
from common_stage_c import (  # noqa: E402
    N, K_TRUE, R_TRUE, MASTER_SEED, EMBEDDING_METHOD,
    FactorAdjustment, NIRVAR,
    generate_world, structural_seed_for_index,
    sha256_of_array, sha256_of_file,
    stationary_idiosyncratic_correlation,
    mp_eigendecomposition,
)

N_STRUCTURAL_REPLICATIONS = 20
T_GRID = [200, 500, 1000, 1500, 3000]
T_MAX = 3000
BURNIN = 500
T_MAX_BURNIN = T_MAX + BURNIN  # 3500
N_PATHS = 50
BRANCHES_50PATH = ["iid_marginal", "var_stationary_start", "var_zero_start", "var_burnin_500"]
ALL_BRANCHES = BRANCHES_50PATH + ["released_replay_validation"]
LEADING_EIGS = 8


# ---------------------------------------------------------------------------
# Seed tree (sec 6 of registered design specification / DECISIONS.md)
# ---------------------------------------------------------------------------
def stage_b_children(structural_seed: int):
    """
    SeedSequence(structural_seed).spawn(6); children[0..2] are reserved
    (network/factor/gmm, consumed elsewhere by generate_world(), never here);
    children[3] = branch-A root; children[4] = common VAR root (shared CRN
    for branches B/C/D); children[5] = reserved spare, unused.
    """
    children = np.random.SeedSequence(structural_seed).spawn(6)
    return children


def path_seed_int(seed_seq: np.random.SeedSequence) -> int:
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def branch_a_path_rs(structural_seed: int, path: int) -> np.random.RandomState:
    children = stage_b_children(structural_seed)
    path_child = children[3].spawn(N_PATHS)[path]
    return np.random.RandomState(path_seed_int(path_child))


def branch_bcd_path_children(structural_seed: int, path: int):
    """Returns (x0_rs, eps_rs) for path `path`, shared identically across
    branches B, C, D for that (world, path)."""
    children = stage_b_children(structural_seed)
    path_child = children[4].spawn(N_PATHS)[path]
    x0_eps_children = path_child.spawn(2)
    x0_rs = np.random.RandomState(path_seed_int(x0_eps_children[0]))
    eps_rs = np.random.RandomState(path_seed_int(x0_eps_children[1]))
    return x0_rs, eps_rs


# ---------------------------------------------------------------------------
# Population spectrum (Lyapunov solve, reused unmodified from common_stage_c)
# ---------------------------------------------------------------------------
def population_spectrum_row(Phi: np.ndarray, Omega: np.ndarray) -> dict:
    stat = stationary_idiosyncratic_correlation(Phi, Omega)
    theta = stat["leading_eigs_Corr_P"]
    theta = list(theta) + [float("nan")] * (LEADING_EIGS - len(theta))
    row = {
        "lyap_residual_rel": stat["lyap_residual_rel"],
        "finite_sum_converged": stat["finite_sum_converged"],
        "finite_sum_k_used": stat["k_used"],
        "finite_sum_last_ratio": stat["last_ratio"],
        "sum_vs_lyap_close": stat["sum_vs_lyap_close"],
        "corr_is_finite": stat["is_finite"],
        "corr_is_symmetric": stat["is_symmetric"],
        "corr_positive_diagonal": stat["positive_diagonal"],
        "corr_unit_diagonal": stat["unit_diagonal"],
        "corr_is_psd": stat["is_psd"],
        "corr_min_eig": stat["min_eig"],
    }
    for i in range(LEADING_EIGS):
        row[f"theta_{i+1}"] = theta[i]
    row["theta_2_minus_theta_3"] = theta[1] - theta[2]
    row["theta_4_minus_theta_5"] = theta[3] - theta[4]
    row["theta_3_over_theta_2"] = theta[2] / theta[1] if theta[1] not in (0, None) and not np.isnan(theta[1]) else float("nan")
    row["theta_4_over_theta_2"] = theta[3] / theta[1] if theta[1] not in (0, None) and not np.isnan(theta[1]) else float("nan")
    row["Sigma_xi"] = stat["Sigma_xi"]  # kept in-memory only, not written to any CSV
    return row


def cholesky_or_eigh_sqrt(Sigma: np.ndarray):
    """Cholesky factor of a covariance matrix, with an eigh-based PSD square
    root fallback if Cholesky fails on a matrix that is PSD only up to
    numerical tolerance (never silently regularizes an invalid covariance)."""
    try:
        L = np.linalg.cholesky(Sigma)
        return L, "cholesky"
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals_clipped = np.clip(eigvals, 0, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals_clipped))
        return L, "eigh_sqrt_fallback"


# ---------------------------------------------------------------------------
# Sample MP row (reuses mp_eigendecomposition unmodified)
# ---------------------------------------------------------------------------
def sample_mp_row(X: np.ndarray) -> dict:
    """X: (T, N) window. Returns the executed-MP-rule quantities (sec 7)."""
    mp = mp_eigendecomposition(X)
    eigvals_desc = mp["eigvals_desc"]
    edge = mp["mp_edge"]
    d_hat = mp["d_hat_independent"]
    row = {
        "T_window": X.shape[0],
        "mp_edge": edge,
        "d_hat": d_hat,
        "corr_finite": mp["corr_finite"],
        "corr_symmetric": mp["corr_symmetric"],
        "leading_eigenvalues": mp["leading_eigs"],
    }
    for j in range(1, 7):
        lam = eigvals_desc[j - 1] if len(eigvals_desc) >= j else float("nan")
        row[f"lambda_{j}_minus_edge"] = (lam - edge) if not np.isnan(lam) else float("nan")
        row[f"indicator_dhat_ge_{j}"] = int(d_hat is not None and d_hat >= j)
    return row


def package_dhat(X: np.ndarray, gmm_seed: int) -> int:
    """Construct the released NIRVAR object with d=None and return its
    package-computed d_hat (marchenko_pastur_estimate())."""
    model = NIRVAR(Xi=X, d=None, K=K_TRUE, embedding_method=EMBEDDING_METHOD,
                   gmm_random_int=gmm_seed)
    return int(model.d)


# ---------------------------------------------------------------------------
# Branch data generators
# ---------------------------------------------------------------------------
def generate_branch_A(structural_seed: int, Sigma_xi: np.ndarray) -> np.ndarray:
    """iid_marginal: shape (N_PATHS, T_MAX, N)."""
    L, _ = cholesky_or_eigh_sqrt(Sigma_xi)
    out = np.empty((N_PATHS, T_MAX, N), dtype=np.float64)
    for p in range(N_PATHS):
        rs = branch_a_path_rs(structural_seed, p)
        Z = rs.standard_normal((T_MAX, N))
        out[p] = Z @ L.T
    return out


def generate_branch_BCD(structural_seed: int, Phi: np.ndarray, Omega: np.ndarray,
                         Sigma_xi: np.ndarray):
    """
    Returns (X_B, X_C, X_D): stationary-start, zero-start, and burn-in-500
    trajectories, shapes (N_PATHS, T_MAX, N), (N_PATHS, T_MAX, N),
    (N_PATHS, T_MAX, N) respectively (D already has burn-in discarded).
    All three share the identical per-path epsilon innovation draws (CRN).
    """
    L_sigma, _ = cholesky_or_eigh_sqrt(Sigma_xi)
    omega_diag_sqrt = np.sqrt(np.diag(Omega))

    eps_full = np.empty((N_PATHS, T_MAX_BURNIN, N), dtype=np.float64)
    x0_draws = np.empty((N_PATHS, N), dtype=np.float64)
    for p in range(N_PATHS):
        x0_rs, eps_rs = branch_bcd_path_children(structural_seed, p)
        z0 = x0_rs.standard_normal(N)
        x0_draws[p] = L_sigma @ z0
        eps_full[p] = eps_rs.standard_normal((T_MAX_BURNIN, N)) * omega_diag_sqrt[None, :]

    # Branch B: stationary start, T_MAX steps using eps_full[:, :T_MAX, :]
    state_B = x0_draws.copy()
    traj_B = np.empty((N_PATHS, T_MAX, N), dtype=np.float64)
    for t in range(T_MAX):
        state_B = state_B @ Phi.T + eps_full[:, t, :]
        traj_B[:, t, :] = state_B

    # Branch C: zero start, T_MAX steps, SAME eps_full[:, :T_MAX, :]
    state_C = np.zeros((N_PATHS, N))
    traj_C = np.empty((N_PATHS, T_MAX, N), dtype=np.float64)
    for t in range(T_MAX):
        state_C = state_C @ Phi.T + eps_full[:, t, :]
        traj_C[:, t, :] = state_C

    # Branch D: zero start, T_MAX_BURNIN steps, SAME eps_full (all 3500),
    # retain steps [BURNIN:T_MAX_BURNIN] (the last T_MAX steps).
    state_D = np.zeros((N_PATHS, N))
    traj_D_full = np.empty((N_PATHS, T_MAX_BURNIN, N), dtype=np.float64)
    for t in range(T_MAX_BURNIN):
        state_D = state_D @ Phi.T + eps_full[:, t, :]
        traj_D_full[:, t, :] = state_D
    traj_D = traj_D_full[:, BURNIN:, :]

    return traj_B, traj_C, traj_D


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction and paired t-test (self-contained;
# not imported from stage_c_full_grid_extension/, which is a sibling
# extension, not a natural Stage B dependency -- kept self-contained instead).
# ---------------------------------------------------------------------------
def holm_adjust(pvalues) -> list:
    """Holm (1979) step-down adjustment; returns adjusted p-values in the
    ORIGINAL input order (matches R's p.adjust(method="holm") /
    statsmodels' multipletests)."""
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
    """Fixed toy vectors, hand-checkable (same construction as the analogous
    Stage-C-full-grid unit test)."""
    cases = [
        ([0.01, 0.02, 0.03, 0.20], [0.04, 0.06, 0.06, 0.20]),
        ([0.20, 0.01], [0.20, 0.02]),
        ([0.05, 0.04, 0.5], [0.12, 0.12, 0.5]),
    ]
    ok = True
    for toy, expected in cases:
        got = holm_adjust(toy)
        ok &= all(abs(a - b) < 1e-12 for a, b in zip(got, expected))
    return ok


def paired_t_test(delta) -> dict:
    """Two-sided one-sample t-test of `delta` against 0."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    if n < 2:
        return {"mean": float(delta.mean()) if n else float("nan"), "sd": float("nan"),
                "se": float("nan"), "t_stat": float("nan"), "p_value": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd / np.sqrt(n)
    t_stat = mean / se if se > 0 else (float("inf") if mean != 0 else 0.0)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if se > 0 else (0.0 if mean != 0 else 1.0)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return {"mean": mean, "sd": sd, "se": se, "t_stat": float(t_stat), "p_value": p_value,
            "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se, "n": n}
