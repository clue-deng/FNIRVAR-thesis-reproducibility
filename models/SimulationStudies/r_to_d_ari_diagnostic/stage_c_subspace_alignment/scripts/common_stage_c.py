#!/usr/bin/env python3
"""
Stage C shared helpers: subspace targets, orth(), Lyapunov-solved stationary
idiosyncratic correlation, and alignment metrics.

Imports (unchanged) the frozen `thesis_main_completed_20260816/scripts/common.py`
helper for DGP generation and seed derivation, and the released `fnirvar`
package. Modifies neither. Does not modify `fnirvar/modeling/`, the formal run,
or the preliminary suite.

Generator-recursion orientation (registered design sec 4, item 1): the executed
line (t_distribution=False branch, generativeVAR.py:624) is

    X = np.sum(np.sum(self.phi_coefficients*X,axis=2),axis=2) + Z

With X shape (N,Q) broadcasting against phi_coefficients shape (N,Q,N,Q) and
both axis=2 reductions collapsing the trailing (j,qj) axes:
    X_new[i,qi] = sum_{j,qj} Phi[i,qi,j,qj] * X_old[j,qj]
With Q=1 and Phi = phi_coefficients[:,0,:,0], this is the standard column-vector
recursion x_t = Phi @ x_{t-1} + z_t (NOT Phi.T @ x). The Lyapunov equation below
therefore uses A = Phi, untransposed. See reports/IMPLEMENTATION_TRACE.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

_HERE = Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[5] if len(_HERE.parents) > 5 else _HERE.parent
REPO_ROOT = Path(os.environ.get("FNIRVAR_REPO_ROOT", _DEFAULT_ROOT))
FORMAL_DIR = (
    REPO_ROOT
    / "models/SimulationStudies/r_to_d_ari_diagnostic/thesis_main_completed_20260816"
)
sys.path.insert(0, str(FORMAL_DIR / "scripts"))
# Frozen formal helper, imported unchanged (registered design sec 4).
from common import (  # noqa: E402
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
)

LEADING_EIGS = 8
ORTH_RTOL = 1e-10


# ---------------------------------------------------------------------------
# orth() and Lambda reconstruction
# ---------------------------------------------------------------------------
def orth(A: np.ndarray, rtol: float = ORTH_RTOL) -> np.ndarray:
    """SVD-based orthonormal basis for the column space of A, fixed relative
    tolerance `rtol` against the largest singular value."""
    A = np.atleast_2d(A)
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], 0))
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    if S.size == 0 or S[0] <= 0:
        return np.zeros((A.shape[0], 0))
    tol = rtol * S[0]
    rank = int(np.count_nonzero(S > tol))
    return U[:, :rank]


def reconstruct_lambda(factor_seed: int) -> np.ndarray:
    """
    Reconstruct Lambda without touching the frozen helper: a fresh
    RandomState(factor_seed), one call to loadings(N, R_TRUE, LOADING_SIGMA, rs),
    scaled by LOADING_SCALE, with no other draw consumed first -- identical to
    the first draws generate_world() makes from the same factor_seed.
    """
    rs = np.random.RandomState(factor_seed)
    return LOADING_SCALE * loadings(N, R_TRUE, LOADING_SIGMA, rs)


# ---------------------------------------------------------------------------
# Community-indicator targets
# ---------------------------------------------------------------------------
def community_targets(true_labels: np.ndarray):
    labels = np.asarray(true_labels)
    classes = np.sort(np.unique(labels))
    assert len(classes) == K_TRUE, f"expected {K_TRUE} classes, got {len(classes)}"
    Z_full = np.zeros((N, K_TRUE))
    for col, c in enumerate(classes):
        Z_full[labels == c, col] = 1.0
    centering = np.eye(N) - np.ones((N, N)) / N
    Z_contrast = centering @ Z_full
    return Z_full, Z_contrast


# ---------------------------------------------------------------------------
# Stationary idiosyncratic correlation (population, dynamics-aware target)
# ---------------------------------------------------------------------------
def finite_sum_stationary_covariance(A: np.ndarray, Omega: np.ndarray,
                                      rtol: float = 1e-12, k_max: int = 10000):
    """
    Independent cross-check of solve_discrete_lyapunov: truncated finite sum
    Sigma_sum = sum_{k=0}^{K} A^k Omega (A^k)^T, increasing K until the next
    term's Frobenius norm / accumulated-sum Frobenius norm < rtol (hard cap
    k_max). Returns (Sigma_sum, k_used, last_ratio, converged).
    """
    term = Omega.astype(np.float64).copy()
    total = term.copy()
    k_used = 0
    ratio = np.inf
    converged = False
    for k in range(1, k_max + 1):
        term = A @ term @ A.T
        total_norm = np.linalg.norm(total, ord="fro")
        term_norm = np.linalg.norm(term, ord="fro")
        ratio = term_norm / total_norm if total_norm > 0 else np.inf
        total = total + term
        k_used = k
        if ratio < rtol:
            converged = True
            break
    return total, k_used, float(ratio), converged


def stationary_idiosyncratic_correlation(Phi: np.ndarray, Omega: np.ndarray) -> dict:
    """
    Solve Sigma_xi = Phi @ Sigma_xi @ Phi.T + Omega (A = Phi, column-vector
    convention -- see module docstring), cross-check with an independent
    truncated finite sum, convert to correlation, and extract the top-K_TRUE
    eigenbasis Q_P4.
    """
    A = Phi
    Sigma_lyap = solve_discrete_lyapunov(A, Omega)
    Sigma_sum, k_used, last_ratio, converged = finite_sum_stationary_covariance(A, Omega)

    sum_vs_lyap_close = bool(np.allclose(Sigma_sum, Sigma_lyap, rtol=1e-10, atol=1e-10))
    residual = A @ Sigma_lyap @ A.T + Omega - Sigma_lyap
    lyap_residual_rel = float(
        np.linalg.norm(residual, "fro") / max(np.linalg.norm(Sigma_lyap, "fro"), 1e-300)
    )

    diag = np.diag(Sigma_lyap)
    is_finite = bool(np.all(np.isfinite(Sigma_lyap)))
    is_symmetric = bool(np.allclose(Sigma_lyap, Sigma_lyap.T, atol=1e-8))
    positive_diagonal = bool(np.all(diag > 0)) if is_finite else False

    if is_finite and positive_diagonal:
        d_inv = 1.0 / np.sqrt(diag)
        Corr_P = (Sigma_lyap * d_inv[:, None]) * d_inv[None, :]
        unit_diagonal = bool(np.allclose(np.diag(Corr_P), 1.0, atol=1e-8))
        eigvals_full = np.linalg.eigvalsh(Corr_P)
        min_eig = float(eigvals_full.min())
        is_psd = bool(min_eig > -1e-8)
        eigvals_desc = eigvals_full[::-1]
        eigvecs_desc = np.linalg.eigh(Corr_P)[1][:, ::-1]
        Q_P4 = eigvecs_desc[:, :K_TRUE]
        rank_ok = Q_P4.shape[1] == K_TRUE
    else:
        Corr_P = np.full_like(Sigma_lyap, np.nan)
        unit_diagonal = False
        is_psd = False
        min_eig = float("nan")
        eigvals_desc = np.full(Sigma_lyap.shape[0], np.nan)
        Q_P4 = np.zeros((Sigma_lyap.shape[0], 0))
        rank_ok = False

    return {
        "Sigma_xi": Sigma_lyap,
        "Sigma_sum": Sigma_sum,
        "k_used": k_used,
        "last_ratio": last_ratio,
        "finite_sum_converged": converged,
        "sum_vs_lyap_close": sum_vs_lyap_close,
        "lyap_residual_rel": lyap_residual_rel,
        "Corr_P": Corr_P,
        "is_finite": is_finite,
        "is_symmetric": is_symmetric,
        "positive_diagonal": positive_diagonal,
        "unit_diagonal": unit_diagonal,
        "is_psd": is_psd,
        "min_eig": min_eig,
        "leading_eigs_Corr_P": eigvals_desc[:LEADING_EIGS].tolist(),
        "Q_P4": Q_P4,
        "Q_P4_rank_ok": rank_ok,
    }


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------
def subspace_comparison(U: np.ndarray, Q: np.ndarray, n: int = N) -> dict:
    """
    purity/capture/canonical-correlation/principal-angle metrics for
    orthonormal bases U (estimated) and Q (target), plus isotropic
    random-subspace excess-* references.
    """
    dU = 0 if U is None else U.shape[1]
    dQ = 0 if Q is None else Q.shape[1]

    if dU > 0 and dQ > 0:
        M = U.T @ Q
        sv = np.linalg.svd(M, compute_uv=False)
        shared_energy = float(np.sum(sv**2))
    else:
        sv = np.array([])
        shared_energy = 0.0

    purity = (shared_energy / dU) if dU > 0 else float("nan")
    capture = (shared_energy / dQ) if dQ > 0 else float("nan")

    expected_random_purity = dQ / n
    expected_random_capture = dU / n
    excess_purity = (purity - expected_random_purity) if not np.isnan(purity) else float("nan")
    excess_capture = (capture - expected_random_capture) if not np.isnan(capture) else float("nan")

    principal_angles = np.arccos(np.clip(sv, -1.0, 1.0)).tolist() if sv.size else []
    largest_canonical_correlation = float(sv[0]) if sv.size else float("nan")
    smallest_principal_angle = float(principal_angles[0]) if principal_angles else float("nan")

    return {
        "dim_U": dU,
        "dim_Q": dQ,
        "shared_energy": shared_energy,
        "purity": purity,
        "capture": capture,
        "expected_random_purity": expected_random_purity,
        "expected_random_capture": expected_random_capture,
        "excess_purity": excess_purity,
        "excess_capture": excess_capture,
        "canonical_correlations": sv.tolist(),
        "largest_canonical_correlation": largest_canonical_correlation,
        "principal_angles": principal_angles,
        "smallest_principal_angle": smallest_principal_angle,
    }


def mp_eigendecomposition(residual: np.ndarray) -> dict:
    """
    Full symmetric eigendecomposition of the exact N x N Pearson correlation
    matrix used by the released MP selector (registered design sec 5): eigh,
    descending sort, literal MP edge, U_MP (eigenvectors with eigenvalue >
    edge) and U_4 (top four eigenvectors, independent of d_hat).
    """
    T_local, N_local = residual.shape
    corr = np.corrcoef(residual.T)
    is_finite = bool(np.all(np.isfinite(corr)))
    is_symmetric = bool(np.allclose(corr, corr.T, atol=1e-10)) if is_finite else False
    if not is_finite:
        return {
            "corr_finite": False, "corr_symmetric": is_symmetric,
            "mp_edge": float("nan"), "eigvals_desc": np.array([]),
            "leading_eigs": [], "d_hat_independent": None,
            "U_MP": np.zeros((N_local, 0)), "U_4": np.zeros((N_local, 0)),
        }
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals_desc = eigvals[order]
    eigvecs_desc = eigvecs[:, order]
    edge = (1.0 + np.sqrt(N_local / T_local)) ** 2
    d_hat_independent = int(np.count_nonzero(eigvals_desc > edge))
    U_MP = eigvecs_desc[:, :d_hat_independent]
    U_4 = eigvecs_desc[:, :4]
    return {
        "corr_finite": is_finite,
        "corr_symmetric": is_symmetric,
        "mp_edge": float(edge),
        "eigvals_desc": eigvals_desc,
        "leading_eigs": eigvals_desc[:LEADING_EIGS].tolist(),
        "d_hat_independent": d_hat_independent,
        "U_MP": U_MP,
        "U_4": U_4,
    }


def nesting_residual_norm(Q_small: np.ndarray, Q_big: np.ndarray) -> float:
    """||(I - Q_big Q_big.T) @ Q_small||_F -- 0 iff Q_small's column space is
    contained in Q_big's."""
    n = Q_small.shape[0]
    if Q_small.shape[1] == 0:
        return 0.0
    if Q_big.shape[1] == 0:
        return float(np.linalg.norm(Q_small, "fro"))
    proj_perp = np.eye(n) - Q_big @ Q_big.T
    return float(np.linalg.norm(proj_perp @ Q_small, "fro"))


def project_out(Q_target: np.ndarray, Q_remove: np.ndarray, rtol: float = ORTH_RTOL) -> np.ndarray:
    """orth((I - Q_remove Q_remove.T) @ Q_target)."""
    n = Q_target.shape[0]
    if Q_remove.shape[1] == 0:
        return orth(Q_target, rtol)
    proj_perp = np.eye(n) - Q_remove @ Q_remove.T
    return orth(proj_perp @ Q_target, rtol)
