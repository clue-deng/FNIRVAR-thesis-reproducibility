#!/usr/bin/env python3
"""
Stage F: DGP signal-boundary experiment.

Adds ONLY a parameterised DGP constructor and the population diagnostics
(Assumption-1 eigengap, densities) on top of the Stage-D feedback machinery,
which is reused byte-identically via `reused_stage_d.py`.

Intervention versus mechanism (DECISIONS.md F-17):
  Family S manipulates COMMUNITY SEPARATION at matched NOMINAL density.
  Family E manipulates LOADING SCALE. The achieved eigengap Delta is an INDUCED
  mechanism variable, never an independently manipulated one. No causal claim
  about the sign of Delta separately from factor strength is admissible.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import reused_stage_d as U
from reused_stage_d import (  # noqa: F401  (verbatim Stage-D machinery)
    GenerateFNIRVAR, GenerateNIRVAR, FactorAdjustment, NIRVAR,
    R_GRID, KMAX_BAING, BAING_JJ, VARIANTS, BAING_BRANCHES, K_BRANCHES,
    MAX_ITER, K_TRUE, R_TRUE, EMBEDDING_METHOD, STOP_STATES,
    baing_ic_array, decide_from_ic, released_baing_call, within_block_phi,
    variant_panels, variant_c_ic, iterate_from_table, child_seed, loadings,
    sha256_of_array, REPO_ROOT,
)

# --- frozen common design (DECISIONS.md section 6 of the execution brief) -----
N = 100
T_FULL = 3000
T_EVAL = 1500
L_F = 2
Q_SHOCKS = 5
RHO_F = 0.7
VAR_SPECTRAL_RADIUS_MULTIPLIER = 0.9
SYMMETRIZE_PHI = False
GLOBAL_NOISE = 1.0
LOADING_SIGMA = 0.1
LOADING_SCALE_CANONICAL = 0.4
LOOKBACK_WINDOW = 1000
FIRST_PREDICTION_DAY = 1000
ORIGINS = (0, 124, 249, 374, 498)
MASTER_SEED = 20260727
CALIBRATION_MASTER_SEED = 20260728
N_CALIBRATION_WORLDS = 5

# --- the seven approved anchors (DECISIONS.md F-11, F-18) ---------------------
ANCHORS = {
    "S0": dict(family="S", p_in=0.90, p_out=0.10, loading_scale=0.40, separation=0.80,
               role="canonical centre; shared reference for Family S and Family E"),
    "S1": dict(family="S", p_in=0.75, p_out=0.15, loading_scale=0.40, separation=0.60, role=""),
    "S2": dict(family="S", p_in=0.60, p_out=0.20, loading_scale=0.40, separation=0.40, role=""),
    "S3": dict(family="S", p_in=0.45, p_out=0.25, loading_scale=0.40, separation=0.20, role=""),
    "S4": dict(family="S", p_in=0.30, p_out=0.30, loading_scale=0.40, separation=0.00,
               role="no-community negative control"),
    "loading_scale_negative_gap": dict(family="E", p_in=0.90, p_out=0.10, loading_scale=0.20,
                                       separation=0.80, role="Family E negative anchor"),
    "loading_scale_positive_gap": dict(family="E", p_in=0.90, p_out=0.10, loading_scale=0.60,
                                       separation=0.80, role="Family E positive anchor"),
}
NOMINAL_DENSITY = 0.30
DENSITY_MARGIN_REL = 0.25   # calibration classification margin, DECISIONS.md F-10


def structural_seed_for_index(index: int, n_total: int = 20) -> int:
    """Byte-identical derivation to Stage D / thesis_main. Never redesigned."""
    return int(np.random.SeedSequence(MASTER_SEED).spawn(n_total)[index]
               .generate_state(1, dtype=np.uint32)[0])


def calibration_seeds(n: int = N_CALIBRATION_WORLDS) -> list:
    return [int(s.generate_state(1, dtype=np.uint32)[0])
            for s in np.random.SeedSequence(CALIBRATION_MASTER_SEED).spawn(n)]


def sha256_of_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- densities (DECISIONS.md F-12, F-19) --------------------------------------
def p_same_block_finite_n(n: int = N, k: int = K_TRUE) -> float:
    """P(same block) over ORDERED OFF-DIAGONAL pairs, equal blocks of n/k."""
    b = n // k
    return (k * b * (b - 1)) / (n * (n - 1))


def nominal_density(p_in: float, p_out: float, k: int = K_TRUE) -> float:
    return (1.0 / k) * p_in + (1.0 - 1.0 / k) * p_out


def finite_n_expected_offdiag_density(p_in: float, p_out: float) -> float:
    ps = p_same_block_finite_n()
    return ps * p_in + (1.0 - ps) * p_out


# --- Assumption 1 (paper, section 3): Delta = lam_min(Gamma_chi) - lam_max(Gamma_xi)
def gamma_F(P: np.ndarray, N0: np.ndarray) -> np.ndarray:
    """Stationary covariance of the factor VAR(l_F) via the companion Lyapunov."""
    lF, r, _ = P.shape
    A = np.zeros((lF * r, lF * r))
    for k in range(lF):
        A[:r, k * r:(k + 1) * r] = P[k]
    if lF > 1:
        A[r:, :-r] = np.eye((lF - 1) * r)
    Qc = np.zeros((lF * r, lF * r))
    Qc[:r, :r] = N0 @ N0.T
    return solve_discrete_lyapunov(A, Qc)[:r, :r]


def generate_world(structural_seed: int, p_in: float, p_out: float,
                   loading_scale: float, loading_sigma: float = LOADING_SIGMA) -> dict:
    """
    Stage-D `generate_world` with (p_in, p_out, loading_scale, loading_sigma)
    exposed. The RNG stream order -- SeedSequence(seed).spawn(3); child 0 network,
    child 1 loadings then factors, child 2 GMM -- is unchanged, and none of the
    exposed parameters alters the NUMBER of draws, so anchors within a
    replication are common-random-number paired.
    """
    children = np.random.SeedSequence(structural_seed).spawn(3)
    network_seed = child_seed(children[0])
    factor_seed = child_seed(children[1])
    network_rs = np.random.RandomState(network_seed)
    factor_rs = np.random.RandomState(factor_seed)
    gmm_seed = int(children[2].generate_state(1, dtype=np.uint32)[0] % (2 ** 31 - 1))

    with contextlib.redirect_stdout(io.StringIO()):
        network = GenerateNIRVAR(
            random_state=network_rs, T=T_FULL, B=K_TRUE, N=N, Q=1,
            p_in=p_in, p_out=p_out, multiplier=VAR_SPECTRAL_RADIUS_MULTIPLIER,
            global_noise=GLOBAL_NOISE, symmetrize_phi=SYMMETRIZE_PHI)
        xi_full = network.generate()[:, :, 0]

    Phi = network.phi_coefficients[:, 0, :, 0].astype(np.float64)
    Omega = np.diag(network.innovations_variance[:, 0]).astype(np.float64)
    rho = float(np.max(np.abs(np.linalg.eigvals(Phi))))
    A = network.adjacency_matrix[:, 0, :, 0]
    off = ~np.eye(N, dtype=bool)
    labels = np.asarray([network.categories[str(i)] for i in range(N)])
    same = (labels[:, None] == labels[None, :]) & off

    out = {
        "structural_seed": structural_seed, "network_seed": network_seed,
        "factor_seed": factor_seed, "gmm_seed": gmm_seed,
        "p_in": p_in, "p_out": p_out,
        "loading_scale": loading_scale, "loading_sigma": loading_sigma,
        "actual_spectral_radius": rho, "is_stationary": bool(rho < 1.0),
        "nominal_density": nominal_density(p_in, p_out),
        "finite_n_expected_offdiag_density": finite_n_expected_offdiag_density(p_in, p_out),
        "realised_offdiag_density": float(A[off].mean()),
        "realised_within_block_density": float(A[same].mean()),
        "realised_between_block_density": float(A[off & ~same].mean()),
        "adjacency_diagonal_fraction": float(np.diag(A).mean()),
        "adjacency_symmetric": bool(np.array_equal(A, A.T)),
        "phi_sha256": sha256_of_array(Phi), "omega_sha256": sha256_of_array(Omega),
        "true_labels": labels, "Phi": Phi, "Omega": Omega,
    }
    if not np.all(np.isfinite(xi_full)) or not out["is_stationary"]:
        out["generation_status"] = ("explosive_xi_non_finite"
                                    if not np.all(np.isfinite(xi_full))
                                    else "invalid_unstable_DGP")
        return out

    Lambda = loading_scale * loadings(N, R_TRUE, loading_sigma, factor_rs)
    with contextlib.redirect_stdout(io.StringIO()):
        factors = GenerateFNIRVAR(l_F=L_F, T=T_FULL, r=R_TRUE, q=Q_SHOCKS,
                                  rho_F=RHO_F, random_state=factor_rs)
        X_full = factors.generate_data(Lambda, xi=xi_full)

    GF = gamma_F(np.asarray(factors.P, float), np.asarray(factors.N0, float))
    Gxi = solve_discrete_lyapunov(Phi, Omega)
    lam_xi = float(np.linalg.eigvalsh(Gxi).max())
    Gchi = Lambda @ GF @ Lambda.T
    lam_chi_min_nonzero = float(np.linalg.eigvalsh(Gchi)[::-1][R_TRUE - 1])
    d_inv = 1.0 / np.sqrt(np.diag(Gxi))
    theta = np.linalg.eigvalsh((Gxi * d_inv[:, None]) * d_inv[None, :])[::-1][:6]

    out.update({
        "generation_status": "ok", "X_full": X_full, "xi_full": xi_full,
        "Lambda": Lambda, "lambda_sha256": sha256_of_array(Lambda),
        "E_Lambda_sq": float(loading_scale ** 2 * (1 + loading_sigma ** 2)),
        "realised_loading_second_moment": float((Lambda ** 2).mean()),
        "lambda_singular_values": np.linalg.svd(Lambda, compute_uv=False).tolist(),
        "lambda_max_Gamma_xi": lam_xi,
        "lambda_min_nonzero_Gamma_chi": lam_chi_min_nonzero,
        "delta": lam_chi_min_nonzero - lam_xi,
        "delta_over_lambda_max_Gamma_xi": (lam_chi_min_nonzero - lam_xi) / lam_xi,
        "delta_sign": int(np.sign(lam_chi_min_nonzero - lam_xi)),
        "delta_margin_qualified_positive":
            bool((lam_chi_min_nonzero - lam_xi) >= DENSITY_MARGIN_REL * lam_xi),
        "delta_margin_qualified_negative":
            bool((lam_chi_min_nonzero - lam_xi) <= -DENSITY_MARGIN_REL * lam_xi),
        "lyap_residual_rel": float(np.linalg.norm(Phi @ Gxi @ Phi.T + Omega - Gxi, "fro")
                                   / np.linalg.norm(Gxi, "fro")),
        "population_corr_theta_1_6": [float(t) for t in theta],
        "mp_edge_lookback": float((1.0 + np.sqrt(N / LOOKBACK_WINDOW)) ** 2),
    })
    return out


def independent_mp_check(residual: np.ndarray) -> dict:
    T_local, N_local = residual.shape
    corr = np.corrcoef(residual.T)
    eig = np.linalg.eigvalsh(corr)[::-1]
    edge = (1.0 + np.sqrt(N_local / T_local)) ** 2
    return {"mp_edge": float(edge), "leading_eigs": eig[:8].tolist(),
            "d_hat_independent": int(np.count_nonzero(eig > edge))}
