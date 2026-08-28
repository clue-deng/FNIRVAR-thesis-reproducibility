#!/usr/bin/env python3
"""
Shared DGP generation, seed derivation, and verification helpers for the
thesis_main controlled imposed-r simulation study.

Imports the checked-out `fnirvar` package. Never modifies it.

Seed derivation, DGP constants and true-label construction are reused
character-for-character from
`../preliminary_suite/preliminary_diagnostics.py` (read-only reference, not
imported directly to keep thesis_main fully standalone and independent of
future edits to the preliminary suite).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_DEFAULT_ROOT = _HERE.parents[5] if len(_HERE.parents) > 5 else _HERE.parent
REPO_ROOT = Path(os.environ.get("FNIRVAR_REPO_ROOT", _DEFAULT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fnirvar" / "modeling"))
from generativeVAR import GenerateFNIRVAR, GenerateNIRVAR  # noqa: E402
from train import FactorAdjustment, NIRVAR  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical thesis DGP (frozen, DECISIONS.md section "Frozen thesis DGP")
# ---------------------------------------------------------------------------
N = 100
T_FULL = 3000
T_EVAL = 1500
R_TRUE = 5
K_TRUE = 4
L_F = 2
Q_SHOCKS = 5
RHO_F = 0.7
P_IN = 0.9
P_OUT = 0.1
VAR_SPECTRAL_RADIUS_MULTIPLIER = 0.9
SYMMETRIZE_PHI = False
GLOBAL_NOISE = 1.0
LOADING_SCALE = 0.4
LOADING_SIGMA = 0.1
EMBEDDING_METHOD = "Pearson Correlation"
LOOKBACK_WINDOW = 1000
FIRST_PREDICTION_DAY = 1000
N_FORECAST_ORIGINS = T_EVAL - FIRST_PREDICTION_DAY - 1  # = 499

MASTER_SEED = 20260727


def random_state(child: np.random.SeedSequence) -> np.random.RandomState:
    return np.random.RandomState(int(child.generate_state(1, dtype=np.uint32)[0]))


def child_seed(child: np.random.SeedSequence) -> int:
    """Materialise the uint32 seed used to construct a child RandomState."""
    return int(child.generate_state(1, dtype=np.uint32)[0])


def structural_seed_for_index(index: int, n_total: int) -> int:
    """
    Structural seed for replication `index` (0-based) out of `n_total`
    requested structural seeds. `SeedSequence.spawn` is index-stable: the
    first 5 children of `spawn(n_total)` are identical to `spawn(5)`'s
    children regardless of n_total, which is what makes the first-5 hash
    gate against frozen M1 meaningful for n_total > 5.
    """
    top_children = np.random.SeedSequence(MASTER_SEED).spawn(n_total)
    return int(top_children[index].generate_state(1, dtype=np.uint32)[0])


def loadings(n: int, r: int, sigma: float, rs: np.random.RandomState) -> np.ndarray:
    signs = rs.choice([-1, 1], size=(n, r))
    noise = rs.normal(loc=0.0, scale=sigma, size=(n, r))
    return signs + noise


def sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def generate_world(structural_seed: int) -> dict:
    """
    Generate one full (T_FULL, N) world from a structural seed, following
    the exact spawn order used by preliminary_suite/preliminary_diagnostics.py
    and preliminary_suite/mechanism_diagnostics/common_inputs/M1/freeze_inputs.py:
        SeedSequence(structural_seed).spawn(3)
            child 0 -> network random state
            child 1 -> shared loadings-and-factor random state
                       (loadings consumed first, then factor generation)
            child 2 -> GMM random state (derived as an int seed, not used here)
    Does NOT reseed internally; generates the network, coefficients,
    innovations, factors, loadings, true residual, observed X and labels
    exactly once.
    """
    children = np.random.SeedSequence(structural_seed).spawn(3)
    network_seed = child_seed(children[0])
    factor_seed = child_seed(children[1])
    network_rs = np.random.RandomState(network_seed)
    factor_rs = np.random.RandomState(factor_seed)
    gmm_seed = int(children[2].generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))

    network = GenerateNIRVAR(
        random_state=network_rs,
        T=T_FULL,
        B=K_TRUE,
        N=N,
        Q=1,
        p_in=P_IN,
        p_out=P_OUT,
        multiplier=VAR_SPECTRAL_RADIUS_MULTIPLIER,
        global_noise=GLOBAL_NOISE,
        symmetrize_phi=SYMMETRIZE_PHI,
    )
    xi_full = network.generate()[:, :, 0]
    # Explicit node-order labels, after confirming category-key type (str),
    # never relying on dict insertion order (registered design sec 11).
    assert all(isinstance(k, str) for k in network.categories.keys()), (
        "network.categories keys are not strings; node-order label "
        "construction assumption violated."
    )
    true_labels = np.asarray([network.categories[str(i)] for i in range(N)])

    Phi = network.phi_coefficients[:, 0, :, 0].astype(np.float64)
    Omega = np.diag(network.innovations_variance[:, 0]).astype(np.float64)
    actual_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(Phi))))
    is_stationary = actual_spectral_radius < 1.0

    Lambda = LOADING_SCALE * loadings(N, R_TRUE, LOADING_SIGMA, factor_rs)
    factors = GenerateFNIRVAR(
        l_F=L_F,
        T=T_FULL,
        r=R_TRUE,
        q=Q_SHOCKS,
        rho_F=RHO_F,
        random_state=factor_rs,
    )
    X_full = factors.generate_data(Lambda, xi=xi_full)

    return {
        "structural_seed": structural_seed,
        "network_seed": network_seed,
        "factor_seed": factor_seed,
        "gmm_seed": gmm_seed,
        "X_full": X_full,
        "xi_full": xi_full,
        "true_labels": true_labels,
        "Phi": Phi,
        "Omega": Omega,
        "nominal_spectral_radius_multiplier": VAR_SPECTRAL_RADIUS_MULTIPLIER,
        "actual_spectral_radius": actual_spectral_radius,
        "is_stationary": is_stationary,
    }


def independent_mp_check(residual: np.ndarray) -> dict:
    """
    Independently verify the released MP selector on a residual (T, N)
    array, per execution-prompt sec 11:
      1. correlation matrix finite and symmetric
      2. symmetric eigenvalues via numpy.linalg.eigvalsh
      3. literal MP edge, released formula
      4. independent count of eigenvalues exceeding the edge
    Returns a dict; does not compare against the package d_hat (caller does
    that once it has constructed the NIRVAR object).
    """
    T_local, N_local = residual.shape
    corr = np.corrcoef(residual.T)
    is_finite = bool(np.all(np.isfinite(corr)))
    is_symmetric = bool(np.allclose(corr, corr.T, atol=1e-10)) if is_finite else False
    if not is_finite:
        return {
            "corr_finite": False,
            "corr_symmetric": is_symmetric,
            "mp_edge": float("nan"),
            "leading_eigs": [],
            "d_hat_independent": None,
        }
    eigvalsh = np.linalg.eigvalsh(corr)[::-1]
    edge = (1.0 + np.sqrt(N_local / T_local)) ** 2
    d_hat_independent = int(np.count_nonzero(eigvalsh > edge))
    return {
        "corr_finite": is_finite,
        "corr_symmetric": is_symmetric,
        "mp_edge": float(edge),
        "leading_eigs": eigvalsh[:8].tolist(),
        "d_hat_independent": d_hat_independent,
    }
