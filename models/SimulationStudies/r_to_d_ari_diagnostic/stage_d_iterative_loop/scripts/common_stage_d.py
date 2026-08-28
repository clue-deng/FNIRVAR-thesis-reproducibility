#!/usr/bin/env python3
"""
Shared machinery for Stage D: the proposal-Sec-4.2 iterative feedback map.

Design rules inherited from thesis_main / stage_c (see ../DECISIONS.md):
  * imports the checked-out `fnirvar` package read-only, never modifies it;
  * DGP generation, seed derivation and true labels are reused
    character-for-character from
    `thesis_main_completed_20260816/scripts/common.py`;
  * the inferential unit is the structural world, never a forecast origin.

New to Stage D:
  * three frozen operationalisations of "within-block residual" (Sec 2 of
    DECISIONS.md), because Proposal Sec 4.2 does not determine one uniquely;
  * two Bai-Ng branches, `released` and `zero_fixed`, because the released
    `baing()` cannot return 0 (Sec 3 of DECISIONS.md).
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
from train import FactorAdjustment, NIRVAR, baing as released_baing  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical thesis DGP -- identical to thesis_main/scripts/common.py
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

# Stage D frozen constants
R_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9)
KMAX_BAING = 10
BAING_JJ = 2
VARIANTS = ("A_incremental", "B_absolute", "C_criterion")
BAING_BRANCHES = ("released", "zero_fixed")
K_BRANCHES = ("primary_fixed_K", "robustness_K_equals_d_hat")
MAX_ITER = 20

DGPS = {
    "strong_sbm": {"p_in": 0.9, "p_out": 0.1},
    "weak_sbm": {"p_in": 0.6, "p_out": 0.4},
}


# ---------------------------------------------------------------------------
# Seed derivation -- byte-identical to thesis_main/scripts/common.py
# ---------------------------------------------------------------------------
def child_seed(child: np.random.SeedSequence) -> int:
    return int(child.generate_state(1, dtype=np.uint32)[0])


def structural_seed_for_index(index: int, n_total: int) -> int:
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


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def generate_world(structural_seed: int, p_in: float = P_IN, p_out: float = P_OUT) -> dict:
    """
    Identical to thesis_main/scripts/common.py:generate_world() when called with
    the canonical (p_in, p_out) = (0.9, 0.1); the two block probabilities are the
    only Stage D parameterisation, added for the pre-specified weak-SBM anchor.
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
        p_in=p_in,
        p_out=p_out,
        multiplier=VAR_SPECTRAL_RADIUS_MULTIPLIER,
        global_noise=GLOBAL_NOISE,
        symmetrize_phi=SYMMETRIZE_PHI,
    )
    xi_full = network.generate()[:, :, 0]
    # `GenerateNIRVAR.phi` rescales by `np.max(phi_eigs)`, which on a complex
    # spectrum is the lexicographic (real, imag) max and NOT the modulus max
    # (documented as DGP-001 in thesis_main/reports/IMPLEMENTATION_AUDIT.md sec 4).
    # The nominal multiplier 0.9 is therefore not a guaranteed spectral radius, and
    # a realised radius >= 1 gives an explosive trajectory. Detect that here rather
    # than letting a non-finite array propagate into the factor step.
    if not np.all(np.isfinite(xi_full)):
        return {
            "structural_seed": structural_seed,
            "network_seed": network_seed,
            "factor_seed": factor_seed,
            "gmm_seed": gmm_seed,
            "generation_status": "explosive_xi_non_finite",
            "is_stationary": False,
            "actual_spectral_radius": float(
                np.max(np.abs(np.linalg.eigvals(
                    network.phi_coefficients[:, 0, :, 0].astype(np.float64))))
            ),
            "phi_sha256": "", "omega_sha256": "",
        }
    assert all(isinstance(k, str) for k in network.categories.keys())
    true_labels = np.asarray([network.categories[str(i)] for i in range(N)])

    Phi = network.phi_coefficients[:, 0, :, 0].astype(np.float64)
    Omega = np.diag(network.innovations_variance[:, 0]).astype(np.float64)
    actual_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(Phi))))

    # Stop before factor generation when the released generator has produced an
    # unstable VAR.  In the executed 2026-08-19 source this check occurred only
    # in the caller, after factor generation; a sufficiently explosive but still
    # finite xi path could therefore throw and be mislabelled as a generic world
    # generation failure.  The executed source is preserved under
    # scripts/executed_20260819/.
    if actual_spectral_radius >= 1.0:
        return {
            "structural_seed": structural_seed,
            "network_seed": network_seed,
            "factor_seed": factor_seed,
            "gmm_seed": gmm_seed,
            "generation_status": "unstable_DGP",
            "is_stationary": False,
            "actual_spectral_radius": actual_spectral_radius,
            "phi_sha256": sha256_of_array(Phi),
            "omega_sha256": sha256_of_array(Omega),
        }

    Lambda = LOADING_SCALE * loadings(N, R_TRUE, LOADING_SIGMA, factor_rs)
    factors = GenerateFNIRVAR(
        l_F=L_F, T=T_FULL, r=R_TRUE, q=Q_SHOCKS, rho_F=RHO_F, random_state=factor_rs
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
        "phi_sha256": sha256_of_array(Phi),
        "omega_sha256": sha256_of_array(Omega),
        "generation_status": "ok",
        "actual_spectral_radius": actual_spectral_radius,
        "is_stationary": actual_spectral_radius < 1.0,
    }


# ---------------------------------------------------------------------------
# Bai-Ng: released call, independent IC reconstruction, zero-fixed decision
# ---------------------------------------------------------------------------
def baing_ic_array(X: np.ndarray, kmax: int = KMAX_BAING, jj: int = BAING_JJ) -> np.ndarray:
    """
    Reconstruct the released `baing()` information-criterion array independently.

    Returns IC of length kmax+1 where IC[i] (i = 0..kmax-1) is the criterion for
    (i+1) factors and IC[kmax] is the criterion for ZERO factors -- exactly the
    released indexing convention (train.py:235-366).

    The penalty for jj=2 is  k * (N+T)/(NT) * log(min(N,T))  applied to log V(k),
    i.e. Bai and Ng (2002) IC_p2. The released docstring labels it PC_p2; the
    implemented penalty is IC_p2's. See ../reports/BAING_AUDIT.md.
    """
    T, n = X.shape
    NT = n * T
    NT1 = n + T
    ii = np.arange(1, kmax + 1)
    GCT = min(n, T)
    if jj == 1:
        CT = np.log(NT / NT1) * ii * (NT1 / NT)
    elif jj == 2:
        CT = np.log(min(n, T)) * ii * (NT1 / NT)
    elif jj == 3:
        CT = np.log(GCT) / GCT * ii
    else:
        raise ValueError("jj must be 1, 2 or 3")

    if T < n:
        ev, _, _ = np.linalg.svd(X @ X.T)
        Fhat0 = ev * np.sqrt(T)
        Lambda0 = X.T @ Fhat0 / T
    else:
        ev, _, _ = np.linalg.svd(X.T @ X)
        Lambda0 = ev * np.sqrt(n)
        Fhat0 = X @ Lambda0 / n

    IC = np.zeros(kmax + 1)
    for i in range(kmax):
        chat = Fhat0[:, : i + 1] @ Lambda0[:, : i + 1].T
        ehat = X - chat
        sigma = ((ehat * ehat / T).sum(axis=0)).mean()
        IC[i] = np.log(sigma) + CT[i]
    IC[kmax] = np.log((X * X / T).sum(axis=0).mean())
    return IC


def decide_from_ic(IC: np.ndarray, kmax: int = KMAX_BAING) -> dict:
    """
    Two decisions from one IC array.

    released    -- replicates train.py:baing()'s final three lines, including the
                   defect that a zero-factor argmin is returned as 1.
    zero_fixed  -- returns 0 when the criterion is minimised at the no-factor
                   slot; identical to `released` otherwise.
    """
    idx = int(np.argmin(IC))
    tied = int(np.count_nonzero(IC == IC[idx])) > 1
    released = int(idx * (idx < kmax)) + 1
    zero_fixed = 0 if idx == kmax else idx + 1
    return {
        "argmin_index": idx,
        "tie": tied,
        "released": released,
        "zero_fixed": zero_fixed,
    }


def released_baing_call(X: np.ndarray, kmax: int = KMAX_BAING, jj: int = BAING_JJ) -> dict:
    """Call the released package function; never let its tie-assert kill a run."""
    try:
        r_hat = int(released_baing(X, kmax, jj)[0])
        return {"ok": True, "r_hat": r_hat, "error": ""}
    except Exception as exc:  # AssertionError on criterion ties, anything else
        return {"ok": False, "r_hat": None, "error": f"{type(exc).__name__}: {exc}"[:300]}


# ---------------------------------------------------------------------------
# The three frozen operationalisations of the Sec-4.2 update
# ---------------------------------------------------------------------------
def within_block_phi(xi: np.ndarray, K_branch: str, gmm_seed: int):
    """
    One iteration's factor-adjusted residual -> MP -> clustering -> restricted OLS.

    Returns (d_hat, labels, phi_hat, status). `phi_hat` is the released
    NIRVAR.ols_parameters() estimate under the estimated block restriction, i.e.
    the block-restricted VAR(1) coefficient matrix.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        probe = NIRVAR(
            Xi=xi, d=None, K=K_TRUE, embedding_method=EMBEDDING_METHOD,
            gmm_random_int=gmm_seed,
        )
    d_hat = int(probe.d)
    if d_hat == 0:
        return d_hat, None, None, "d_hat_zero"
    if K_branch == "primary_fixed_K":
        model = probe
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            model = NIRVAR(
                Xi=xi, d=d_hat, K=d_hat, embedding_method=EMBEDDING_METHOD,
                gmm_random_int=gmm_seed,
            )
    similarity, labels = model.gmm()
    phi_hat = model.ols_parameters(similarity)
    return d_hat, labels, phi_hat, "ok"


def variant_panels(window: np.ndarray, xi: np.ndarray, phi_hat: np.ndarray) -> dict:
    """
    Variant A -- within-block residual of the factor-adjusted data:
        eps_t = xi_t - Phi_hat xi_{t-1}
      Bai-Ng on eps counts factor signal SURVIVING the network filter; the
      update is incremental, r' = r + k_extra.

    Variant B -- network-filtered observed panel:
        Xtilde_t = X_t - Phi_hat xi_{t-1}
      Bai-Ng on Xtilde re-selects r on a panel from which the estimated network
      dynamics have been removed; the update is absolute, r' = k.
    """
    eps = xi[1:] - xi[:-1] @ phi_hat.T
    x_tilde = window[1:] - xi[:-1] @ phi_hat.T
    return {"A_incremental": eps, "B_absolute": x_tilde}


def variant_c_ic(window: np.ndarray, phi_hat: np.ndarray,
                 kmax: int = KMAX_BAING, jj: int = BAING_JJ) -> np.ndarray:
    """
    Variant C -- network-aware criterion.

    PCA is run on the observed window exactly as the released `baing()` does, but
    the criterion's V(k) is evaluated on the NETWORK-FILTERED k-factor residual
    e_k[1:] - e_k[:-1] Phi_hat', so that idiosyncratic network dynamics no longer
    inflate V(k). The update is absolute, r' = argmin.
    """
    T, n = window.shape
    NT = n * T
    NT1 = n + T
    ii = np.arange(1, kmax + 1)
    GCT = min(n, T)
    if jj == 1:
        CT = np.log(NT / NT1) * ii * (NT1 / NT)
    elif jj == 2:
        CT = np.log(min(n, T)) * ii * (NT1 / NT)
    else:
        CT = np.log(GCT) / GCT * ii

    if T < n:
        ev, _, _ = np.linalg.svd(window @ window.T)
        Fhat0 = ev * np.sqrt(T)
        Lambda0 = window.T @ Fhat0 / T
    else:
        ev, _, _ = np.linalg.svd(window.T @ window)
        Lambda0 = ev * np.sqrt(n)
        Fhat0 = window @ Lambda0 / n

    def filtered_v(e: np.ndarray) -> float:
        f = e[1:] - e[:-1] @ phi_hat.T
        return float(((f * f / f.shape[0]).sum(axis=0)).mean())

    IC = np.zeros(kmax + 1)
    for i in range(kmax):
        chat = Fhat0[:, : i + 1] @ Lambda0[:, : i + 1].T
        IC[i] = np.log(filtered_v(window - chat)) + CT[i]
    IC[kmax] = np.log(filtered_v(window))
    return IC


# ---------------------------------------------------------------------------
# Trajectory classification (Sec 5 of DECISIONS.md)
# ---------------------------------------------------------------------------
STOP_STATES = (
    "correct_fixed_point",
    "wrong_fixed_point",
    "two_cycle",
    "longer_cycle",
    "out_of_grid",
    "baing_failure",
    "clustering_failure",
    "max_iter_reached",
)


def iterate_from_table(F: dict, r0: int, max_iter: int = MAX_ITER) -> dict:
    """
    Walk the frozen transition table F: r -> {'next': int|None, 'status': str}.

    F is a total function on R_GRID for one (world, origin, DGP, variant,
    baing branch, K branch) cell, so trajectories are table look-ups. This is
    only valid because the GMM seed is frozen per world and does NOT vary with
    the iteration index -- see DECISIONS.md D-06.
    """
    traj = [r0]
    seen = {r0: 0}
    for _ in range(max_iter):
        cur = traj[-1]
        cell = F.get(cur)
        if cell is None or cell["status"] not in ("ok",):
            return {"trajectory": traj, "stop_state": cell["status"] if cell else "out_of_grid",
                    "n_iter": len(traj) - 1, "terminal_r": None}
        nxt = cell["next"]
        if nxt not in R_GRID:
            return {"trajectory": traj + [nxt], "stop_state": "out_of_grid",
                    "n_iter": len(traj), "terminal_r": nxt}
        if nxt == cur:
            state = "correct_fixed_point" if cur == R_TRUE else "wrong_fixed_point"
            return {"trajectory": traj + [nxt], "stop_state": state,
                    "n_iter": len(traj), "terminal_r": nxt}
        if nxt in seen:
            cycle_len = len(traj) - seen[nxt]
            state = "two_cycle" if cycle_len == 2 else "longer_cycle"
            return {"trajectory": traj + [nxt], "stop_state": state,
                    "n_iter": len(traj), "terminal_r": nxt}
        seen[nxt] = len(traj)
        traj.append(nxt)
    return {"trajectory": traj, "stop_state": "max_iter_reached",
            "n_iter": max_iter, "terminal_r": traj[-1]}
