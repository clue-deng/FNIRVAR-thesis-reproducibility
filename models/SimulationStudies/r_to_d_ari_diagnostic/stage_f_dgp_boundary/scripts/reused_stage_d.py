#!/usr/bin/env python3
"""
Transition-map machinery reused VERBATIM from Stage D.

Every function below is a byte-identical copy of the corresponding function in

    stage_d_iterative_loop/scripts/executed_20260819/common_stage_d.py
    sha256 1470f78717c00c0660039af95b65bff2f45612c9251b9a3474ced398a2a7ac3d

so that Stage F cannot silently drift from the audited Stage-D feedback
definition. `scripts/validate_stage_f.py` re-extracts each function from that
frozen file and asserts byte equality with the copy here. Stage F adds no new
feedback logic; only the DGP constructor is parameterised.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
REPO_ROOT = Path(os.environ.get("FNIRVAR_REPO_ROOT", _HERE.parents[5]))
sys.path.insert(0, str(REPO_ROOT / "fnirvar" / "modeling"))
from generativeVAR import GenerateFNIRVAR, GenerateNIRVAR  # noqa: E402,F401
from train import FactorAdjustment, NIRVAR, baing as released_baing  # noqa: E402,F401

R_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9)
KMAX_BAING = 10
BAING_JJ = 2
VARIANTS = ("A_incremental", "B_absolute", "C_criterion")
BAING_BRANCHES = ("released", "zero_fixed")
K_BRANCHES = ("primary_fixed_K", "robustness_K_equals_d_hat")
MAX_ITER = 20
K_TRUE = 4
EMBEDDING_METHOD = "Pearson Correlation"
R_TRUE = 5

STOP_STATES = (
    "correct_fixed_point", "wrong_fixed_point", "two_cycle", "longer_cycle",
    "out_of_grid", "baing_failure", "clustering_failure", "max_iter_reached",
)

# --- BEGIN VERBATIM STAGE-D BLOCKS -------------------------------------------
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


def child_seed(child: np.random.SeedSequence) -> int:
    return int(child.generate_state(1, dtype=np.uint32)[0])


def loadings(n: int, r: int, sigma: float, rs: np.random.RandomState) -> np.ndarray:
    signs = rs.choice([-1, 1], size=(n, r))
    noise = rs.normal(loc=0.0, scale=sigma, size=(n, r))
    return signs + noise


def sha256_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
# --- END VERBATIM STAGE-D BLOCKS ---------------------------------------------
