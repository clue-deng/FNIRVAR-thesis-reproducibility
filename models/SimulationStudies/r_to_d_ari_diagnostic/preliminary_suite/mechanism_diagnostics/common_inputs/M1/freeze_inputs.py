#!/usr/bin/env python3
"""
Freeze the realised (Phi, Omega) for M1, one folder per structural replication.

Reconstructs network_rs identically to preliminary_diagnostics.py's
generate_full_dataset() (same SeedSequence/spawn order, same GenerateNIRVAR
call), so Phi/Omega here are bit-identical to what already produced the
xi_full used in preliminary_results.csv. Does not call .generate() -- Phi and
Omega are fully determined inside GenerateNIRVAR.__init__, before any time
series would be drawn, so no innovations are consumed here.

Writes, per replicate: phi.npy, omega.npy, manifest.json (shapes/dtypes,
spectral radius, DGP config, SHA-256 hashes of both .npy files).

Does not solve anything and does not write any interpretation -- inputs only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO_ROOT / "fnirvar" / "modeling"))
from generativeVAR import GenerateNIRVAR  # noqa: E402

# Exactly the values in preliminary_diagnostics.py
N = 100
K_TRUE = 4
P_IN = 0.9
P_OUT = 0.1
VAR_SPECTRAL_RADIUS = 0.9
SYMMETRIZE_PHI = False
GLOBAL_NOISE = 1.0
T_LIST = [200, 500, 1000, 1500, 3000]  # only used here to size T= in the constructor
MASTER_SEED = 20260727
N_STRUCTURAL_SEEDS = 5

HERE = Path(__file__).resolve().parent


def random_state(child: np.random.SeedSequence) -> np.random.RandomState:
    return np.random.RandomState(int(child.generate_state(1, dtype=np.uint32)[0]))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    top_children = np.random.SeedSequence(MASTER_SEED).spawn(N_STRUCTURAL_SEEDS)

    for replicate, top_child in enumerate(top_children):
        structural_seed = int(top_child.generate_state(1, dtype=np.uint32)[0])

        # identical spawn order to generate_full_dataset(): [network, factor, gmm]
        children = np.random.SeedSequence(structural_seed).spawn(3)
        network_rs = random_state(children[0])

        network = GenerateNIRVAR(
            random_state=network_rs,
            T=max(T_LIST),
            B=K_TRUE,
            N=N,
            Q=1,
            p_in=P_IN,
            p_out=P_OUT,
            multiplier=VAR_SPECTRAL_RADIUS,
            global_noise=GLOBAL_NOISE,
            symmetrize_phi=SYMMETRIZE_PHI,
        )
        Phi = network.phi_coefficients[:, 0, :, 0].astype(np.float64)
        Omega = np.diag(network.innovations_variance[:, 0]).astype(np.float64)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(Phi))))

        outdir = HERE / f"replicate{replicate}_seed{structural_seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        phi_path = outdir / "phi.npy"
        omega_path = outdir / "omega.npy"
        np.save(phi_path, Phi)
        np.save(omega_path, Omega)

        manifest = {
            "replicate": replicate,
            "structural_seed": structural_seed,
            "dgp_config": {
                "N": N, "K_true_B": K_TRUE, "p_in": P_IN, "p_out": P_OUT,
                "VAR_spectral_radius_target": VAR_SPECTRAL_RADIUS,
                "symmetrize_phi": SYMMETRIZE_PHI, "global_noise": GLOBAL_NOISE,
                "T_used_for_construction": max(T_LIST), "Q": 1,
            },
            "seed_derivation": {
                "MASTER_SEED": MASTER_SEED,
                "top_child_index": replicate,
                "network_rs_spawn_index": 0,
                "spawn_order": ["network", "factor", "gmm"],
            },
            "phi_shape": list(Phi.shape), "phi_dtype": str(Phi.dtype),
            "omega_shape": list(Omega.shape), "omega_dtype": str(Omega.dtype),
            "phi_spectral_radius_actual": spectral_radius,
            "phi_spectral_radius_target": VAR_SPECTRAL_RADIUS,
            "sha256": {
                "phi.npy": sha256_of(phi_path),
                "omega.npy": sha256_of(omega_path),
            },
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"replicate {replicate}: seed={structural_seed} rho(Phi)={spectral_radius:.4f} -> {outdir}")


if __name__ == "__main__":
    main()
