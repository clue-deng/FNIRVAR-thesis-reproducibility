#!/usr/bin/env python3
"""
Small, reproducible FNIRVAR diagnostics for a supervisor update.

This file imports the checked-out FNIRVAR package but never modifies it.
It runs three linked designs:

1. Paired-T true-residual baseline.
2. Manual r-misspecification at T=1500.
3. Repository Bai-Ng (jj=2) bridge at T=1500.

Each residual is evaluated with three branches:
  oracle       : d=K_true, K=K_true
  mp_fixed_k   : d=d_hat,  K=K_true
  full_pipeline: d=d_hat,  K=d_hat
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score


N = 100
R_TRUE = 5
L_F = 2
Q_SHOCKS = 5
RHO_F = 0.7
K_TRUE = 4
P_IN = 0.9
P_OUT = 0.1
VAR_SPECTRAL_RADIUS = 0.9
LOADING_SCALE = 0.4
LOADING_SIGMA = 0.1

MASTER_SEED = 20260727
N_STRUCTURAL_SEEDS = 5
T_LIST = [200, 500, 1000, 1500, 3000]
T_ANCHOR = 1500
R_USED_LIST = [3, 5, 7]
KMAX = 10
N_LEADING_EIGS = 8


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Path containing fnirvar/modeling and models/SimulationStudies.",
    )
    parser.add_argument("--outdir", type=Path, default=here)
    return parser.parse_args()


def import_repo(repo_root: Path):
    model_dir = repo_root / "fnirvar" / "modeling"
    if not model_dir.exists():
        raise FileNotFoundError(f"Could not find modeling directory: {model_dir}")
    sys.path.insert(0, str(model_dir))
    from generativeVAR import GenerateFNIRVAR, GenerateNIRVAR
    from train import FactorAdjustment, NIRVAR, baing

    return GenerateFNIRVAR, GenerateNIRVAR, FactorAdjustment, NIRVAR, baing


def random_state(child: np.random.SeedSequence) -> np.random.RandomState:
    return np.random.RandomState(int(child.generate_state(1, dtype=np.uint32)[0]))


def loadings(n: int, r: int, sigma: float, rs: np.random.RandomState) -> np.ndarray:
    signs = rs.choice([-1, 1], size=(n, r))
    noise = rs.normal(loc=0.0, scale=sigma, size=(n, r))
    return signs + noise


def generate_full_dataset(
    seed: int,
    GenerateFNIRVAR,
    GenerateNIRVAR,
):
    children = np.random.SeedSequence(seed).spawn(3)
    network_rs = random_state(children[0])
    factor_rs = random_state(children[1])
    gmm_seed = int(children[2].generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))

    network = GenerateNIRVAR(
        random_state=network_rs,
        T=max(T_LIST),
        B=K_TRUE,
        N=N,
        Q=1,
        p_in=P_IN,
        p_out=P_OUT,
        multiplier=VAR_SPECTRAL_RADIUS,
        global_noise=1.0,
        symmetrize_phi=False,
    )
    xi_full = network.generate()[:, :, 0]
    true_labels = np.asarray([network.categories[str(i)] for i in range(N)])

    Lambda = LOADING_SCALE * loadings(N, R_TRUE, LOADING_SIGMA, factor_rs)
    factors = GenerateFNIRVAR(
        l_F=L_F,
        T=max(T_LIST),
        r=R_TRUE,
        q=Q_SHOCKS,
        rho_F=RHO_F,
        random_state=factor_rs,
    )
    X_full = factors.generate_data(Lambda, xi=xi_full)
    return X_full, xi_full, true_labels, gmm_seed


def spectrum(xi: np.ndarray):
    correlation = np.corrcoef(xi.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    edge = (1.0 + np.sqrt(xi.shape[1] / xi.shape[0])) ** 2
    d_hat = int(np.count_nonzero(eigenvalues > edge))
    return correlation, eigenvalues, edge, d_hat


def base_row(
    experiment,
    structural_seed,
    T,
    residual_type,
    r_used,
    r_selected,
    branch,
    d_hat,
    d_embed,
    K_gmm,
    ari,
    status,
    edge,
    eigenvalues,
):
    row = {
        "experiment": experiment,
        "structural_seed": structural_seed,
        "T": T,
        "residual_type": residual_type,
        "r_used": r_used,
        "r_selected": r_selected,
        "branch": branch,
        "d_hat": d_hat,
        "d_embed": d_embed,
        "K_gmm": K_gmm,
        "ARI": ari,
        "status": status,
        "MP_edge": edge,
    }
    row.update({f"eig_{i + 1}": eigenvalues[i] for i in range(N_LEADING_EIGS)})
    return row


def evaluate_residual(
    xi,
    true_labels,
    gmm_seed,
    NIRVAR,
    experiment,
    structural_seed,
    T,
    residual_type,
    r_used=None,
    r_selected=None,
):
    _, eigenvalues, edge, d_hat = spectrum(xi)

    probe = NIRVAR(Xi=xi, d=1, K=1, embedding_method="Pearson Correlation")
    repo_d_hat = int(probe.marchenko_pastur_estimate())
    if repo_d_hat != d_hat:
        raise AssertionError(f"Independent d_hat={d_hat}, repository d_hat={repo_d_hat}")

    designs = [
        ("oracle", K_TRUE, K_TRUE),
        ("mp_fixed_k", d_hat, K_TRUE),
        ("full_pipeline", d_hat, d_hat),
    ]
    rows = []
    for branch, d_embed, K_gmm in designs:
        if d_embed < 1 or K_gmm < 1:
            rows.append(
                base_row(
                    experiment, structural_seed, T, residual_type, r_used, r_selected,
                    branch, d_hat, d_embed, K_gmm, math.nan, "d_hat_zero",
                    edge, eigenvalues,
                )
            )
            continue
        try:
            model = NIRVAR(
                Xi=xi,
                d=int(d_embed),
                K=int(K_gmm),
                embedding_method="Pearson Correlation",
                gmm_random_int=gmm_seed,
            )
            _, labels = model.gmm()
            ari = float(adjusted_rand_score(true_labels, labels))
            status = "ok"
        except Exception as exc:
            ari = math.nan
            status = f"error:{type(exc).__name__}"
        rows.append(
            base_row(
                experiment, structural_seed, T, residual_type, r_used, r_selected,
                branch, d_hat, d_embed, K_gmm, ari, status, edge, eigenvalues,
            )
        )
    return rows


def implementation_audit(repo_root: Path, NIRVAR, sample_xi):
    train_path = repo_root / "fnirvar" / "modeling" / "train.py"
    text = train_path.read_text()
    _, eigs, edge, d_sym = spectrum(sample_xi)
    d_general = int(np.count_nonzero(np.linalg.eigvals(np.corrcoef(sample_xi.T)) > edge))

    pearson = NIRVAR(sample_xi, d=2, K=K_TRUE, embedding_method="Pearson Correlation")
    covariance = NIRVAR(sample_xi, d=2, K=K_TRUE, embedding_method="Covariance Matrix")
    pearson_mp = int(pearson.marchenko_pastur_estimate())
    covariance_mp = int(covariance.marchenko_pastur_estimate())
    embedding_difference = float(np.linalg.norm(pearson.embed() - covariance.embed()))

    return {
        "mp_uses_correlation": "Sigma = self.pearson_correlations()" in text,
        "literal_cutoff_present": "cutoff = (1 + np.sqrt(self.N/self.T))**2" in text,
        "covariance_selector_calls_correlation": (
            "elif self.embedding_method == 'Covariance Matrix':\n"
            "            Sigma = self.pearson_correlations()" in text
        ),
        "covariance_embedding_calls_covariance": (
            'elif self.embedding_method == "Covariance Matrix":\n'
            "            embedding_object = self.covariance_matrix()" in text
        ),
        "ks_import_commented": "# from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution" in text,
        "ks_name_occurrences": text.count("ks_statistic"),
        "baing_uses_log_residual_variance": "IC1[i] = np.log(Sigma[i]) + CT[i]" in text,
        "eigvals_and_eigvalsh_counts_match": d_general == d_sym,
        "eigvals_count": d_general,
        "eigvalsh_count": d_sym,
        "pearson_and_covariance_selector_d_match": pearson_mp == covariance_mp,
        "pearson_selector_d": pearson_mp,
        "covariance_selector_d": covariance_mp,
        "pearson_vs_covariance_embedding_l2_difference": embedding_difference,
        "explicit_d_bypasses_selector": int(NIRVAR(sample_xi, d=K_TRUE, K=K_TRUE).d) == K_TRUE,
        "gmm_default_seed": inspect.signature(NIRVAR.__init__).parameters[
            "gmm_random_int"
        ].default,
        "mp_edge_sample": edge,
        "leading_eigenvalues_sample": [float(x) for x in eigs[:N_LEADING_EIGS]],
    }


def write_csv(path: Path, rows):
    fields = [
        "experiment", "structural_seed", "T", "residual_type", "r_used",
        "r_selected", "branch", "status", "d_hat", "d_embed", "K_gmm",
        "ARI", "MP_edge",
    ] + [f"eig_{i + 1}" for i in range(N_LEADING_EIGS)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    values = [x for x in values if not math.isnan(x)]
    return float(np.mean(values)) if values else math.nan


def summarize(rows):
    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["experiment"], row["T"], row["residual_type"],
                row["r_used"], row["r_selected"], row["branch"],
            )
        ].append(row)

    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        d_values = [int(row["d_hat"]) for row in group]
        ari_values = [float(row["ARI"]) for row in group]
        summary.append(
            {
                "experiment": key[0],
                "T": key[1],
                "residual_type": key[2],
                "r_used": key[3],
                "r_selected": key[4],
                "branch": key[5],
                "n": len(group),
                "n_valid_ARI": sum(not math.isnan(x) for x in ari_values),
                "n_d_hat_zero": sum(x == 0 for x in d_values),
                "mean_d_hat": float(np.mean(d_values)),
                "sd_d_hat": float(np.std(d_values, ddof=1)) if len(d_values) > 1 else 0.0,
                "mean_ARI": finite_mean(ari_values),
                "sd_ARI": (
                    float(np.std([x for x in ari_values if not math.isnan(x)], ddof=1))
                    if sum(not math.isnan(x) for x in ari_values) > 1 else 0.0
                ),
            }
        )
    return summary


def write_summary_csv(path: Path, rows):
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    GenerateFNIRVAR, GenerateNIRVAR, FactorAdjustment, NIRVAR, baing = import_repo(
        args.repo_root
    )

    top_children = np.random.SeedSequence(MASTER_SEED).spawn(N_STRUCTURAL_SEEDS)
    rows = []
    selected_r_values = []
    sample_xi = None

    for replicate, child in enumerate(top_children):
        structural_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        X_full, xi_full, true_labels, gmm_seed = generate_full_dataset(
            structural_seed, GenerateFNIRVAR, GenerateNIRVAR
        )
        if sample_xi is None:
            sample_xi = xi_full[:T_ANCHOR]

        # Experiment 1: paired-T baseline using the same DGP and nested trajectory.
        for T in T_LIST:
            rows.extend(
                evaluate_residual(
                    xi_full[:T], true_labels, gmm_seed, NIRVAR,
                    "paired_T_true_residual", structural_seed, T, "true",
                )
            )

        X_anchor = X_full[:T_ANCHOR]

        # Experiment 2: controlled manual intervention on r.
        for r_used in R_USED_LIST:
            xi_hat = FactorAdjustment(X_anchor, r_used, L_F).get_idiosyncratic_component()
            rows.extend(
                evaluate_residual(
                    xi_hat, true_labels, gmm_seed, NIRVAR,
                    "manual_r_sweep", structural_seed, T_ANCHOR, "estimated",
                    r_used=r_used,
                )
            )

        # Experiment 3: bridge to the repository's factor-selection implementation.
        r_selected, _, _, _ = baing(X_anchor, kmax=KMAX, jj=2)
        r_selected = int(r_selected)
        selected_r_values.append(r_selected)
        xi_selected = FactorAdjustment(
            X_anchor, max(r_selected, 1), L_F
        ).get_idiosyncratic_component()
        rows.extend(
            evaluate_residual(
                xi_selected, true_labels, gmm_seed, NIRVAR,
                "baing_jj2_bridge", structural_seed, T_ANCHOR, "estimated",
                r_used=r_selected, r_selected=r_selected,
            )
        )
        print(f"replicate {replicate + 1}/{N_STRUCTURAL_SEEDS}: r_selected={r_selected}")

    audit = implementation_audit(args.repo_root, NIRVAR, sample_xi)
    audit["baing_jj2_selected_r_values"] = selected_r_values
    audit["repo_root"] = str(args.repo_root)
    audit["package_modified"] = False

    raw_path = args.outdir / "preliminary_results.csv"
    summary_path = args.outdir / "preliminary_summary.csv"
    audit_path = args.outdir / "implementation_audit.json"
    write_csv(raw_path, rows)
    summary_rows = summarize(rows)
    write_summary_csv(summary_path, summary_rows)
    audit_path.write_text(json.dumps(audit, indent=2))

    print(f"wrote {raw_path} ({len(rows)} rows)")
    print(f"wrote {summary_path} ({len(summary_rows)} rows)")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
