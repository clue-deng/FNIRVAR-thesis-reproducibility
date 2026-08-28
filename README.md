# FNIRVAR factor-dimension propagation experiments

This repository is a publication-oriented copy of the code and data used to study how factor-dimension misspecification propagates through FNIRVAR.

## Research question

How does using the wrong number of common factors change the residual spectrum, MP-selected embedding dimension, community recovery and prediction, and under what signal and initialisation conditions is iterative feedback feasible?

## Repository layout

- fnirvar/modeling/: released FNIRVAR generator and estimator source used by the experiments.
- models/SimulationStudies/mspe_factor_performance/: released MSPE reference scaffold.
- models/SimulationStudies/weak_factors/: released weak-factor reference scaffold.
- models/SimulationStudies/r_to_d_ari_diagnostic/: thesis experiments, ordered and explained in its README.
- reproduce/: checksum, figure-regeneration and full-simulation commands.
- FILE_MANIFEST.csv: one row per public file with its role, size and SHA-256 digest.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    python reproduce/verify_checksums.py
    bash reproduce/regenerate_figures.sh

The formal main run recorded Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3 and scikit-learn 1.7.2. Numerical linear algebra can differ slightly across operating systems and BLAS/LAPACK builds; use the included frozen CSVs for reported tables and the full commands for an independent rerun.

## Reproducibility levels

- Exact integrity: SHA256SUMS.txt verifies the curated files byte-for-byte.
- Exact reanalysis: Stage E-lite can be recomputed from the included 20 origin-level CSVs.
- Independent simulation: all stages retain runner, configuration and summary code. Tiny floating-point differences can occur across platforms.
- Publication outputs: reported summary CSVs and figures are included under their stage directories.

## License and attribution

The released FNIRVAR package and reference simulations retain the original MIT license and authorship metadata. Thesis experiment scripts are provided for reproducibility.
