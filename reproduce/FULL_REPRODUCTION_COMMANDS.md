# Full reproduction commands

Run every command from the repository root after installing requirements.txt.

Common setup:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export FNIRVAR_REPO_ROOT="$PWD"
    D="$PWD/models/SimulationStudies/r_to_d_ari_diagnostic"
    mkdir -p reproduced_runs reproduced_results

The exact stage-by-stage commands are collected in reproduce/run_all_stages.sh. That file is intentionally conservative: it writes new runs below reproduced_runs and new summaries below reproduced_results.

Quick checks:

    python reproduce/verify_checksums.py
    bash reproduce/regenerate_figures.sh

Important notes:

- Stage E-lite consumes the included 20 frozen origin-level CSVs.
- Stage C and its full-grid extension also use those rows and the accompanying minimal manifests.
- Full Monte Carlo reruns can take several hours.
- Run figure regeneration in a disposable clone if byte-identical PDF hashes matter.
