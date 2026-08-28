# FNIRVAR thesis experiment suite

## Scientific sequence

1. Preliminary suite checks the pipeline and interpretation of labels, MP counting and residual construction.
2. Main propagation experiment changes the imposed factor dimension and measures downstream d_hat, ARI and MSPE.
3. Stage C identifies the affected subspaces; the full-grid extension tests the mechanism over r=1,...,9.
4. Stage B/M1 separates population detectability from finite-sample and serial-dependence explanations for d_hat below K.
5. Stage D implements iterative feedback and measures fixed points, basins and initialisation dependence.
6. Stage E-lite tests whether held-out MSPE selects a structurally appropriate r from frozen origin-level results.
7. Stage F locates feasibility boundaries as community separation and factor loading strength change.

## Reading order

Read each stage README, then its small summary CSVs, then figures. Large cell-level CSVs are included so published aggregates can be reconstructed. The runs directory is retained only where later stages directly consume frozen rows.

## Public-repository scope

This copy contains released source code, experiment code, configurations, result CSVs, figures and reproduction commands. Internal work logs, temporary runs, qualification records, execution briefs and stale exploratory directories are omitted.
