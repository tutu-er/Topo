# Independent arithmetic replay of DR pool v2

The completed matrix has 48 evaluation cells, 42 unique observation fits and 6 reuse cells. All 42 unique fits passed arithmetic replay. No optimizer or physical simulator was run during this audit. Identification source files and matrix outputs were left unchanged.

`audit_completed.py` independently reads the frozen physical pickle observations, samples the specified real dates, reconstructs block R/X matrices and computes L1 losses by both matrix and explicit block sums. It checks all selection-candidate losses, pre/post-contraction returned models, the original 3% selection and 0.16 contraction, input hashes, physical-day grouping, finite nonnegative bounded weights, laminar pool membership, saved LP statuses, source hashes and runtime versions. Daily centering is compared against direct physical-date subtraction. LP optimality is checked from recorded native status and residual audit; this does not independently prove optimality or tree recovery.

The full receipt remains in the local scratch `wzzt_dr_pool_v2/audit/final_arithmetic_receipt.json`; its SHA256 and the unchanged first-eight receipt SHA256 are in the compact receipt. No raw household trajectories are stored here.

Run with the same environment and local data:

```text
python audit_completed.py --run-root <scratch>/wzzt_dr_pool_v2/identification_v1 --repo-root <repo> --receipt <scratch>/wzzt_dr_pool_v2/audit/new_receipt.json
```

The script requires all 42 declared unique results. The environment comparison is intentionally exact. Its arithmetic agreement establishes implementation consistency, while demand-response efficacy is assessed separately from the matrix results.
