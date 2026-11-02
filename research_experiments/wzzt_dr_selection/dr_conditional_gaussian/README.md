# Conditional Gaussian DR recovery entry

Read MODEL.md for the full modeled contract and REPORT.md for the bounded pilot
results. The implementation consists of conditional_integral.py (exact inner
probability and coefficient gradients), problem.py (history Gaussian plus
continuous DR), and runner.py (prepare/solve/evaluate/labels).

The pilot is private under
`C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_conditional_dr_stage/run_v1`.
It binds the old history, candidate pool, resources, and raw response predictor
by SHA256. Source snapshot, protocol, problem, solver trace/checkpoint, frozen
prices, independent random stream, scalar path results and end receipt support
recovery. Do not rerun completed phases or redraw evaluation paths.

Pure tests: `python -m pytest <source-directory> -q`.
Fresh bounded run: `python runner.py prepare --output <new-private-directory>`,
then `solve`, `evaluate`, `labels` with the same output. A source-hash mismatch
must be resolved by restoring the saved execution snapshot, not by overwriting
the protocol. Existing production defaults and previous results remain unchanged.

Only source, mathematical description, explicit scalar aggregates and hashes
may be published. Full R/X, P/Q/Y, response/price paths and random draws stay in
the private run. This is frozen-candidate evaluation, not a new RNJ/W search.
