# Compact adaptive evidence

All calibration/evaluation/stress histories remain in their original requested
denominators. `independent_summary.json` was recomputed without importing
project modules or running AC, selectors or optimizers.

`sources` contains byte-preserving, deterministic gzip copies of the original
postprocessed fits, actions, gates, bounds, empirical/impedance CSV/JSON and
configuration/progress/summary records. `source_manifest.json` records source
and stored SHA-256 and exact byte counts. Its `projection=true` entries are
explicit loss/success/coverage/timing projections of large raw trial JSON;
they have their own projection hashes and retain the original raw source
hash. Those projections are not claimed to be byte-identical raw histories.
All raw NPZ are omitted, with hashes retained. The original output records
were read only. The original source root is recorded for optional live hash
comparison; the offline verifier needs only this directory plus NumPy/SciPy.

Run: `python verify_adaptive_compact.py .` from this directory.
The verifier checks archive bytes and recomputes rank thresholds, all history
events, candidate exclusion numerators/denominators, full and last-round
timing, solver workload and impedance missing/applicable counts. Recorded
AC objective replay statuses are checked as historical evidence; the
offline verifier does not replay AC or certify global MLE. Stress batches
do not inherit the nominal-population marginal gate guarantee.

Final retention uses only the last round. False rejection uses any round.
Gates do not prune the action policy; a candidate may re-enter next round.
Candidate/edge/round counts are not independent history units.
