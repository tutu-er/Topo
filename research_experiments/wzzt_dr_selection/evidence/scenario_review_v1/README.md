# Bounded scenario review evidence

Read ../../SCENARIO_PROGRESS.md first. This stage contains no new price optimizer or DR recovery experiment.

- candidate_audit_summary.json is the short result; candidate_audit.json.gz holds all160 records. Decompress it byte-for-byte to candidate_audit.json before replaying private-root commands. Its uncompressed hash is the original hash in audit_independent.json.
- candidate_extract.json.gz contains only synthetic fitted parameters and saved historical validation scores, not P/Q/V observations, truth, household records, or test scores. Original result hashes identify the source of each extract.
- source_snapshot.zip preserves13 source files, including original archived selector/pipeline sources. Archive readback was byte-verified.
- independent_candidate_audit.py was assembled after the executed stdin audit. Its original audit receipt remains unchanged; the file is recovery code, not a retroactively frozen run source.
- Observation files were only checked for existence. Their content hashes/shapes remain pending before any new use.
-160 conditions are paired within10 original bundle blocks; historical candidate diversity and RNJ/WZZT differences are not recovery gains or confidence coverage.
