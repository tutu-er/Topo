# Maintained Files and Data

The current entry point is `rnj_wzzt_dr/run.py`. This English edition follows source snapshot `e66aae90`.

- `rnj_wzzt_dr/identification/`, `dr/`, `resources/`, `tests/`, and the package-level Python files contain the main identification and conditional demand-response workflow.
- `rnj_wzzt_dr/physical/` contains the AC simulator and four reference network tables.
- `rnj_wzzt_dr/experiments/` contains research implementations and protocol-specific checks.
- Local `data/` contains input and reference data, including the 296-entry evaluation catalog and the frozen banks from the two latest studies. The repository ignores this directory.
- Local `data/legacy_sources/` preserves selected historical source bytes for provenance checks.

The default evaluation catalog includes 240 main benchmark entries, 40 history288 entries, and 16 history10 entries. The retired pressure cohort of 160 entries remains available as historical metadata. The groups differ in observation budget and provenance.

Original data files retain their stored bytes and SHA-256 values. The path resolver uses retained locations when reading old references. The [cleanup manifest](cleanup_manifest.json) records what was retained and how to locate the private recovery records.
