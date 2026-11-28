# What moved here and what changed (2026-11-27)

This directory is the revised, self-contained version of the case-study project for review. The old locations are
left unchanged; once this version is accepted they can be marked as superseded.

## 1. Where things were and are now

| Before | Now |
| --- | --- |
| `rnj_wzzt_dr/identification_v8/*.py` (method, at c5c4aae) | `ptid/method/*.py` |
| `identification_v8/test_v8.py` | `ptid/tests/test_method.py` (more tests, section 4) |
| `experiments/case_library_20261125/dev_cases.py`, `build.load`, `build.to_input` | `ptid/cases/library.py` |
| `experiments/case_library_20261125/physics.py` | `ptid/cases/physics.py` (a copy; the radial ordering copied in) |
| `wzzt_evidence_20261124/capture.load_input` (real inputs) | `ptid/cases/real.py` (reads the files without that code) |
| `probe_case.py` (`World`; `run_case`) | `ptid/cases/world.py`; `ptid/runs/probe_cases.py` |
| `dev_v8.py` (development runner) | `ptid/runs/identify_cases.py` (development and real; summary; compare) |
| `check_extra_lines.py`, `check_wide_nodes.py` | `ptid/runs/checks.py` |
| `manifest_dev.json` | `ptid/cases/manifest_dev.json` (a copy, for the hash check) |
| `paper/procedure.tex` (draft v15) | `paper/procedure.tex` (draft v16) |
| `paper/PROPOSALS_UNDER_TEST.md` | `PROPOSALS.md` |
| `paper/DEV_RESULTS_V12.md`; `RESULTS_SEALED.md` (summary) | `RESULTS.md` |
| `IDENTIFICATION_PLAN.md` (history and open items) | `PLAN.md` |
| `DESIGN.md` (case library) | `PROTOCOL.md`, section 2 (summary; the full design stays where it is) |
| `PROTOCOL_ID.md` (registered, spent) | `RESULTS.md`, section 1 (summary) |
| run files under `outputs/case_library_20261125/` | new runs under `outputs/topology_study/` |

## 2. Not moved, on purpose

- **The case generator and the case files.** The generator is registered and frozen; copying it would make a second
  copy that could drift. This directory reads the cases and checks every file's hash.
- **The registered predecessor** (`identification/`) and its sealed-test runner. The legacy tests
  (`python -m ptid.tests.test_method --legacy`) still compare against it.
- **Earlier exploratory scripts** of the case library (ablations, comparisons with RNJ/NJ, the price study, robustness).
  Their findings that still matter are in RESULTS.md and PLAN.md.

## 3. Code changes against `identification_v8` at c5c4aae

The method is the same code, with two refinements found in the review:

1. **The node MIQP's coefficients are scaled to the size of their columns**, like the tree MIQP's since c5c4aae
   (`model.factor_scaled`, now shared by both). The scaling is an exact change of variables. It protects pricing's
   "certified" claims from the numerical fault that produced false optima in the tree MIQP. Checked against
   enumeration in two tests (synthetic and on a development case) and on wide nodes of huge cases.
2. **Probing: an outside target's flip test is checked against its best local swap** (the known flipped tree), as the
   labels inside the tree already were. Only Algorithm 2 is affected.

Docstrings now refer to draft v16. Everything else is unchanged, and the readers reproduce the old readings and truth
exactly (development and real cases). RESULTS.md, section 4, records the equality check of whole runs.

## 4. Runners and tests

- One runner for development and real inputs, with a summary by kind (including solver outcomes and numerical failures)
  and a comparison of two runs on their common cases.
- The real inputs are read without the earlier study's code.
- `one_twin_per_feeder()`: the 72-case selection of the 2026-11-27 run.
- Two truths in the measurements (2026-11-27): besides the identifiable truth (3 milliohm floor), each run records
  the exact truth, and the summary counts decided labels that are not physical lines. Older run files get the exact
  truth when they are read. The method's output is untouched (RESULTS.md §4.1 says why).
- New tests:
  - the tree MIQP against enumeration of every laminar subset, with and without the per-meter bounds and with a
    required grouping;
  - the claim check;
  - convergence of `solve`;
  - the node MIQP against enumeration (synthetic);
  - the case readers and the selection.
- The package runs with only this directory on the path: 12 tests without data pass in isolation; 3 data tests and
  2 legacy tests pass with the case files and the repository.

## 5. Node MIQP tests (2026-11-28, branch `exp/node_miqp_20261128`)

Method code, all backward compatible (defaults keep earlier results):

- `search.node_miqp(..., exclude=[S])`: one no-good cut per excluded union. With the best union excluded, the solve is
  the node's flip test (best other union and a bound on every other union).
- `search.node_miqp(..., start=S)`: the binaries' MIP start; speed only.
- `search.node_price` and `best_addition(..., node_gamma=)`, `identify.Settings.node_gamma` (default 0): a line found
  at a node with c children must also pay gamma ln(2^c - c - 2), the extended BIC. Not adopted; tested in
  experiments/node_miqp (N2b, N3 arm D).

Runners `ptid/runs/node_study.py` (feeders with one wide junction, AC simulation, node scan with flip test, enumeration),
`node_exact.py` (N1), `node_recovery.py` (N2), `node_price.py` (N2b), `node_pipeline.py` (N3), `node_probe.py` (N4),
`node_report.py` (all tables into experiments/node_miqp/TABLES.md). Tests `ptid/tests/test_node_study.py` (8 checks:
the flip test and runner-up against enumeration, the exclusion cut, warm start changes nothing, the extra price, one-home
orders equal the draft's block rule, paired meter noise, verdicts).
