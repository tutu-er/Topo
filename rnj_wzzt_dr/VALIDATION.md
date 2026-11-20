# Validation

Install the solver, physical, and test extras for the full set of checks below. Run them from the top-level directory.

~~~sh
python -m rnj_wzzt_dr doctor
python -m rnj_wzzt_dr check --solver
python -m pytest -q rnj_wzzt_dr/tests rnj_wzzt_dr/experiments/wzzt_long_grid_20261117 rnj_wzzt_dr/experiments/w_importance_20261117
~~~

`doctor` verifies the packaged source hashes. `check --solver` tests small identification and pricing cases with Gurobi. The regression suite also covers the saved candidate bank, parameter fingerprints, strict-MSE selection, relative response floor, W modes, and path relocation. The study tests exercise the controllers and independent evaluators on small fixtures.

The November 18 source cleanup recorded 125 passing tests, 2 optional solver skips, and 66 passing subtests, plus 18 passing solver interface checks. It compared the candidate parameters and selected index across 1,777 successful bank exports and kept all 23 long-grid failure records. See the [cleanup receipt](../cleanup_manifest.json) for the underlying counts.

The tests exercise the software interfaces. Study performance and the conditions of the numerical comparisons are described in the [loss study](experiments/wzzt_long_grid_20261117/README.md) and the [W ablation](experiments/w_importance_20261117/README.md).
