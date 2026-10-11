2026-10-11: Shared source and joint historical refitting, local mechanism study

Scope
  Exact linear squared-voltage-drop model; three observed terminals and the
  three candidate cherries 12, 13, 23. Source location is the endpoint of the
  common trunk, so its response is a times the all-ones vector. Continuous
  line parameters are refitted in every candidate. Source amplitudes vary
  arbitrarily by slot within a declared bound. No AC, unknown-source-location,
  production WZZT search, or automatic site/time selection claim is made.

Main files
  local_shared_source_v6.py             isolated exact-LP study
  local_shared_source_v6_theory.md      independent mathematical derivation
  test_local_shared_source_v6.py        independent small mathematical tests
  results_v6/frozen_protocol.json       protocol and effective environment
  results_v6/results.json               all 204 LP results and certificates
  results_v6/artifact_audit_v6.json      actual saved-data matching audit
  results_v6/independent_tests.json      independent tests and source hashes
  results_v6/summary.txt                 concise experiment interpretation
  docs/mobile_ev_probing_report_v2.tex   existing report updated in place

Reproduce from repository root using the existing Topo Python environment:
  & 'D:\apps\miniconda3\envs\Topo\python.exe' -m research_experiments.mobile_ev_probing.local_shared_source_v6 --output research_experiments/mobile_ev_probing/results_v6_replay
  & 'D:\apps\miniconda3\envs\Topo\python.exe' -m pytest -q research_experiments/mobile_ev_probing/test_local_shared_source_v6.py

Use a new output directory for replay; preserve the frozen original artifacts.
The command paths above describe the verified local environment, not a portable
dependency requirement. The study requires Python, NumPy, SciPy/HiGHS; tests
also require pytest. Versions and exact command evidence are archived.

Comparisons and interpretation
  Constant 1.5 kW and 0/3 kW ABBA both use 24 fifteen-minute slots and grid-side
  9 kWh. Same-seed formal observations share background, source and meter error.
  joint: one parameter vector across historical and new observations.
  reset: same observations, a separate vector for each batch, one max-slack
         objective. It is not the new-only information ablation.
  shared: one source-error scalar per slot across all outputs.
  box: independent source-error scalars at each output with the same bounds.
  The nearest wrong-candidate noiseless distance is 0.000375 for joint constant
  charging and 0.000750 for joint ABBA. All box comparisons remain unresolved.
  Both policies exclude both wrong candidates at the formal epsilon=1e-5;
  ABBA increases the separation margin, not this already saturated exclusion
  count. Historical data add no noiseless margin to ABBA in this example.
  Larger error budget and a collapsed internal branch are retained negatives.

Numerical certificate details
  Stored floating LP rows are an outward-rounded relaxation of the exact
  rational model defined by the stored inputs. Dual multipliers are corrected
  by minimizing their residual over a finite variable box. The feasible upper
  bound is independently repaired and checked against the original rational
  rows, not merely the relaxation. tau<=1 is specific to this study; every
  archived original-model feasible objective is below 0.001126, so this cap
  cannot change the original uncapped optimum in these records.
  Tiny exploration float-rounding budgets differ by policy. Only the formal
  evaluation has bitwise-identical epsilon. The independent artifact audit
  verifies this distinction; protocol declarations are not evidence alone.

Physical and statistical limits
  Source budgets and device capabilities are declared example assumptions,
  not field calibrations. EV checks cover slot-average power and a constant-
  efficiency SOC recursion, not switching/ramp transients or AC voltage limits.
  Three independent seeds check arithmetic/model behavior, not cross-network
  statistical significance. Conditional deterministic compatibility is not a
  topology posterior probability or a field 95-percent confidence guarantee.
