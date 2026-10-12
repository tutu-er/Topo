v9: passive RNJ/WZZT proposals, balanced AC evidence, and metered EV probing
Date: 2026-10-12

Scope
-----
Research integration calling the existing rnj_wzzt_core estimators. Production
pipeline defaults are unchanged. No global topology coverage or probability
claim. Synthetic 6/12-terminal trees have 12/23 physical buses. Known X/R=.5,
root physical voltage=1, one unmetered source at the common trunk endpoint,
all physical voltages in [.9,1.1] pu. These assumptions are part of each claim.

Core entry points
-----------------
passive_bridge_v9.build_passive_candidates(P,Q,Y,...)
  Calls fit_projected_sensitivity, rooted_neighbor_joining, and the existing
  research fit_source_aware_wzzt adapter. RNJ supports are proposals, not fixed
  constraints. WZZT receives certified_supports=() and restarts at singletons.
  WZZT separate R/X/source-path fits propose trees; AC parameters are refitted.
  Local family deletion/exchange and terminal relabeling expand proposals.
  Finite search budget is not an exclusion certificate.

ac_general_v9.RadialCandidate.from_supports(name,n,supports)
  Converts a laminar terminal family to explicit physical tree/path incidence.
  Common trunk and singleton terminal edges are included automatically.

ac_general_v9.solve_outer(observed,candidate,box=...,shared_voltage_gain=...)
  AC outer LP: common source, shared edge losses, observed PQ uncertainty,
  optional fixed meter gains across all historical and new rows. Only an
  outward-rounded, residual-corrected positive dual lower bound excludes.
  Zero/slack upper bound/solver failure never establishes original AC feasibility.

integrated_study_v9.choose_probe(history,bank,evidence)
  Uses up to two local AC references and a centered LINEAR joint-refit proxy.
  References need not be witnesses in the shared-gain domain. Scores are
  heuristics, not T-optimality certificates, rejection bounds, or probabilities.
  The selector sees historical readings and their mean-load forecast only.

integrated_study_v9.append_and_refit(...)
  Reruns passive proposals on all actual old/new observations. Retains all old
  unresolved hypotheses even when new proposal count hits the search budget.
  Compare evidence gains on the common INITIAL bank; separately report updated
  bank size, true-family coverage, and top-ranked clade precision/recall/F1.

Metering and comparisons
-----------------------
P/Q are total metered load, already including EV. Positive denotes consumption.
All readings include actual synthetic noise. Voltage total amplitude bounds
.001 pu compare fixed-gain-dominated and fully time-varying regimes; .0001 pu
is a separate precision control. Fixed gain/jitter split is an EXTRA assumption,
not promised by the manufacturer's total accuracy. PQ boxes are derived from
observed readings with outward rounding. Source/gain/line parameters share
only the declared physical relationships; no truth enters candidate selection.

The fixed-gain original physical model is structured, but its current outer LP
and the independent-voltage-box outer LP are not proven nested. Neither may
be advertised as uniformly tighter. Both are run on the same actual data.

All charging policies deliver 27 kWh over 6 h. Single-site modulation is 0/6 kW;
matched constant is 4.5 kW. Mobile policies have two dwells, two travel slots,
and additional 2 kWh battery travel consumption. EV site labels are 0-based
TERMINAL labels, not the physical root. Installed meters remain stationary.
Each slot is a steady state: temporal aggregation error, field connection,
real travel feasibility, and all-candidate pre-dispatch safety are not verified.

Reproduction (run from repository root with NumPy/SciPy/pandas/pytest available)
-----------------------------------------------------------------------------
python -m pytest research_experiments/mobile_ev_probing/test_passive_bridge_v9.py research_experiments/mobile_ev_probing/test_ac_general_v9.py research_experiments/mobile_ev_probing/test_integrated_study_v9.py research_experiments/mobile_ev_probing/test_wzzt_adapter.py

Use a NEW output directory; never overwrite frozen completed results:
python research_experiments/mobile_ev_probing/integrated_study_v9.py --prepare-only --output <new_directory>
python research_experiments/mobile_ev_probing/run_parallel_v9.py <new_directory> --workers 3
python research_experiments/mobile_ev_probing/audit_integrated_v9.py <new_directory> --replays 3
python research_experiments/mobile_ev_probing/diagnose_integrated_v9.py <new_directory> --require-complete

Protocol hashes cover raw source bytes including production dependencies. A
different checkout's line endings may change hashes; generate a fresh protocol
for fresh runs rather than altering archived manifests. Fixed synthetic seed
9101 provides mechanism checks, not repeated-sampling field accuracy estimates.

Audit and diagnosis
-------------------
audit_integrated_v9.py checks source/input/truth hashes, original data pairing,
independent nodal AC, all physical/measurement bounds, EV SOC, true-family
coverage, no false true-model exclusion, preserved candidates, and LP replay.
diagnose_integrated_v9.py compares actual verified evidence on fixed initial
banks and constructs post-hoc original-AC branch-deletion explanations. It fits
one gain interval per meter across ALL old/new samples and checks jitter
directly. Numerical witnesses include margins and independent KCL residuals;
they are not interval existence proofs. Failure to construct a witness does not
prove absence of other nonlinear explanations. Truth is diagnosis-only.

Results and report
------------------
results_v9/formal/protocol.json       frozen cases and source hashes
results_v9/formal/<case>/result.json  actual candidates/actions/AC bounds/data hashes
results_v9/formal/artifact_audit.json complete audit (partial audit named separately)
results_v9/formal/diagnostics.json    selection failures and AC deletion witnesses
results_v9/*_tests/                  archived 14 + 25 + 39 tests
results_v9/pilot/                    earlier exploratory negative result; not pooled
docs/mobile_ev_probing_report_v2.tex editable report, current overview in first section

The old pilot used a different candidate refresh/rounding protocol. Its exact
driver snapshot is retained. Do not combine its counts with the frozen study.
PDF is compiled in place; verify_report_pdf_v9.py prepares content/render checks.
Visual-review completion is recorded only after inspecting the generated pages.
