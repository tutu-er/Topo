# Additional 900-second solve of the twelve fixed flexibility quotations

2026-11-14. Locked before extended optimization. The user requested increased solve time and complete records.

## Fixed inputs and mathematical problem

Revisit all twelve completed quotations from `flexibility_resource_paired_20261114_v1` (four new36 cases) and `flexibility_resource_fresh_20261114_v1` (four old23 and four new36). Retain exact training/validation, frozen candidate parameters, shared sigma, 99% target set, alpha, tie and 1e-8 gate buffer, all ten direction guards, original P contracts and Q capacities, prices, forecasts, response/meter seeds and original final native-bank MSE rule. No fitting, resource adjustment, risk reallocation or selection of favorable paths.

Use the already derived exact elimination of distance auxiliaries. In this case four absolute gates are identically satisfied and the remaining gate is J >= 6.7321002053; keep all direction guards and resource constraints. General double-interval cases fall back to the original solver rather than dropping a branch. The returned point is checked against the original full-prediction inequalities and resource domain. This is equivalent algebra, not a convex inner approximation. Report original numerical feasibility tolerances and buffered residuals explicitly.

## Initialization, runtime and bounds

Each quotation restarts from its own verified final 300-second incumbent plus its sealed previous shared seed array. Match exact candidate, forecast and full resource configuration; use no former execution, AC observation or success label to choose a seed. The predecessor incumbent is retained if no cheaper verified point is obtained. A fixed feasible initialization does not establish optimality or prove that every nonconvex solver needs one.

Budget: 900 additional seconds per quote, including current seed loading/auditing, model construction and solver call. Previous 300-second computation and its earlier 180/30-second starts are separate sunk costs. Rebuilding a model with a warm start does not resume the previous branch-and-bound tree. Both time and algebraic representation change relative to the last batch; do not attribute all change solely to extra time.

Use one Gurobi solver thread per quote. Run the four paired tasks with two workers and the eight former-fresh tasks with four workers, at most six workers simultaneously. Initial host inspection found 32 logical CPUs and about 9 GiB available memory. Record actual elapsed time and overrun rather than treating a TimeLimit setting as a hard wall-clock deadline. Do not increase the workload while this batch is active.

Save an individual private solver log, sampled actual callback time/best incumbent/bound/node count, raw final solver attributes including ObjBound and ObjBoundC when available, selected valid lower bound, verified upper bound, gap, status, initialization source, fee improvement and maximum action change. A numerical change below 1e-6 yuan is not material improvement. Nonfinite attributes are unavailable, not fabricated zeros; the universal nonnegative-cost lower bound may be reported separately. If a numerical bound conflicts with a verified feasible incumbent, withdraw the conflicting bound and record why.

## Re-evaluation and independent checking

Seal each new plan before execution. Reuse the existing AC bridge, load day, forecast, whole-day response multipliers and P/Q/V errors for no procurement, optimized procurement and same-planned-fee rotation. Any changed action must be evaluated on all eight original execution paths; the reused seeds are no longer an independent confirmation. Replaying an unchanged plan provides a consistency check, not additional evidence.

Compare each extended quote with its own 300-second predecessor: requested fee and valid bounds first, followed by same-path native topology recovery, correct/wrong high relative weights, clean and measured evidence increments, P/Q delivery, settlement fee, all-bus voltages, currents, root supply and losses. Lower procurement cost alone does not ensure better identification. Preserve every adverse change, solver timeout, unchanged incumbent, failed execution and unresolved bound. Thermal limits remain unknown because original ampacities/nameplates are absent.

Independently replay the original algebra, resource bounds, exact score/forecast/configuration and random-array identity, complete native candidate selection and saved balanced AC states. Keep the raw sequences, fitted coefficients, per-quote actions/logs and all quote trajectories in ignored outputs. Public commits contain code, this protocol and aggregate results only. Preserve previous completed output directories, locks, sources and all failures. No production-default changes and no new candidate-bank scope repairs in this comparison.
