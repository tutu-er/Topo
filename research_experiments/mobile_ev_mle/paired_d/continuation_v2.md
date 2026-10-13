# continuation_v2 execution protocol

The original eight-hour cutoff remains recorded in runs_v1 and is not changed.
The user authorized continuation at 2026-10-13 01:54 UTC. This new execution
extends only the no-new-history and final checkpoint deadline to
2026-10-13 03:00 UTC. Work here is outside the original eight-hour budget.

The statistics, calibration seeds 940001-940032, evaluation seeds
950001-950048, 32/48 requested counts, candidate bank, two methods, H14 R/X
initialization, common fit seed, budgets, risk allocation, failure retention,
and paired comparison rules are unchanged. The two pilot seeds are
941101/941102; repeating them checks this integrated runner and adds no new
independent pilot units. Pilot observations never enter formal denominators.

The original runs_v1/frozen.json, pilot and protocol are immutable. This
continuation has its own freeze, output and source hashes. Evaluation can
start only after all 32 calibration histories have been attempted and the
new calibration thresholds have been written and frozen. Partial batches
and failures remain recorded; neither counts nor deadlines will be padded.
Thread count is one and Windows process priority must be BelowNormal.
