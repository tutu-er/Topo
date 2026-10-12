# Frozen initial protocol (2026-10-12)

Central question: under an explicitly calibrated Gaussian P/Q/V meter model,
can exact AC joint refitting discriminate a three-terminal local clade when
arbitrary per-time hidden P/Q enters one common junction, and can feasible EV
actions add information beyond equal-duration/equal-energy controls?

Implementation is a research experiment, separate from production. Local
multistart solves yield feasible likelihood values, not global maxima or
certificates. Free R and X are shared across history/new data; terminal true
total P/Q and hidden source P/Q are per-time states. Root voltage and fixed
meter offsets are independently calibrated. Feeder master P/Q anchors source
amplitudes. Known independent Gaussian sigmas are synthetic assumptions.

Selector observes only historical meters, candidate metadata, forecasts made
from historical fits, charger access, and fixed constraints. It never receives
true topology, true impedances, future source, background or future noise.
EV is added once to terminal total P; Q control defaults to zero.

Start n=3, history=12, future=8 quarter-hour steady-state snapshots. Candidate
repair bank is star plus all three two-terminal clades, ordered from passive
RNJ/WZZT proposals; this is complete only for this declared reduced local
tree class. RNJ X/R=.5 is a proposal proxy only, never an AC fit restriction.
WZZT failures/budgets are retained, not treated as evidence. Larger n=5 is
only attempted after physical/noiseless and local-fit tests pass.

Controls: no EV of equal duration; constant charging at fixed site0; pulsed
charging at fixed site0; randomly selected feasible action; designed action.
All charging actions have identical grid energy=6kWh, base=20kW, max6kW,
dt=.25h, battery60kWh, efficiency.95, initial SOC.4, bounds[.1,.9].
Availability and travel slots are enforced; movement consumes battery energy.
No-EV is explicitly not an equal-energy control.

Independent calibration and evaluation seeds will be recorded in the actual
run configuration before execution. Calibration reruns passive proposals,
historical fit, adaptive selection and joint fit; thresholds are simulation
quantiles for the declared generating scenarios, not chi-square assumptions.
Evaluation reports false rejection of the generating candidate, removal of
wrong candidates, unresolved true/weak clades, missing-truth diagnostics,
impedance/source errors, local likelihood gaps, model adequacy and runtime.
Calibration distributions and failure counts are preserved, never relabeled
topology probabilities. Single-seed pilot results cannot establish calibration.

Budgets: initial fit max_nfev150, two starts; pilot decides affordable formal
seed counts. Track failed starts, AC constraints, restart loss spread, bounds
and profiled sensitivity rank. Inner design local minima give upper estimates
of competitor best attainable loss; a large value is not guaranteed separation.
Optimization failures do not count as rejection.

15-minute snapshots are synthetic steady states; equations at averaged loads
need not equal average measured voltages. Balanced constant-power AC, fixed
source junction, radial topology and time-invariant impedances delimit scope.
