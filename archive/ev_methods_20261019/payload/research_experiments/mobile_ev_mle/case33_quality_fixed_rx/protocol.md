# case33 fixed-R/X quality-score factorial: protocol before execution

2026-10-18. User-authorized small diagnostic; no EM, EV selection, topology
search, or line-parameter recovery. The original case33 topology and physical
R/X are deliberately given to all six methods. This is an oracle-lines
diagnostic, not a deployable topology-identification result.

## Frozen cases and noise

Use all 32 nonroot loads and the 32 original closed physical lines from the
project's pinned MATPOWER 7.1 case33 table (12.66 kV, 10 MVA). Exact zero virtual
observation leaves preserve internal loads. Root voltage is known, 1 pu;
calibrated biases and the unmetered source are zero. All 32 P/Q/V and both
root P/Q channels have independent Gaussian noise. P/Q may be negative.

Three declared single-slot numerical states (not measured time-series):

|state|load P multiplier|load Q multiplier|PV capacity factor|
|---|---:|---:|---:|
|evening|1.00|1.00|0.00|
|day_pv|0.65|0.70|0.80|
|reverse_pv|0.40|0.50|1.00|

Installed PV is 60% of original total nominal P (2229 kW), allocated to buses
18/25/33 with weights .35/.30/.35. Unit-PF PV changes net P only. This is a
declared benchmark modification, not case33's original DER configuration.
States are not filtered by their scores or voltage ranges.

Two power-meter noise levels: .002 and .01 times nominal channel scale;
Q scale=max(abs(Q_nom),.2 P_nom,1 kvar). P scale=max(P_nom,1 kW). Voltage
sigma=.0005 pu; root P/Q sigma=.001 times nominal total P/abs(Q), with 20-unit
floors. These are declared Gaussian standard deviations, not meter-class
error limits. Seeds 2026101801 through 2026101808. The two noise levels share
standardized draws; all six methods share each observed sample. Three extra
noise-free cases are diagnostics, never used to tune the frozen cases.

## Six methods and common objective

M is LD (lossless squared-voltage DistFlow), DF-I2 (complete downstream
I-squared loss correction), or AC (full nonlinear constant-power AC).
All model outputs use voltage magnitude, not voltage squared.

F_M(z)=.5 ||(z-z_obs)/sigma_z||^2 +
       .5 ||(h_M(z)-y_obs)/sigma_y||^2, z=(P,Q).

FIX uses z=z_obs (no unknown state remains). EIV minimizes F_M in z. There
are no fitted R/X or unmetered-source variables. Normalize optimizer states
as w=(z-z_obs)/sigma_z; do not impose nonnegative-power restrictions.

AC-FIX is the within-FIX reference. AC-EIV is the within-EIV numerical
reference; label it a high-accuracy local solution, not a global certificate.
The AC score at true z is only an oracle noise residual, not the fitted
minimum. Raw FIX-vs-EIV loss reductions do not prove topology improvement.

## Numerical verification and metrics

Use exact existing model Jacobians, double precision, single BLAS thread,
SciPy least_squares max_nfev=400, tolerances 1e-12. Primary LM from observed
P/Q; audits use TRF from observed P/Q and LM from a fixed .25 standardized
sinusoidal perturbation. Retain all starts/failures. AC reference is also
checked from the other models' EIV endpoints (charge this as audit cost).
Reference target: scaled-coordinate gradient infinity norm <=1e-6 and
retained-start objective agreement <=1e-7*(1+J). If target fails, preserve
the case as reference-unqualified rather than silently excluding it.

For each method save self score F_M(z_M), common AC back-score F_AC(z_M),
task AC reference, signed/absolute/relative score error, model score bias,
state-fitting loss, P/Q recovery error, voltage/root errors, evaluations,
timing and solver diagnostics. The identity is

self-reference = (self-AC_back) + (AC_back-reference).

Relative error uses max(reference,1) in the denominator. Report all failures
and optimization-unqualified denominators. Timings separate primary solve,
numerical audits and independent physics replay; cached model construction
and existing AC full-R/X derivative overhead must be disclosed.

Validate full AC against the independent physical-tree backward/forward
solver at generated and fitted states; check all three Jacobians by central
directional differences; check noise-free AC; check EIV <= corresponding
FIX as an attained-value audit. No parameter/state truth enters fitting
apart from the deliberately known topology and R/X.

Save raw data/configuration, all results, dependency/source hashes, source
snapshot, summary, tests and a short report. No change to frozen experiments,
the long PDF or pyproject. Commit only this new research subdirectory.
