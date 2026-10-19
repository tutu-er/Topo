# Continuous-profile MILP and impedance-scale diagnostic protocol
Date: 2026-10-14. New development study, no frozen records changed.

## Claim and scope
Test whether a continuous EV curve optimized by a piecewise-square MILP increases full-AC competing-topology discrimination under a common battery service. First isolate a single site and four 15-minute periods (0--6 kW, 3 kWh grid energy, efficiency .95, initial battery 24 kWh, terminal 26.85 kWh). No V2G/crew/route expansion in this diagnostic. All four local candidates remain in final refits.

## Frozen local construction
For ordered retained candidates a,b, construct the complete whitened historical and future residual r=c+B u+C z. The common line increments appear in both blocks; independent historical/future true-total P/Q and same-interface source states appear in their own blocks. Historical residual intercept is retained. All candidate internal lines are included. Future reference mean/derivative comes only from H estimates and a common H forecast; EV enters total P once.

Unconstrained local profiling gives q_free(u)=min_z ||c+B u+C z||^2=||P c+P B u||^2, where P projects off the columns of C. This is a computational relaxation of the bounded local QP, so q_free<=q_bounded. It may use physically invalid increments; record them, and independently solve bounded local QPs on inspected curves. This free projection does not certify physical or nonlinear separation. Compress the affine residual by an orthogonal transformation (at most T+1 rows), preserving its squared norm.

Reference self-profile q_aa has zero control derivative in the total-P/Q coordinate; verify this numerically rather than assume. Use its computed constant as offset. Score S_ab=(q_ab-q_aa)/2. New MILP optimizes min of the two directions selected by H-only ranking; all four still enter final AC refits. Failure of a required center/fit causes explicit fallback, not removal. Fixed-MLE response-difference objective is an ablation.

## MILP and error
For each compressed residual x in [l,h], express x as a convex combination of the two endpoints of exactly one selected segment. Interpolated square U(x) satisfies 0<=U(x)-x^2<=(segment width)^2/4. Continuous u remains continuous: segment choices are not an action library. Max-min of the approximated sums is a MILP. Its global solver best bound is an upper bound for the true frozen quadratic optimum; evaluated incumbent curves provide lower bounds. Record both, runtime, node count, segment error and gap, including timeouts. No full AC global certificate follows.

## Planned minimum diagnostic, declared before AC runs
Base independent histories: 986101 cherry01, 986102 cherry02, 986103 star.
Internal truth R levels for each cherry: .0003, .003, .012 pu; X=(2/3)R. Star is generated once. Other lines, background path and standardized noise are shared within each seed.
Fit internal-edge R floors: 1e-6, .001, .004 pu; internal X floor=(2/3)R floor (base X floor remains1e-6). Other edges retain old1e-6 minima.
This gives 7 generated histories / 21 fit-domain conditions from 3 independent base seeds, not21 independent histories.
Bounds experiments are distinct:
(1) same truth and data, raise only fit floor: expose prior exclusion and artificial separation;
(2) same original fit domain, strengthen actual internal branch: real signal;
(3) physically covered combinations only: matched prior.
Compute truth-in-domain only in evaluator; excluded truth conditions are model-misspecification diagnostics, never successful topology recovery evidence.

Controls: same-length noEV, same-service constant3kW, continuous profiled MILP, continuous fixed-MLE MILP. Use public temporal voltage-quality profile; no future truth/noise in selector. Use 4 segments initially; repeat8 segments on first valid selected pair to diagnose approximation/gap, not tune by evaluation truth.
History and final full AC: two starts, max_nfev200, tol1e-10, exact linear solver, failures retained. No old rejection threshold, no new risk/probability claim from this pilot.
Record all per-edge parameters, observations, forecasts, seeds, source hashes, MILP bounds, bounded/unbounded local costs, original nonlinear mean response error, actual full-AC margins and physical failures.
On first seed at each truth scale and each floor, separately audit selected/constant curves using full nonlinear joint fits to reference-predicted future means (zero future noise). This compares:
free vs bounded local compensation; bounded linear vs nonlinear AC; predicted mean vs realized noisy outcome.
Cap main MILP at5seconds per solve and pilot at30minutes; checkpoint partial work if cap reached. New seeds required after any strategy tuning.

## Acceptance and next decision
1. Analytic projection, segment bounds, max-min and constraints pass independent tests; tiny grids only audit.
2. All failures/uncertainty retained; source interface, shared R/X, and EV-once checked.
3. Report whether finer segments improve optimization bounds and whether this translates to AC; do not confuse solver error with prediction/linearization error.
4. Compare constant vs optimized paired margins and winners on covered conditions. This is developmental diagnosis, not independent calibrated superiority.
5. If positive competitor inclusion or raised physical branch scale changes little, retain the finding and focus on forecast/compensation limitations. Increasing a fit floor cannot be presented as universally increasing inter-topology distance.

