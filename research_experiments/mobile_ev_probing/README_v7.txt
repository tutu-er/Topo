v7: historical-set constrained local T experimental design
=========================================================

Scope: a synthetic three-terminal linear squared-voltage-drop model with a
known common source response direction. This is not a Dette paper replication,
AC validation, production WZZT implementation, or topology probability method.

Central question: can historical observations alone select an EV probing node
whose actual refitted topology discrimination improves on fixed node 1 and
same-energy constant charging? A wrong reference and no benefit are allowed.

Model and information flow
--------------------------
R_h = a 11' + b z_h z_h' + diag(c), h in cherries 12,13,23.
y_t = R_h p_t + a * source_nominal_t * 1 + w_t * 1 + e_t.
|w_t| <= a B_t; |e_ti| <= epsilon_ti. P includes EV; Q=0.
One theta must explain both historical and new observations. The historical
parameter set is the exact projection of its shared-source feasibility LP.
Shared w is eliminated by intersecting its three output residual intervals
and [-a B,a B]; no source smoothness or EV independence is imposed.

Historical reference: closest normalized coefficient vector to the parameter
box midpoint for each candidate. The local reference has smallest certified
midpoint-distance interval, with explicit tolerance and lexicographic tie
resolution. The committee equally averages per-reference nearest-rival design
scores; these are design weights, not topology probabilities.

Design: minimize sum ||C(R_ref-R_h(theta))p_new||^2 over the full historical
parameter set for each rival; maximize the nearest-rival score across the
three feasible sites. C1=0 and CC'=I. The implementation uses the exact identity
||C d||^2 = [(d1-d2)^2+(d1-d3)^2+(d2-d3)^2]/3.
Reference/selection functions receive only historical observations and declared
future background P. Simulated truth/future Y enter evaluation only.

Six actions: site 1,2,3 x constant 1.5 kW or 0/3 kW endpoint charging, 24 slots
of 15 min, six hours, nine kWh grid energy in both policies. The endpoint curve
is 0,3,3,0 repeated; its ordering has no intrinsic advantage in this time-
homogeneous model. Fixed-energy endpoint dominance is checked independently.

Coefficient domain differs from v6 only in b lower bound: 1e-6, not .005. All
v7 strategies use this same expanded domain, including the weak-edge profile.
Three truth topologies x symmetric/asymmetric/weak profiles x seeds 7201-7203
= 27 formal episodes, 6 actions, 3 final candidate fits = 486 final LPs.
The two pilot seeds are 601 and 602. They are excluded from formal aggregates.

Numerical verification
----------------------
QP incumbents are checked against exact rational projected constraints.
Convexity plus a tangent plane gives a linear support problem over an outward-
rounded historical domain. Its LP dual, corrected over [0,1]^5, gives an exact
rational lower bound. The upper/lower gap, solver status, and repair fraction
are saved. Score-interval overlap is explicitly unresolved/tied; deterministic
execution does not imply a unique optimal node. An inward feasibility solver
failure cannot remove a candidate; it makes design unresolved.

Evaluation calls unchanged local_shared_source_v6.solve_lp with the v7 box.
It retains full original outputs, joint history, common source, and certified
additional L-infinity slack. Design L2 score need not rank actions identically
to final L-infinity margin. Oracle site comparisons use future truth only as
a post-hoc evaluation ceiling.

Reproduce (PowerShell, repository root)
--------------------------------------
& D:\apps\miniconda3\envs\Topo\python.exe -m pytest research_experiments\mobile_ev_probing\test_design_v7.py -q -p no:cacheprovider
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\design_v7.py --mode pilot --output research_experiments\mobile_ev_probing\results_v7\new_pilot
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\design_v7.py --mode formal --output research_experiments\mobile_ev_probing\results_v7\new_formal
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\controls_v7.py --output research_experiments\mobile_ev_probing\results_v7\new_controls
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\audit_design_v7.py research_experiments\mobile_ev_probing\results_v7\new_formal
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\controls_v7.py --formal research_experiments\mobile_ev_probing\results_v7\new_formal\formal_results.json --output research_experiments\mobile_ev_probing\results_v7\new_formal\reference_sensitivity.json

Use a fresh output directory: a frozen formal protocol is never overwritten.
Every episode saves observations and evaluation truth separately, hashes,
all design score intervals, decisions/ties, all final LP certificates, and
negative outcomes. Frozen protocol records source and v6 hashes, package
versions, seeds, numerical tolerances, and parameter settings.

The 57 extra control LPs are separate from formal aggregates: fully correlated
source/EV, a larger measurement box, informative history without new probing,
and same-data shared/box and joint/reset comparisons. Correlated-source actions
intentionally have different source series, because the source follows EV;
this control tests the absence of an independence requirement. Formal actions
use exactly matched source series and generated noise.

Pilot audit
-----------
pilot/initial_large_gap_results.json and pilot/design_before_incumbent_repair.py
preserve the initial numerical failure: overly large scaling and rejection of
slightly infeasible SLSQP points produced broad certified intervals. No claim
of optimal ranking was made. Historical Chebyshev interior starts, convex
incumbent repair, and scaling 1e4 corrected this without changing the pilot
data or formal seed table. pilot_repaired/ preserves that follow-up.

Known limitations
-----------------
No unknown source location, reactive control, travel, AC physics, or device
transients. SOC is an energy check for a 50 kWh vehicle, charge efficiency .95,
initial .4, bounds [.2,.9]. The .571 final SOC is feasible, not field validation.
No calibrated stochastic error law or probability is assumed. Finite candidate
exclusion does not imply a global topology certificate or posterior certainty.
