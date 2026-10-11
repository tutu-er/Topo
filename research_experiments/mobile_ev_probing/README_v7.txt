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
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\audit_design_v7.py research_experiments\mobile_ev_probing\results_v7\new_controls --controls
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

Frozen formal results (27 episodes, source SHA-256
5d7fb37ad5e6c4e19def9d13333f39bc8b1792f881d29ef6c6e740e3dfa24b13)
--------------------------------------------------------------------------
972 design QPs + 81 historical-reference QPs; 486 final shared-source joint LPs.
Both wrong candidates excluded: local 15/27, uniform committee 18/27, fixed
site 1 endpoint 12/27. The same selected-site constant policies achieve the
same exclusion counts at this error budget. Endpoint excitation improves the
margin, not the success count. All 9 weak-edge episodes remain unresolved.
All actual truth candidates are retained. All saved-input/hash/domain/pairing
checks pass in formal/artifact_audit.json. Exact upper bounds are below the
implementation tau cap 1 (maximum .00134708992891096), so the cap is inactive.
Maximum design-QP gap: 9.831085430514451e-10; historical-reference normalized
midpoint objective gap: 9.957578297506894e-7; final-LP gap: 5.595933717367997e-10.

Important negative: true cherry13, asymmetric profile, all three seeds select
the wrong local reference cherry23. Its site2/site3 scores tie; lexicographic
site2 execution excludes no rival, while fixed site1 and committee-selected
site3 both exclude two. Do not call the local site a uniquely optimal choice.
Among 18 strong-edge episodes the L2-selected site hits the best posterior
L-infinity margin site in 8 local / 10 committee / 6 fixed cases, not universally.
The other 9 weak-edge cases have all-zero tied margins; reporting them as
successful oracle matches would hide the absence of information.

Mean endpoint / same-site constant certified nearest-wrong margin:
local       .0005479512449679735 / .00027269744008658276
committee   .0006975323266846827 / .00034725165329833874
fixed site1 .00046462644985143   / .00023125177910579816
These are descriptive averages over this frozen synthetic grid, not confidence
intervals, population rates, topology probabilities, or universal dominance.

The 57 prespecified control LPs also pass exact certificates and saved-artifact
audits. Source perfectly following EV retains site1 endpoint margin
.00074706902055583; larger epsilon=.002 leaves all six actions unresolved.
Independent source boxes remove the gain. Historical parameter reset removes
constant charging discrimination but leaves within-window endpoint excitation
useful. Rich history alone excludes both wrong candidates with margin lower
9.396797592990971e-5, so EV is unnecessary in that control. All truths retained.
Maximum control-LP gap 4.195382034853368e-16, maximum verified upper bound
.0007471440205558415 < implementation cap 1.
