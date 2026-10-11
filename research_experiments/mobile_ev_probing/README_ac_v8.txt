v8: passive-first nonlinear AC evidence with path-shared loss remainders
======================================================================

Scope and central question
--------------------------
This is a small research implementation, not production WZZT or a field EV
deployment. Can a common hidden source and spatially shared AC losses be kept
without losing all local-topology discrimination? Does history-first screening
avoid unnecessary EV, and can two locations mitigate a blind single port?

Declared physical model
-----------------------
Three measured terminals, five radial branches, candidates cherry12/13/23.
Complex constant-power AC, balanced equivalent circuit, no shunts/regulators.
Root voltage is known and fixed at 1 pu; all candidate voltages are assumed in
[.9,1.1] pu. Known X/R=.5; theta=2r uses the unchanged v7 coefficient box.
P and Q are measured terminal TOTAL injections, including EV where connected.
Background Q=.3*P_background, EV Q=0. One hidden source is at the common trunk
junction, source Q=.3*P_source, nominal P=.12 pu, allowed deviation .1 pu.
The actual synthetic deviation is uniform +/-.02 pu. Unknown source position,
arbitrary independent R/X, unbalanced phases and field-meter calibration are
outside this prototype. No candidate-wide operating-safety design is claimed.

Mathematical bridge to AC
------------------------
Exact branch loss remainder:
eta_e=(r_e^2+x_e^2)*ell_e
      +2 sum_{f strictly below e}(r_e*r_f+x_e*x_f)*ell_f >= 0.
Terminal squared-voltage drops contain M_h*eta, with the same eta_e shared by
all terminals below that edge. The linear common-source term remains shared.
Its full AC effect is not exactly canceled by C*1=0, because currents/losses
depend on operating voltages. A common trunk loss itself is still common-mode.

KCL gives ell_e <= [(sum descendant apparent powers + source cap if trunk)
                    /v_min]^2.
Source is added only to its actual downstream path, not to every branch.
Square secants and two McCormick upper planes bound eta affinely over the WHOLE
declared coefficient box. P/Q meter boxes enlarge both linear-response error
and current caps. No fitted impedance or simulated true current is used to
justify the bound. Thus every original AC explanation maps into this LP.

The LP shares theta across old/new data and shared source/loss variables across
appropriate outputs. It minimizes additional absolute squared-drop slack tau.
Positive verified LP lower bound excludes a candidate under these assumptions.
Zero LP lower bound means unresolved, not AC feasible. A local nonlinear fit
is used for design only, never as an exclusion lower bound. Floating-point AC
witnesses are labeled numerical, not interval existence proofs.

Each LP uses exact-rational coefficient rounding guards, a box-corrected weak-
duality lower bound and an exact repaired feasible upper point. The upper is
for the outer LP, not an original AC feasible point. The tau<=1 implementation
cap is checked inactive in the recorded cases. The output-box ablation relaxes
path-shared eta to independent terminal eta bounds; its feasible set contains
the path model. Ignoring AC (eta=0) is an explicitly INVALID negative control.

Selection and device semantics
------------------------------
1. Screen all historical candidates with the valid AC outer LP.
2. One retained candidate plus a numerical historical AC witness: conditional
   singleton and skip EV. None: inconsistency. No feasible reference: compute
   before dispatch. Other numerically unresolved candidates are retained.
3. Fit AC references from history alone, using two initializations. Only refs
   whose original historical residual and voltage checks pass are used.
4. Predict future AC data at nominal source; every competitor refits against
   historical plus predicted data with its complete source/parameter/loss set.
5. Equally average nearest-rival predicted slack. This is a nominal scenario
   design score, not v7's squared T objective, a probability, or a guarantee
   against every future source. Score intervals use outward rational averaging.
6. Rank the three stationary endpoint sites and three mobile routes SEPARATELY.
   Constants are comparison baselines. There is no optimized decision between
   traveling versus staying and no time-window optimizer.
7. Refit on actual old/new PQV to determine final exclusions.

Six hours, 24 quarter-hour slots, 20 kW base: stationary constant4.5kW versus
0/6kW endpoint at75% duty, both27kWh. A mobile route dwells11slots at eachof two
sites, has9 charging slots per dwell and2 zero-injection travel slots between.
It also consumes2kWh battery energy for travel. These are declared access/time/
device assumptions, not GIS or charger validation. All nonzero setpoints are
4.5 or6kW; no V2G or independent Q capability is required. Battery60kWh, initial
SOC.4, efficiency.95. Stationary finalSOC.8275; mobile finalSOC.7941667.
Same grid charging energy does not mean same total energy including travel.

Frozen experiment and boundaries
--------------------------------
Formal18cases: 3truths x (flat ordinary, rich ordinary, flat weak) x seeds8201,
8202. Histories12slots; rich history cycles+.06pu at eachterminal. These are
stylized passive trajectories, not sampled real feeder histories. Ten future
actions percase: noEV,3constants,3endpoints,3mobile routes. Source and voltage
noise are paired across every action. No simulation truth enters selection.

Squared-voltage error bound1e-5pu^2 is demanding (roughly5e-6pu amplitude error
near1pu). Actual noise is uniform inside80% of the box. Counts are descriptive
synthetic results, not population accuracy or field confidence. Controls retain
the realized noise while increasing its DECLARED budget; separate PQ controls
add new bounded input measurement errors. These operations are not conflated.

Formal source and dependency hashes are frozen before execution. The original
sequential run was interrupted after complete cases were saved; resume_ac_v8.py
validates and preserves those files, archives incomplete files, distributes only
remaining cases across3processes, and aggregates only after all18finish. This
changes compute scheduling, not protocol, seed set, action library or core.

Files and reproduction
----------------------
ac_shared_v8.py: AC generator, batch AC reference fits, exact guarded outer LP,
                observation-only selection, matched experiment.
test_ac_shared_v8.py: independent rectangular nodal KCL solver, all32impedance
                      corners across3trees, envelope containment, false linear
                      exclusion, path-strengthening, batch/scalar consistency.
audit_ac_v8.py: input hashes/pairing/domain/interval/selection consistency.
controls_ac_v8.py: fixed-action voltage-budget and new PQ-error controls.
resume_ac_v8.py: unchanged-protocol restart and parallel scheduling only.
ac_audit_v8.md and theory_options_v8.md: independent proofs, source checks,
                                      limitations and WZZT integration plan.

Commands from repository root (always use fresh output folders):
& D:\apps\miniconda3\envs\Topo\python.exe -m pytest research_experiments\mobile_ev_probing\test_ac_shared_v8.py -q -p no:cacheprovider
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\ac_shared_v8.py --mode formal --output research_experiments\mobile_ev_probing\results_v8\new_ac
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\audit_ac_v8.py research_experiments\mobile_ev_probing\results_v8\new_ac
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\controls_ac_v8.py research_experiments\mobile_ev_probing\results_v8\new_ac research_experiments\mobile_ev_probing\results_v8\new_controls

Audit limitation: it verifies saved exact intervals and physical data, but does
not reconstruct dual multipliers that are not stored. Independent tests and
selected fresh LP replays supplement this check. Existing v5 AC results are
historical, separately audited evidence, not part of this new formal table.

Verified results (2026-10-12)
-----------------------------
All18 formal episodes and180 paired action batches completed. History alone
separates6/18 (all six rich ordinary cases), so those controllers skip EV.
Selected single endpoint and same-site constant both separate11/18; fixedsite1
separates10/18. Selectedmobile and fixedmobile12 both separate12/18. In the six
flat ordinary cases these counts are0,5,5,4,6,6 respectively. All six weak cases
remain unresolved. No field success rate or optimized-route superiority follows.

Among flat ordinary cases, mean nearest-wrong lower margin is.0005909422 for
selected endpoints versus.0004284580 for matched constants: margin improves,
count does not. Cherry23/8201 yields zero tied single-site scores, lexicographic
site1 fails, mobile12 succeeds. Cherry13/8202 yields near-zero scores but actual
site1 succeeds. A low nominal design score does not certify EV is useless.

Path sharing separates all six rich cases without EV; output-wise loss boxes
separate none on identical no-EV data.204/540 candidate-action comparisons have
path lower > output upper, proving strict relaxation strengthening. Both valid
AC models retain the truth in all180 batches. Invalid eta=0 linear screening
falsely excludes truth in58/180.2322 saved LP intervals pass audit, maximumgap
3.5600884763e-10;9 raw-input LP replays agree.19 independent physical/model tests
pass, including all32 parameter-box corners for eachtree. Actual voltages range
.96781133 to.99183168; maxAC residual2.805e-16. These are solver checks, not field
model validation. policy_summary.json is derived by summarize_ac_v8.py.

Three predeclared flat ordinary/8201 truth cases, fixed main-study actions:
increasing only DECLARED voltage-error bound to1e-4 or5e-4 gives single/fixedsite1
2/3 and selected/fixedmobile12 3/3. Bound.002 gives0/3 for everypolicy. Realized
voltage noise is unchanged. A separate actual P/Q measurement-error addition
(<=.0008pu, declaredhalfwidth.001pu) preserves2/3,2/3,3/3,3/3.96 distinct control
LPs pass; all truths retained; maxgap3.827e-16. This is not a three-case test of
larger actual voltage noise and not evidence that the original nonlinear models
are intrinsically indistinguishable at the larger declared bound.
