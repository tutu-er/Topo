# AC audit for the passive-first EV workflow

Audit date: 2026-10-12. This note audits existing source and stored results; it does not rerun the v5 large experiment. The proposed v8 extension below must be judged by its own tests.

## Direct conclusion

`design_v7.py` explicitly models a three-terminal **linear squared-voltage-drop** problem with P control, a known common source direction, and no AC. Its exact convex inner solves establish validity for that model. They do not establish validity for nonlinear AC observations. The existing v5 AC certificate and v7 common-source design are separate implementations.

The v7 selector also constructs representatives for all three historical candidates; it does not first reject candidates and stop when historical data identify one. Its rich-history control demonstrates that this stopping policy is possible, not that the policy is integrated. A production interface must first obtain plausible candidates from passive WZZT/RNJ, audit their compatibility, and send only unresolved structures to EV design.

## Exact branch identities

Use per-unit line impedances and positive load consumption. On a balanced radial network without shunts, let edge e=(i,j), voltage square v_i=|V_i|^2, sending-end flow P_e+jQ_e, and current square l_e=|I_e|^2. The exact identities are

\[
P_e=p_j+\sum_{f:j\to k}P_f+r_e l_e,\quad
Q_e=q_j+\sum_{f:j\to k}Q_f+x_e l_e,
\]
\[
v_i-v_j=2(r_eP_e+x_eQ_e)-(r_e^2+x_e^2)l_e,
\qquad v_i l_e=P_e^2+Q_e^2.
\]

Expanding downstream flows gives

\[
v_i-v_j=2r_e\sum_{k\downarrow e}p_k
 +2x_e\sum_{k\downarrow e}q_k+\eta_e,
\]
\[
\eta_e=(r_e^2+x_e^2)l_e
 +2\sum_{f\prec e}(r_er_f+x_ex_f)l_f\ge0.
\]

Here f is strictly downstream of e; every downstream loss appears, including losses outside a particular terminal's final branch. If A is terminal-by-edge path incidence, terminal drop satisfies D=2A diag(r) A' p+2A diag(x) A' q+A eta, with extra source injections represented by their actual path-incidence columns. Nonnegative R/X makes eta nonnegative even if measured terminal injections include exports.

For a source at the common root-trunk endpoint and fixed x_e=kappa r_e, set theta_e=2r_e and effective power p_tilde=p+kappa q. This permits a unity-power-factor EV even when background Q is nonzero. In the current v8 case, background Q=.3 P_background and source Q=.3 P_source; EV Q=0. Then

\[
D=R_h(\theta)\widetilde p+\theta_0(\bar s+\delta)\mathbf1+A\eta,
\quad |\delta|\le B,
\]
\[
\eta_e=\frac{1+\kappa^2}{4}
\left(\theta_e^2 l_e+2\sum_{f\prec e}\theta_e\theta_f l_f\right).
\]

Here source s means P_source+kappa Q_source, not active source power alone. Thus C1=0 removes the **linear** source term and the common-trunk remainder column, but not all source dependence: downstream currents and losses change with the source-induced operating voltage. In general C A eta(s) is not constant. Exact nonlinear cancellation must not be claimed.

## Valid local AC outer envelope

KCL gives a bound that does not require line-current meters. For a terminal measurement box, define

\[
s^+_{kt}=\sqrt{(|P^{obs}_{kt}|+b^P_{kt})^2+(|Q^{obs}_{kt}|+b^Q_{kt})^2},
\qquad v^-_{kt}=\max(v_{min},V^{obs}_{kt}-b^V_{kt}).
\]
\[
\sqrt{\bar l_{et}}=
\sum_{k\downarrow e}\frac{s^+_{kt}}{v^-_{kt}}
+\frac{S^{src,+}_t}{v_{min}}\,1\{\text{source below e}\}.
\]

The bounds require compatible positive voltage intervals and the declared engineering domain. When the source is known at the common junction, only the root trunk includes its direct current contribution. Unknown source locations need a more conservative cap or a union over location hypotheses. The original v5 uses one shared v_min and adds the total source cap to every edge; that is valid but looser.

With theta in [L,U], theta_e^2 is bounded above by (L_e+U_e)theta_e-L_eU_e. Each cross product has two global affine upper planes:

\[
\theta_e\theta_f\le U_f\theta_e+L_e\theta_f-L_eU_f,
\qquad
\theta_e\theta_f\le L_f\theta_e+U_e\theta_f-U_eL_f.
\]

All current weights are nonnegative. Substituting either family yields valid affine upper bounds F_e^0(theta), F_e^1(theta). Retain one eta_et variable per edge and time, common to all downstream terminal equations:

\[
0\le\eta_{et}\le F_e^0(\theta),\quad
\eta_{et}\le F_e^1(\theta),\qquad
|w_t|\le\theta_0B_t.
\]
\[
\min\tau\ge0:\quad
|D^{obs}_{it}-[R_h(\theta)\widetilde p_t]_i-\theta_0\bar s_t-w_t-[A\eta_t]_i|
\le\epsilon_{it}+\tau.
\]

This is an outer LP, not exact AC fitting. The shared edge remainder preserves path coupling that output-wise remainder boxes discard. It is at least as tight as the corresponding output-wise box relaxation when both use consistent bounds. It does not establish exact physical feasibility of retained candidates, and it need not be tight enough to exclude them.

If only a constant terminal bound 0<=eta_i<=F_i is retained, shifting Dobs to Dobs-F/2 and widening epsilon to epsilon+F/2 represents that one-sided interval exactly. Widening to epsilon+F without a shift is valid but unnecessarily loses the sign information.

## What exclusion proves

For a stored LP min c'z subject to Mz<=b and l<=z<=u, any y<=0 gives

\[
L(y)=y'b+\sum_j\min\{(c-M'y)_jl_j,(c-M'y)_ju_j\}
\le\tau_h^{LP}.
\]

For rejection, only the implication “every admitted exact AC explanation maps to a zero-slack LP point” is needed. A reliable L>0 rules out such an explanation. No comparison with a differently defined positive-slack AC objective is required: enlarging measurement errors could itself change the current envelope. Solver status, fitted loss, design T score, or a local nonlinear optimizer's positive residual is insufficient.

Keep three outputs separate: (1) incompatible under the declared domain when a reliable lower bound is positive; (2) an explicit AC witness compatible with observations when one is validated; (3) unresolved otherwise. A zero outer-LP lower bound does not supply an AC witness. If the complete model and simultaneous measurement event cover the truth, true-candidate retention follows. Probability calibration is external, and finite-bank identification is not an all-topology result.

## Existing v5 results directly checked

The twelve compressed candidate archives in `results_v5/bridge/<case>/` were read and positive `certified_lower_bound` values counted again. They agree with stored summaries:

| Case | Wrong candidates | Global | Local current | Local affine loss |
|---|---:|---:|---:|---:|
| paper15 | 93 | 0 | 24 | 79 |
| soumalas11 | 45 | 0 | 27 | 38 |
| flynn16 | 102 | 0 | 0 | 39 |
| pengwah18 | 136 | 0 | 41 | 121 |
| Total | 376 | 0 | 92 | 277 |

All twelve true-candidate bounds are zero. Maximum recorded certificate gap is 1.4308e-11. These are the same 96 observations across the three v5 relaxations, but a new explicitly recorded noise stream relative to the older audit; they are not an exact replay of older raw observations. These bridge cases have **zero hidden-source cap**, so the table cannot validate common-source AC performance.

The archived v5 full-study audit has 332 policy records, 284 distinct LP batches, and 22,276 candidate LP results. It reports zero false exclusions, 320 records satisfying all physical/source/noise conditions, and 12 out-of-domain records also retained. Policy records are not independent trials. The historical verification record reports 108 tests on 2026-10-09; they were not rerun in this audit.

## Reusable APIs and remaining work

- `observation_lp_v5.solve_observation_lp`: whole-box AC outer certificate with shared R/X, measured P/Q error, local currents, affine loss envelopes, and exact-rational corrected dual lower bounds. Unknown sources are relaxed independently by output; its common-source coupling is insufficient for the v7 question.
- `rnj_wzzt.models.ac_powerflow.solve_ac_power_flow_snapshot` / `solve_ac_power_flow_timeseries`: backward-forward sweep physical simulator. Return convergence and current data. Check nodal/DistFlow residuals independently; convergence status alone does not certify all algebraic constraints.
- `observation_study_v5.historical_design`: frozen geometry/PQ-load heuristic, not refitted AC T-optimal design. The node score uses equal midpoint impedances and the time score uses total apparent-power squared; observed voltage is not used to estimate the candidate parameters.

A minimal followup should compare rich passive history, weak passive history, and weak structural separation. It should audit history first; stop when one candidate remains; otherwise compare same-energy constant charging, a fixed site, and an observation-only selected site. Generate observations by nonlinear AC, use the same shared parameters on history and new data, preserve unresolved outcomes, and evaluate true-candidate retention and wrong-candidate exclusion separately. A promising concrete contribution is preserving **shared source and shared edge-loss explanations** while asking for the least additional excitation needed to resolve a local ambiguity. Generic EV probing, committee averaging, and convex bounds alone are not new.

## New independent v8 verification

`test_ac_shared_v8.py` supplies a rectangular nodal-KCL solver independent of the implementation's path-matrix fixed point. The final evidence-recorded run has 19 tests passing in 1.87 seconds on 2026-10-12 local time; command, stdout, JUnit, and before/after source hashes are in `results_v8/ac_tests/`. These cover the three candidate trees at lower/upper/mixed parameter values; all 32 parameter-box corners for each topology; upper-plane containment using independently reconstructed physical branch voltage drops; true-candidate retention at measurement-error corners; strict path-sharing improvement over the matching output relaxation; source-dependent nonlinear contrasts; false exclusion when AC losses are omitted; and agreement of batched, scalar, and independent nodal solvers.

Concrete negative control: `make_data('cherry12', [.02,.01,.06,.06,.06], 9801, 'rich', 0, 'endpoint')` followed by a linear-only fit excludes the true candidate with lower bound 9.3902699250e-6 pu^2, despite allowing parameter refitting. The AC outer model retains it. This is a reproducible single case, not an estimated failure rate.

On the same symmetric parameters and seed with flat history, the wrong `cherry13` candidate has lower bound 0.00070900553117 under shared path remainders, versus upper bound 0.00022457200343 under output-wise remainders. This establishes a strict gain from path coupling on identical observations and parameter bounds in that case. It does not guarantee such a gain for every candidate.

For the asymmetric parameters [.027,.018,.035,.081,.052], p=[.5,.2,.2], q=[.06,.06,.06], and source P changing .02 to .22 with Q=.3P, the max-minus-min across terminal squared-drop changes is 1.8203e-6. The projected common-source response is therefore not exactly zero. Its size can nevertheless be below measurement noise; the purpose of the envelope is to make that approximation controlled, not to assert that AC necessarily destroys EV discrimination.

## Frozen v8 experiment audit

The completed `results_v8/ac_formal/artifact_audit.json` passes all checks for 18 planned episodes, 180 observation batches, and 2,322 stored LP records (history, design, and final evaluation together). The largest recorded certified interval width is 3.5600884763e-10. The protocol, core/dependency hashes, separate observation/truth files, paired historical observations/source/noise, and physical/noise domains were checked. In 204 candidate/action comparisons, the path-shared relaxation's lower bound exceeds the output-wise relaxation's upper bound. This is a strict same-domain tightening in those comparisons, not a count of independent experiments.

Nine predeclared LPs were rebuilt from raw observations: fixed mobile route 1-to-2, flat symmetric history, seed 8201, each of the three truths tested against each of the three candidates. Every rebuilt exact bound interval overlaps its recorded interval and every classification agrees. These are replays through the inspected implementation, not a second independent optimizer. Saved dual multipliers are not independently reconstructed because the runner does not save them.

| Six episodes per group | History alone identifies | Selected stationary EV identifies | Selected mobile EV identifies | Fixed mobile 1-to-2 identifies |
|---|---:|---:|---:|---:|
| Informative symmetric history | 6 | Skipped | Skipped | 6, counterfactual |
| Flat symmetric history | 0 | 5 | 6 | 6 |
| Flat history, weak cherry edge | 0 | 0 | 0 | 0 |

Identification here means that both wrong members of the three-candidate bank are excluded and the truth is retained. All six informative-history singletons also have checked numerical AC witnesses. The rich history is prescribed cyclic passive variation, not field data or a deployed WZZT baseline. For flat symmetric history, fixed site 1 identifies 4/6; the selected same-site constant charging identifies the same 5/6 as the selected modulated charging. The failed selected stationary action for `cherry23`, seed 8201 is blind site 1. These counts favor two-port complementarity in this small family; they do not establish superiority of the optimized mobile route over the fixed route, or a statistically established site-selection advantage.

The invalid linear-only ablation excludes the true AC topology in 58 of the 180 observation batches. The AC outer variants have no true-candidate exclusion in this experiment. These are dependent synthetic cases, not estimated field error rates.

Stationary sites and mobile routes are ranked separately; the code does not optimize a stationary-versus-mobile travel/energy tradeoff. Candidate future AC voltages used for action scoring are not screened over the complete uncertainty set. Checking the actually simulated voltage domain does not constitute an operational safety guarantee.

The formal squared-voltage error halfwidth 1e-5 pu^2 corresponds near 1 pu to about 5e-6 pu voltage magnitude (roughly 0.002 V on a 400 V base). It is an unusually tight declared observation bound and has no field calibration here. Error-budget sensitivity and actual P/Q-meter error controls must therefore be reported separately from the formal low-noise result.

## Fixed-action sensitivity controls

`controls_ac_v8.py` uses the three predeclared flat/symmetric/seed-8201 truths and keeps their formal action choices fixed. It does not redesign after enlarging error budgets or adding P/Q noise. The completed controls have 32 distinct observation batches and 96 LPs after identical policy actions are deduplicated. The maximum certificate gap is 3.8268e-16; every true candidate is retained. `audit_controls_ac_v8.py` independently verifies all input hashes, original-data differences, policy mappings, interval directions and counts, saving `results_v8/ac_controls/input_audit.json`.

| Control, three truths each | Selected stationary | Fixed site 1 | Selected mobile | Fixed mobile 1-to-2 |
|---|---:|---:|---:|---:|
| Declared voltage-drop error 1e-4 | 2/3 | 2/3 | 3/3 | 3/3 |
| Declared voltage-drop error 5e-4 | 2/3 | 2/3 | 3/3 | 3/3 |
| Declared voltage-drop error 2e-3 | 0/3 | 0/3 | 0/3 | 0/3 |
| Added P/Q meter errors, each box .001 pu | 2/3 | 2/3 | 3/3 | 3/3 |

The first three rows hold all original observed P/Q/V data and realized tiny voltage noise fixed, changing only the declared voltage-error halfwidth. They isolate conservativeness of the larger admissible error domain; they are not simulations with larger realized voltage noise. The final row adds independently generated bounded P/Q errors of magnitude at most .0008 pu, declares .001 pu coordinate error limits, and retains the original 1e-5 pu^2 voltage-error budget. Each truth case uses the same added P/Q errors across its actions. These controls preserve the stationary blind-port failure and the high-voltage-budget failure; they do not supply a population-level success probability.
