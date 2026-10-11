v8 passive-first controller (research prototype)
================================================

Question
--------
Can passive observations eliminate the need for EV, and can a history-only
site selector obtain additional evidence when passive history is ambiguous?
This is an operational experiment, not a new identification theorem or a
production WZZT integration. AC validation is a separate study.

What changed from v7
--------------------
1. Every candidate first receives the full shared-source historical LP fit.
   Only a verified globally valid strictly positive minimum-slack lower bound
   excludes it. A solver failure, absent witness, or zero lower bound retains it.
2. No retained candidate: report inconsistency with the declared family/model.
   One exactly feasible retained candidate: stop and deploy no EV. A very small
   positive primal slack is never rounded to zero: an exactly feasible point
   of the equivalent historical polytope supplies a separate witness if needed.
3. Two or more retained candidates: refit one historical representative per
   candidate; compute local and uniformly weighted committee T scores only over
   the retained bank; choose the committee endpoint action. Failed numerical
   design retains every candidate and reports unresolved design.
4. Refit the original, unprojected outputs jointly across history and new data.
   Compare matched no-EV, same-site constant and endpoint charging. Rich history
   cases use no EV, so forced-charge counterfactuals are not run for those cases.

Conditional guarantee
----------------------
Let F_h(D) be the set of candidate-h parameter/source explanations that satisfy
all observation bounds with zero extra slack. If truth h* belongs to the finite
candidate bank and its actual parameters/source/errors lie inside the declared
domains, then F_h*(D) is nonempty. A globally valid lower bound L_h(D)>0 on the
minimum nonnegative slack proves F_h(D) is empty, so valid exclusion cannot
remove h*. Adding observations without changing the model or bounds gives
F_h(D_old union D_new) subset F_h(D_old) after projection on common parameters;
in particular the minimum additional slack cannot decrease. Thus an excluded
candidate cannot return solely because new data were added. Domain expansion
or model revision requires screening the original bank again: old exclusions
do not automatically remain valid under different assumptions.

A singleton is conditional identification inside that bank/model, not a global
topology certificate, calibrated coverage, or topology probability. A singleton
without an exact feasible witness is explicitly unresolved. The design score
does not itself exclude anything or guarantee worthwhile EV deployment.

Frozen experiment
-----------------
3 true cherries x 3 coefficient profiles (symmetric, asymmetric, weak internal
branch) x 3 history patterns x 2 seeds (8101,8102) = 54 matched cases.
Same v7 coefficient box and linear P-only model; known common source response,
source budget .1 pu and actual uniform deviations +/-.02 pu; squared-drop
error bound 1e-5 with independent generated errors inside +/-.8 of that bound.
Source and noise draws are exactly paired across future policies and retained
separately from the observation-only selector input.

The inherited epsilon=1e-5 squared-voltage-pu bound is a demanding measurement
assumption: near 1 pu voltage it corresponds approximately to 5e-6 pu voltage
error (about 1.15 mV on an illustrative 230 V base). No field meter/tracking
calibration establishes this bound here. The earlier v7 epsilon=.002 control
remained unresolved. These synthetic rates are not deployment accuracy rates.

12 historical samples: flat P=.2 pu, node-1-only +.02 pu variation, or cyclic
+.02 pu variation across all three terminals. These are stylized measured load
patterns, not measured feeder traces and not evidence of real-world frequencies.
Future: 24 x 15 minutes, background .2 pu, selected node with either no EV,
constant 1.5 kW, or endpoint 0/3 kW. Both charging policies use 9 kWh. A 50 kWh
vehicle with .95 charge efficiency moves from SOC .4 to .571. Travel, connection
hardware, reactive power, AC equations and PQ metering errors are not modeled.
The no-EV future background is constant; comparison with additional naturally
varying passive measurements would require a measured or declared load process.

Interpretation
--------------
Historical informativeness is a property of measurements and assumptions,
not an unconditional capability of the algorithm. Rich passive data can avoid
EV entirely. Weak/irrelevant passive variation leaves ambiguity. An extremely
short internal branch can remain indistinguishable even after chosen EV input.
The committee avoids relying on a single selected reference, but its weights
are design weights and its local T score is not a worst-case performance bound.
Endpoint ordering has no intrinsic time-selection meaning in the static model.

An exact negative mechanism for the weak profile
-----------------------------------------------
Here b=1e-6 and the same (a,b,c) belongs to every candidate's declared box.
For any two cherries h,g, with p=.2*1+u*e_j and 0<=u<=.3,
max_i |[(R_h-R_g)p]_i| <= b*(.4+.3) = 7e-7.
The common source and diagonal terms cancel. The historical variation has a
smaller bound. If actual observation error is at most .8*epsilon (up to the
explicitly checked floating-point rounding), every rival with that same theta
and same source fits within .8e-5+7e-7 < epsilon=1e-5. Therefore no admissible
single-EV waveform can exclude these rivals for this generated weak mechanism.
This is stronger than observing failure of one site selector; it is not a
universal claim that all low-scoring probes are useless, nor a theorem about AC.

Reproduction (fresh output directory required)
---------------------------------------------
& D:\apps\miniconda3\envs\Topo\python.exe -m pytest research_experiments\mobile_ev_probing\test_passive_first_v8.py -q -p no:cacheprovider
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\passive_first_v8.py --output research_experiments\mobile_ev_probing\results_v8\new_passive
& D:\apps\miniconda3\envs\Topo\python.exe research_experiments\mobile_ev_probing\audit_passive_first_v8.py research_experiments\mobile_ev_probing\results_v8\new_passive

Eight tests cover informative-history stop, missing/invalid solver evidence,
unverified singleton, inconsistent data, reduced-bank design, exact action
pairing, weak-edge failure and its all-input-vertices bound. The artifact audit
independently reconstructs
every truth response with exact rational arithmetic, verifies error/source/
parameter coverage, input hashes, policy pairing, outward bound intervals,
the slack cap, retained design bank and monotonic historical exclusions.
It also independently re-solves 27 saved-input LPs covering all history/profile
combinations. All 2,598 recorded audit checks pass for the frozen run.

Frozen results (54 cases, descriptive counts only)
-------------------------------------------------
History pattern   Cases   Passive singleton   EV deployments   Final singleton
Flat                 18                   0               18                12
Node 1 variation     18                   8               10                12
Three-node variation 18                  12                6                12
Total                54                  20               34                36

All actual truths remain retained. The 18 unresolved cases are exactly the
weak-branch cases. No-EV future observations resolve only the original 20
singletons. Among 34 probing cases, both selected-site constant and endpoint
charging resolve the same 16 additional cases. Endpoint charging increases
mean certified nearest-wrong margin from 0.0002517788877 to 0.0004992455811
(including the 18 zero-margin failures); it does not increase success count.
The local midpoint reference is wrong in 14/34 probing cases, another reason
not to interpret representative selection as a confidence statement.

Example: cherry23, symmetric parameters, node-1-only history, seed8101 remains
ambiguous before probing. Committee chooses site3. No EV excludes no rival;
constant charging excludes both with nearest-wrong lower .0004115137241;
endpoint charging excludes both with lower .0007887752665. This is a concrete
case where the EV site complements passive information from another terminal.

Deployment is 306 kWh across this synthetic grid versus 486 kWh if every case
received 9 kWh, saving 180 kWh by passive stopping. However, 162 kWh is still
spent in the 18 weak cases without resolution. This negative is not removed:
the controller does not yet predict an EV's value reliably or stop early within
the six-hour action. No comparison with real fleet cost or field incidence is
implied.

528 LP certificates: maximum gap 3.94054e-15; maximum verified upper bound
.00142274394, below the implementation tau cap 1. 612 design QPs: maximum gap
8.76667e-10. 102 normalized-reference QPs: maximum gap 1.01546e-6. 20 extra
exact feasible-history witnesses justify singleton stopping. Selection takes
83.07 s total, maximum 2.43 s per case on this machine; final evaluation and
artifact writing are outside that selection timing.

Frozen source SHA-256:
317250ea0c56a2c7147cb3754aeceaaa39bed44e796a1a2ddc8412de5a8d8cf2
Saved results: results_v8/passive/passive_results.json, passive_summary.json,
passive_frozen_protocol.json, passive_artifact_audit.json and paired raw inputs.
