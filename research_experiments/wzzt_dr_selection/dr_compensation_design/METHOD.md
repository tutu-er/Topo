# Historical compensation faces for a bounded DR test

This is a new small heuristic test, not a validated WZZT price method. It does
not maximize a gradient norm or numerical rank. No production default or frozen
earlier experiment is modified.

The information boundary is old observed P/Q/V only. First run the existing
finite-pool WZZT on all 288 training and 288 validation rows to its stopping
condition. Let F be its selected *uncontracted* family. For an actually available
candidate C, form G_C by deleting the crossing members of F and adding C. This
crossing repair belongs to the research implementation, not the original core.

For each G_C, the historical compensation face is defined by:

- every r and x independently in [0,2], including zero;
- training mean L1 at most J_F + 1e-10 + .001 J_F;
- old validation mean L1 at most 1.03 times the minimum over the entire saved
  historical candidate/path point list, plus 1e-12.

All weights are free in these constraints. Take the minimum and maximum of the
new block's effect r_C + .35 x_C over this face, as well as the original F fit.
Infeasible faces and unresolved solves are recorded. No family is discarded
because one arbitrary parameter representative fails the validation condition.
The face is defined by algorithm tolerances, not a physical noise confidence set.

Forecast each future quarter's P by the mean of three old training days. The
declared synthetic planning forecast sets Q=.35P, including the DR increment.
Thus K=R+.35X determines every predicted future, and K-equivalent endpoint
representatives are deduplicated before equally weighting them. Zero/weak
coefficients remain allowed. Finite equal weights are an explicit heuristic,
not calibrated model probabilities or full parameter coverage.

For every possible executed two-event training trajectory, use each endpoint
to predict future voltages. Refit F, all actual native RNJ families, and every
G_C on old plus predicted future training observations. All r/x are reoptimized.
Apply the original candidate ordering and gain >1e-10+.001 J_F rule once. Compare
the accepted point, F and complete native candidates by the original 3% validation
rule and .16 contraction, using historical calibration in planning. Validation
execution trajectories are integrated independently of training trajectories.

The score of a price menu is the exact joint-response expectation of the final
complete block error against the endpoint that generated the predicted data,
averaged across the finite endpoints. Ties first prefer lower expected payment,
then are randomized uniformly using frozen independent planning seeds. This is
a one-step approximation with a frozen bank/calibration, not full future WZZT.
Actual evaluation rebuilds the current observed bank/calibration and runs the
complete original research paths without those planning approximations.

The 16 predeclared menus comprise zero, six repeated-single menus (three nodes,
price1/2), six ordered price2 rotations, and three simultaneous *opposite-shift*
price1 pairs. The method may select single-point, repeated, rotating or zero
actions. Nothing forces diversification. This is not global price optimization.

Budget and execution remain unchanged: four whole events, .2kW/.07kvar shift,
.2kWh reduction and rebound, task terminal .4kWh, power in[0,.4]kW, prices0/1/2,
participation0/.5/.75, worst payment<=1.6. Acceptance precedes an independent
whole-event direction. Planning integrates the independent joint law exactly;
the common-participation law is pure-tested but not claimed tested in this batch.

The locked actual comparison is two independent behavior/measurement streams,
seven true models (one zero, all three weak, all three strong pairs), and five
policies (guided, zero with equal additional observations, random repeated
single2, random rotation2, random simultaneous opposed pair1):70 mappings.
Three methods see each same observation: current RNJ, current WZZT, and RNJ with
optional root. Every truth is used only after all prices are frozen. The natural
trajectory is shared and retained; 70 mappings are not 70 independent samples.

Pass/fail is based on final complete errors and payments, especially against
rotation, not the planning score, intermediate accepts, gradients or rank.
Failures, incomplete scans and zero/weak blocks remain in denominators. If the
rule selects zero, reproduces exploration, or fails to improve final output, it
is not rescued by changing bounds, noise, signal amplitude, seeds or budget.
Total new LP calls<=3000 and wall time<=900 seconds across both stages.

Run `study.py prepare --root <new-private-root>` before `study.py run --root ...`.
The controller freezes source/history hashes, then all actual menus and streams
before creating actual futures. Household and observed trajectories remain
private. Result/source hashes and summaries may be published separately.
