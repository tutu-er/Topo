# Four saved rejection events: local optimizer stability

This is a post-hoc diagnostic of the saved evaluation, not a new policy or calibration. The four conditions are selected only by the already saved individual-policy false-rejection flags: (989201, none), (989205, charge_fixed), (989210, none), (989211, none). All four candidate topologies are fitted in each condition. No observations are generated and no EV plan is changed.

For candidate \(k\), let \(J_k(x)=\|\rho_k(x)\|_2^2\) be its complete standardized H+future residual sum of squares. The physical coordinates \(x\) contain shared line R/X, per-time terminal P/Q and per-time common-source P/Q at interface 1. Biases, root voltages, meters, observations and parameter boxes remain as saved. Let \(x_k^0\) be the original saved joint state. The expanded search returns feasible states \(x_k^1,\ldots,x_k^m\), and the retained attained value is

\[
\widehat J_k=\min\{J_k(x_k^0),J_k(x_k^1),\ldots,J_k(x_k^m)\}
\le J_k(x_k^0).
\]

The inequality follows by retaining the original feasible state. Every included state is replayed using a separate backward-current/forward-voltage calculation, its complete Gaussian residuals, and the saved R/X box. A feasible state provides an objective upper bound even if the software stopped before convergence; its solver status is retained. No global lower bound is available.

The first new start uses the entire original joint physical state. The fitter moves values at exact box boundaries to its usual small interior margin, so the original state is also kept separately. Seven additional starts apply the frozen fitter's seeded lognormal perturbation to R/X and retain the same joint state coordinates. The seed is the saved policy seed + 802 + candidate-bank index, as in the original noisy joint-fit rule. The expanded budget is eight starts and max_nfev=500 per start, tol=1e-10, exact linear algebra. Original fits used two starts and max_nfev=250.

For the saved true candidate \(a\), the diagnostic reports the original and attained relative costs

\[
g_a^0=J_a(x_a^0)-\min_k J_k(x_k^0),\qquad
\widehat g_a=\widehat J_a-\min_k\widehat J_k.
\]

Truth is used only for this final offline comparison, never for starts, solver inputs or action selection. Candidate costs can improve by different amounts, so the gap need not decrease. Small changes in all candidate SSE values support local search stability for these four observed conditions. They do not establish global MLEs or uniform optimizer accuracy.

The original thresholds, decisions and raw results remain unchanged. The saved original threshold is included only as scale context for the diagnosed event; the new attained fits are not substituted into the old calibrated decision procedure. The four selected rejection events are not a fresh random evaluation sample.
