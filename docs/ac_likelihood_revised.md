# Error-aware AC candidate likelihood

## Scope

The AC stage is a finite-candidate physical ranker. It does not create hidden
nodes. Ordered RNJ, one-layer aggregation, and GTLS generate candidate rooted
trees; this stage estimates each fixed tree and selects among them.

## Strict execution order

For candidate tree (T_k), let (A_k) be its terminal-to-edge path incidence.
The reduced matrices are constrained to

\[
R_k=A_k\operatorname{diag}(\rho_k)A_k^\top,\qquad
X_k=A_k\operatorname{diag}(\chi_k)A_k^\top,
\qquad \rho_k,\chi_k\ge 0.
\]

1. Fit separable GTLS on the training scenarios to obtain common initial
   reduced (R/X) matrices.
2. Project the GTLS matrices onto every fixed candidate tree.
3. Refit its branch coefficients by structured-EIV IRLS. At each iteration,
   P/Q/V meter variances are propagated through the current tree matrices and
   used as heteroscedastic regression weights.
4. Run radial AC power flow on representative training operating points. Use
   the difference between nonlinear AC squared-voltage drop and the linear
   tree prediction as one sequential-convex correction. Accept the first full
   or damped update that reduces training AC voltage RMSE.
5. Evaluate the updated tree on held-out representative scenarios. The
   predictive variance contains V meter variance, propagated P/Q variance, and
   a training-estimated model-mismatch floor.
6. Rank by mean Gaussian predictive NLL, a cross-scenario dispersion penalty,
   and a normalized edge-parameter complexity penalty.
7. Apply quartet contradiction and near-zero hidden-edge checks only after AC
   selection.

The validation scenarios are no longer the last entries in the case bank.
Their P/Q operating features are standardized; the scenario nearest the
feature centroid is selected first, followed by farthest-point coverage. The
default validation counts are 1, 2, and 3 for case banks of 3, 5, and 9
scenarios.

## Candidate coverage

AC likelihood cannot recover a topology absent from the finite candidate set.
This separation is deliberate: likelihood ranking and combinatorial topology
generation are different problems. The complete workflow must therefore
report candidate-oracle recall and expand the candidate bank when necessary.
Reasonable expansion mechanisms include perturbation/bootstrap RNJ trees,
quartet-guided local tree edits, and compatible-clade completion. These belong
before AC ranking, not inside the likelihood optimizer.

## Current model boundary

The implementation assumes a balanced radial constant-PQ feeder, a known root,
zero-injection hidden nodes, synchronized P/Q/V, and known relative meter-noise
levels. Root-meter error, interval-average P/Q versus instantaneous V, phase
imbalance, ZIP loads, regulators, and hidden injections require explicit
observation or device models before the score can be interpreted as a physical
likelihood.

## Implementation

- `terminal_case33/pipeline/rnj_candidates.py`: RNJ candidate generation.
- `terminal_case33/pipeline/ac_likelihood.py`: structured-EIV fitting, AC update,
  representative split, and predictive ranking.
- `terminal_case33/estimation/meter_error.py`: shared P/Q/V error propagation.
- `terminal_case33/estimation/nonnegative.py`: nonnegative ridge solver.
- `terminal_case33/pipeline/topology_validation.py`: shared candidate pool and
  post-selection quartet/edge validation.
