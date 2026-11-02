# Continuous DR with the original frozen-candidate rejection event

This implementation keeps all 288 training and 288 validation rows of the
existing paper15/b2/voltage-noise .0005 case. It reuses the two historically
selected coarse/refined RNJ+WZZT candidates, all their R/X coefficients, and
their .16 output maps. No topology fit, candidate search, or new AC simulation
is performed by this pilot.

## Model and information boundary

There are two existing controllable resources (labels 103, 101), 96 quarter-hour
slots, and 192 continuous signed mean adjustments u. The existing component
baseline 1 kW and maximum 2 kW give mean bounds [-1,1] kW. The reward and
direction are abs(u)/.5 and sign(u). Payment is a prepaid bidirectional mean-volume
contract: C=.25 sum(u^2/.5). There is no aggregate budget cap or energy equality.

Actual response is U=u*(1+cv Z), cv=sqrt(.025/3), with independent resource-slot
standard Gaussians in this case. The generic evaluator also accepts a fixed
correlated multiplier factor. This signed form represents correlated execution
multipliers, not constant correlations between signed kW errors. The independent
marginals agree with the former abs(u)*Z model. Power Q changes by .35U.

The reference R0/X0 is the historical raw ordered response regression already
stored in the same baseline fit. It uses all training data, symmetry,
nonnegativity and diagonal order, but no common tree decomposition. It is not
the refined candidate and not known truth. For each training day,
e0=Y-P R0^T-Q X0^T. The joint [P,Q,e0] mean and three-day sample covariance define
a Gaussian working prediction law, rank at most two. Validation is used only
for historical candidate losses. This is a joint law of observed inputs and
prediction residuals; separate latent true-power and sensor-error distributions
are not estimated. No second independent measurement noise is added.

For sampled new-day inputs, Y=(P+dP)R0^T+(Q+dQ)X0^T+e0. Both candidates receive
the same P/Q/Y. Their residual changes are [(R0-Rk)+.35(X0-Xk)]U/base_kw.
The zero-DR P/Q/Y distribution is unchanged from the old joint [P,Q,eF]
construction: only its linear coordinates have changed.

## Exact selection event

Hk is historical validation MAE and ell_k is new-day MAE. The explicit main
rule is Lk=.6 Hk+.4 ell_k (new-day weight 2), and eligibility is
Lk <= 1.03 min_j Lj + 1e-12. The old scorer's default `primary_rule` is 2%
for the new day and is deliberately NOT used; select w2/tau3 from its `rules`.

In micro p.u.^2 units, D=ell_C-1.03 ell_F and
h=1.5(1.03 H_F-H_C)+2.5e-6. Both candidates must pass the historical gate.
Then D>h is exactly coarse exclusion for this two-candidate pool. For larger
pools it is a sufficient pairwise event, not the entire selection event.
The optimization is min C(u) subject to P(D>h)>=.95 and original mean bounds.
No fixed-sign substitute, 120% target or per-residual risk penalty is used.

## Conditional integration and gradients

Fix a history-only unit direction in the baseline Gaussian factors; write
zbase=v z+V eta. Condition on eta and all execution multipliers. Each residual
is a_i(u)+b_i z. Sorting -a_i/b_i gives at most 2880 knots. The strict event is
linear on each interval. Its Gaussian mass is integrated using CDF differences.
Zero slopes, zero weights, repeated knots, negative L1 weights and threshold
platforms are retained. The inner integration is exact up to floating precision.

At transverse threshold roots t, the coefficient gradient uses
phi(t)/abs(D_z(t)) times the actual residual signs at t. The response derivative
includes its random multiplier 1+cv Z. Conditional gradients are chained to all
192 controls. A finite fixed-sample objective is differentiated and checked by
independent directional finite differences. Interchanging a population
derivative and expectation additionally requires regularity; numerical checks
do not establish that theorem. An ineffective integration factor is a genuine
failure case, not evidence for a zero population gradient.

SLSQP solves this nonconvex finite integration approximation from two predefined
starts with iteration/time limits. Returned feasible points give candidate
costs, not global minimum certificates. Planning uses 128 outer draws, a separate
512-draw probability estimate chooses among returned policies, and 2000 fresh
whole-day paths are opened only after price freeze. All policies share those
2000 paths for paired comparisons. Mean bounds do not bound Gaussian realizations;
all excursions and negative baseline loads are counted without clipping/redraw.

## Outputs and scope

Run the existing frozen scorer on every new observation. Save original3% exits,
raw/final support selections, scalar margins, cost, physical excursions, failures
and solver stops. Read same-case truth only after all selections are frozen.
One-sided Clopper-Pearson bounds refer to the specified Gaussian event, not
topology truth or a real-world distribution. Report a simultaneous bound across
reported policies as well. Model-based improvement and actual physical feasibility
are separate outputs. No independent AC or field improvement is claimed here.
