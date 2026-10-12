General radial AC evidence interface (v9)
========================================

Scope and entry points
----------------------
ac_general_v9.py is a research evidence layer. It does not change production
WZZT or fix an RNJ proposal as certified structure. Its candidate interface is

    tree = RadialCandidate.from_supports(name, n_terminal, supports)

where supports use zero-based terminal indices. Supports must be laminar:
two sets must be disjoint or one must contain the other. The constructor adds
all singleton sets and the full set, orders by decreasing size and then
lexicographically, and connects each set to its smallest strict superset.
Thus tree.supports is also the per-edge parameter order. Root node is 0;
edge e ends at node e+1. tree.parents[e] is -1 for the first common edge and
otherwise the parent edge index. The unmetered source is at node 1, after the
first common edge, and other internal junctions have zero injection.

tree.M: n_terminal by n_edge terminal-path matrix.
tree.B: n_edge by n_edge physical-bus path matrix.
tree.terminal_edges: indices of buses carrying terminal injections.
tree.edges: physical (parent_node, child_node) tuples.

The interface aggregates unary chains with identical terminal support. It
does not represent arbitrary hidden injections along those chains. This is a
specific modeling domain, not a universal distribution-feeder reconstruction.

Physical model and exact identity
--------------------------------
All quantities are per-unit. Root complex voltage is fixed to 1. The model is
balanced constant-power radial AC with no shunts/regulators. Resistance and
reactance are nonnegative with a known common ratio kappa=x/r. Let theta=2r;
z_e=(1+j*kappa)*theta_e/2 and terminal load S=p+j*q. Positive p/q are loads.
Source injections are allowed to vary within declared bounds at every time.

For one sample, exact AC equations imply

 y = M diag(theta) M' (p+kappa*q)
     + theta_0*s_nominal_eff*1 + w*1 + M*eta,
 |w| <= theta_0*source_error_eff,

 eta_e = (1+kappa^2)/4 *
         [theta_e^2*ell_e + 2*sum_{f strictly below e}(theta_e*theta_f*ell_f)].

Here y_i=1-|V_i|^2 and ell_e is squared branch-current magnitude. A common
source changes nonlinear losses too; it cannot be removed entirely by a
terminal-voltage contrast. ac_forward() checks the complex AC residual and
the exact identity; ac_batch() provides efficient numerical predictions.
These fixed-point solutions are numerical witnesses, not interval proofs of
AC existence/uniqueness. Independent tests solve rectangular nodal KCL using
the physical edge list without the implementation's B or M matrices.

Valid nonlinear outer envelope
------------------------------
Every physical bus satisfies |V|>=vmin. For each edge, KCL gives

 ell_e <= [sum_{terminal j below e} S_abs_cap_j
           + 1_{e=0} S_abs_cap_source]^2 / vmin^2,

 S_abs_cap_j = sqrt((|P_measured_j|+P_error_j)^2
                    +(|Q_measured_j|+Q_error_j)^2).

Only the trunk carries the source's own load. Its influence on downstream
voltages/currents remains covered by vmin. All square roots are rounded up.
On the full declared theta box, square secants and both McCormick upper
planes give valid affine upper bounds on each eta_e. One eta_e is shared by
every descendant terminal; it is not replaced by unrelated output errors.

The path model contains every physical response in the declared domain.
The optional output model further relaxes sums of those same edge envelopes
to independent output remainders. Therefore the exact optimal path slack is
at least the exact optimal output slack, on the same data and parameter box.
Terminal P/Q input errors also enlarge the linear response by

 EIV_i(theta) = sum_{e on path i} theta_e *
               sum_{j below e}(P_error_j+kappa*Q_error_j).

All observations, historical and new, use one shared theta. The source and
loss variables may change each sample. Parameters are not frozen to an old
fit. Local fit failure does not exclude any candidate.

Numerical lower-bound guarantee
-------------------------------
solve_outer(observed, tree, box=theta_boxes, kappa=.5, vmin=.9,
            sharing='path', shared_voltage_gain=None)

Minimizes tau>=0 in an outer linear program. Observation consistency is
|linear_response + shared_loss - Y_measured| <= drop_error + EIV(theta)+tau.
All exact coefficients are rational conversions of the supplied floats.
Coefficient-rounding guards and outward-rounded finite variable bounds make
the floating LP an outer relaxation of that rational LP. Its approximate
dual multipliers are clipped to the correct nonpositive sign. For any such
multipliers lambda, the exact box-corrected weak-duality bound is

 L = lambda' b + sum_k min((c-A'lambda)_k*l_k,
                          (c-A'lambda)_k*u_k).

The returned lower=max(0,L) is rounded downward. A rational primal point is
repaired by clipping parameters/source/loss values and increasing tau, then
verified against every original rational inequality. The returned upper is
the corresponding outer-model slack, rounded upward. This upper is NOT an
AC feasible-solution upper bound. A positive valid lower can exclude that
candidate only in the stated physical/error/parameter domain. Zero lower
means unresolved. Failed solves return lower=None and unresolved status.

linear_only=True is an explicitly invalid AC negative control: eta is forced
to zero. Its positive slack may falsely reject genuine AC observations.
The output marks certificate_valid_for_AC=False for this control.

Observed input dictionary
-------------------------
P, Q, Y_drop: arrays of shape (n_time,n_terminal).
P_error, Q_error: nonnegative scalar or broadcastable arrays; default zero.
drop_error_bound: nonnegative squared-voltage-drop error, broadcastable.
source_nominal_eff: nominal P_source+kappa*Q_source, scalar or per-time.
source_error_eff: bound on its deviation, scalar or per-time.
source_p_abs_cap, source_q_abs_cap: absolute source bounds, per-time or scalar.

P/Q here include the actual EV power measurements. EV commands alone do not
establish the actual injection. Error budgets and parameter domains must be
declared independently of the evaluation truth.

Optional fixed-gain meter model
------------------------------
shared_voltage_gain={
  'gain_bounds':(glo,ghi),
  'root_jitter':delta0,
  'terminal_jitter':delta_or_length_n_array,
  'vmax':1.1
}

This REPLACES independent drop-error constraints; it does not add another
generic tolerance to them. It assumes Vhat_it=g_i*|V_it|+xi_it with one fixed
positive gain g_i in [glo,ghi] per meter across all historical/new samples,
|xi_0t|<=delta0, |xi_it|<=delta_i, and physical terminal |V_it|<=vmax. The
true grid-root voltage remains exactly one. No claim is made that a meter
standard provides this fixed-gain/jitter decomposition.

Let h_i=1/g_i^2 and b_i=1-h_i*g_0^2. Then
  y_it = h_i*Yhat_it+b_i-h_i*dnoise_it,
  h_i in [1/ghi^2,1/glo^2],
  1-h_i*ghi^2 <= b_i <= 1-h_i*glo^2,
  |dnoise_it| <= D_i
      = 2*ghi*delta0+delta0^2 + 2*ghi*vmax*delta_i+delta_i^2.

The LP adds h_i,b_i shared over every time, and imposes
 |response_it-h_i*Yhat_it-b_i| <= h_i*D_i+EIV_i(theta)+tau.
This is linear. The common root-gain relation across different b_i is
outer-relaxed, so the construction preserves domain coverage but may lose
information. Returned h_i/b_i do not certify physical meter calibration.
Drifting gains violate this model and require a separate declared domain.

Local references and operational scope
--------------------------------------
fit_ac_reference(history,tree,box=...,outer_result=optional_existing_outer)
uses two least-squares starts and fits one source P value per time. It fixes
source Q/P=.3 only for the representative, intersects source P/Q/effective
bounds, and verifies the original independent voltage-error box and physical
voltage interval. It does NOT fit fixed-gain meter parameters. A representative
used under that gain model must therefore be labeled a nominal design
reference, not a calibrated AC witness. No local-fit success changes the
outer-LP exclusion semantics.

Neither this module nor its LP constitutes an EV actuation safety filter,
thermal protection check, full-topology-bank coverage proof, calibrated
confidence probability, or optimized real-world travel dispatch.

Validation command
------------------
D:\apps\miniconda3\envs\Topo\python.exe -m pytest -p no:cacheprovider \
 research_experiments\mobile_ev_probing\test_ac_general_v9.py -q

Initial verification: 25 tests passed (6.39 seconds). Tests cover 6/12-terminal
trees, nonbinary geometry, kappa=0/.5/1.2, independent nodal AC, unequal theta
box boundaries/interiors, exact v8 reduction, input-error corners, valid
lower-bound direction, solver failure, local reference verification, fixed
gain/jitter corners and outward squared-voltage error conversion.
