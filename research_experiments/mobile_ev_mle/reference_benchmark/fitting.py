"""Sparse local AC fitting with the existing conditional-output objective.

Only the numerical linear algebra and history-only initialization change.
Residuals and exact derivatives come from ev_rx_fit._RxInputProblem.  The source
is still one P/Q injection at the full-support interface, and lines are shared
over time.  CSR/LSMR avoids the dense trust-region SVD; no full information
spectrum is computed.  These are attained local fits, not global MLE proofs.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.sparse import csr_matrix

from .. import likelihood as lk
from ..ev_rx_fit import _RxInputProblem, _RxMap
from ..ev_fixed_train_fit import OUTPUT_CHANNELS, _compact_bias, _FixedInputProblem
from .derivatives import optimizer_problem_jacobian


class _SourceInputProblem:
    """Uniform wrapper for the unchanged fixed-line source-only residual."""
    def __init__(self, candidate, observations, model, *, initial, fixed_r, fixed_x,
                 r_bounds, x_bounds, ratio_bounds, source_p_bounds, source_q_bounds,
                 voltage_bounds, call_counter):
        self.mapping = _RxMap(candidate.m, r_bounds, x_bounds, ratio_bounds)
        if not self.mapping.contains(fixed_r, fixed_x):
            raise ValueError("fixed R/X must lie in original box and ratio cone")
        self.base = _FixedInputProblem(candidate, observations, model, initial=initial,
            fixed_r=fixed_r, fixed_x=fixed_x, r_bounds=r_bounds, x_bounds=x_bounds,
            source_p_bounds=source_p_bounds, source_q_bounds=source_q_bounds,
            voltage_bounds=voltage_bounds, call_counter=call_counter)

    def __getattr__(self, name):
        return getattr(self.base, name)


def history_linear_initial(candidate, observations, model, *, r_bounds, x_bounds,
                           ratio_bounds, source_p_bounds, source_q_bounds):
    """Use arrived squared-voltage drops and P/Q, never true line parameters.

    The first-order law is root^2-|V|^2 = 2 M diag(R) M' P
    + 2 M diag(X) M' Q + 2(R_full*sP+X_full*sQ).  The observed master minus
    summed terminal readings initializes the source; its loss contribution is
    corrected by the subsequent full-AC optimization.
    """
    t, n, m = observations.t, observations.n, candidate.m
    p = observations.p-lk._array(model.bias_p, (t, n), "bias_p")
    q = observations.q-lk._array(model.bias_q, (t, n), "bias_q")
    v = observations.v-lk._array(model.bias_v, (t, n), "bias_v")
    sigma = lk._array(model.v_sigma, (t, n), "v_sigma")
    if observations.master_p is None or observations.master_q is None:
        raise ValueError("historical master P/Q required for source initialization")
    if np.min(v) <= 0 or np.min(sigma) <= 0:
        raise ValueError("positive bias-corrected voltage and sigma required")
    splo, sphi = lk._bounds(source_p_bounds, (t,), "source_p_bounds")
    sqlo, sqhi = lk._bounds(source_q_bounds, (t,), "source_q_bounds")
    sp = np.clip(observations.master_p-p.sum(axis=1), splo, sphi)
    sq = np.clip(observations.master_q-q.sum(axis=1), sqlo, sqhi)
    mapping = _RxMap(m, r_bounds, x_bounds, ratio_bounds)
    branch_p, branch_q = p@candidate.M, q@candidate.M
    branch_p[:, 0] += sp; branch_q[:, 0] += sq
    design = np.concatenate((2*candidate.M[None, :, :]*branch_p[:, None, :],
                             2*candidate.M[None, :, :]*branch_q[:, None, :]), axis=2)
    target = observations.root_v[:, None]**2-v**2
    # Delta-method scaling is only an initializer weight, not a replacement
    # likelihood or a new assumption used in the final objective.
    scale = np.maximum(2*v*sigma, np.finfo(float).tiny)
    a = (design/scale[:, :, None]).reshape(t*n, 2*m)
    b = (target/scale).ravel()
    lower = np.r_[mapping.rlo, mapping.xlo]
    upper = np.r_[mapping.rhi, mapping.xhi]
    free = upper > lower; estimate = lower.copy(); details = {}
    if np.any(free):
        result = lsq_linear(csr_matrix(a[:, free]), b-a@lower+a[:, free]@lower[free],
            bounds=(lower[free], upper[free]), method="trf", lsq_solver="lsmr",
            tol=1e-7, lsmr_tol=1e-8, max_iter=100)
        estimate[free] = result.x
        details = dict(success=bool(result.success), status=int(result.status),
                       message=str(result.message), cost=float(result.cost),
                       optimality=float(result.optimality), iterations=int(result.nit))
    else:
        details = dict(success=True, status=0, message="all public line coordinates fixed")
    r, x = estimate[:m], estimate[m:]
    # The independent NNLS box need not satisfy the X/R cone. First make R
    # large enough to represent its X, then apply the exact closed-section map.
    r = np.maximum(r, x/mapping.rho_high)
    r, alpha = mapping.inverse(r, x)
    r, x, _, _ = mapping.forward(r, alpha)
    initial = dict(r=r, x=x, source_p=sp, source_q=sq)
    return initial, dict(method="arrived_only_nonnegative_V_squared_PQ_linear_fit",
        linear_coefficient_equals="2*physical_R_and_2*physical_X",
        source_start="observed master minus sum observed bias-corrected terminals",
        source_loss_bias_corrected_by_final_full_AC=True, truth_used=False,
        future_used=False, linear_fit=details)


def _sparse_jacobian(problem, vector):
    jac = csr_matrix(optimizer_problem_jacobian(problem, vector))
    jac.eliminate_zeros()
    return jac


def fit_rx_inputs(candidate, observations, model, *, initial=None, fixed_r=None, fixed_x=None,
                  r_bounds=(1e-6, .1), x_bounds=(1e-6, .1),
                  source_p_bounds=(-.5, .5), source_q_bounds=(-.5, .5),
                  voltage_bounds=(.85, 1.15), multistart=2, max_nfev=250, tol=1e-10,
                  seed=0, call_counter=None, ratio_bounds=(.083/.642, .080/.208)):
    """Signature-compatible local sparse fit, preserving failures and boundaries.

    Fixed-line source fits use the original residual with the same CSR/LSMR path.
    A failed starting AC state is moved toward the *same public domain's* lower
    line corner. For fixed lines only the source start moves toward zero. This
    changes only a starting point, and each trial is logged.
    """
    began = perf_counter()
    line_fixed = fixed_r is not None
    if line_fixed != (fixed_x is not None):
        raise ValueError("fixed_r and fixed_x must be supplied together")
    for value, label in ((multistart, "multistart"), (max_nfev, "max_nfev")):
        if int(value) != value or value < 1:
            raise ValueError(label+" must be a positive integer")
    if not np.isfinite(tol) or tol <= np.finfo(float).eps:
        raise ValueError("tol must exceed machine precision")
    if line_fixed:
        initialization = dict(method="fixed_training_lines_and_arrived_master_minus_sum_terminal_source_start",
                              truth_used=False, future_used=False)
    elif initial is None:
        initial, initialization = history_linear_initial(candidate, observations, model,
            r_bounds=r_bounds, x_bounds=x_bounds, ratio_bounds=ratio_bounds,
            source_p_bounds=source_p_bounds, source_q_bounds=source_q_bounds)
    else:
        initialization = dict(method="caller_supplied_observed_only_warm_start",
                              truth_used=False, future_used=False)
    setup = dict(initial=initial, r_bounds=r_bounds, x_bounds=x_bounds, ratio_bounds=ratio_bounds,
        source_p_bounds=source_p_bounds, source_q_bounds=source_q_bounds,
        voltage_bounds=voltage_bounds, call_counter=call_counter)
    problem = (_SourceInputProblem(candidate, observations, model, fixed_r=fixed_r, fixed_x=fixed_x, **setup)
               if line_fixed else _RxInputProblem(candidate, observations, model, **setup))
    if call_counter is not None:
        call_counter['fit_calls'] = call_counter.get('fit_calls', 0)+1
    rng = np.random.default_rng(seed); starts = [problem.start.copy()]
    for _ in range(1, int(multistart)):
        if line_fixed:
            trial = problem.start+rng.normal(0, .01, problem.width)
            starts.append(np.clip(trial, problem.lower+problem.margin, problem.upper-problem.margin))
            continue
        full = problem.expand(problem.start)
        full[:problem.m] *= np.exp(rng.normal(0, .5, problem.m))
        full[problem.m:problem.shared] += rng.normal(0, .25, problem.m)
        full[problem.shared:] += rng.normal(0, .01, 2*problem.t)
        starts.append(np.clip(full[problem.free], problem.lower+problem.margin,
                              problem.upper-problem.margin))
    restarts = []; retained = []; initial_loss = None; solver_calls = 0
    for index, original_start in enumerate(starts):
        start = original_start.copy()
        record = dict(index=index, requested_initial_state=problem.state_record(start),
                      success=False, initial_ac_trials=[])
        # A public box can deliberately be much broader than an AC-feasible
        # operating region. Do not shrink the final box to repair an initial AC.
        for power in range(13):
            factor = .5**power
            if power:
                if line_fixed:
                    start = np.clip(factor*original_start, problem.lower+problem.margin,
                                    problem.upper-problem.margin)
                else:
                    full = problem.expand(original_start)
                    full[:problem.m] = problem.mapping.lower+factor*(full[:problem.m]-problem.mapping.lower)
                    start = np.clip(full[problem.free], problem.lower+problem.margin,
                                    problem.upper-problem.margin)
            factor_name = 'source_factor' if line_fixed else 'line_factor'
            try:
                residual = problem.evaluate(start)
                if not np.isfinite(residual).all():
                    raise FloatingPointError("nonfinite initial residual")
                record['initial_ac_trials'].append({factor_name:factor, 'ac_success':True})
                break
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                record['initial_ac_trials'].append({factor_name:factor, 'ac_success':False,
                                                       'error':type(error).__name__+": "+str(error)})
        try:
            residual = problem.evaluate(start)
            record['initial_state'] = problem.state_record(start)
            record['initial_loss'] = float(residual@residual)
            if initial_loss is None:
                initial_loss = record['initial_loss']
            if call_counter is not None:
                call_counter['solver_calls'] = call_counter.get('solver_calls', 0)+1
            solver_calls += 1
            result = least_squares(problem.evaluate, start,
                jac=lambda vector: _sparse_jacobian(problem, vector),
                bounds=(problem.lower, problem.upper), x_scale=problem.scale,
                method='trf', tr_solver='lsmr',
                tr_options=dict(atol=1e-10, btol=1e-10, maxiter=max(200, 4*problem.width)),
                ftol=tol, xtol=tol, gtol=tol, max_nfev=int(max_nfev))
            residual = problem.evaluate(result.x); loss = float(residual@residual)
            phy = problem.base.cache['physical']; voltage = abs(phy['voltage'])
            r, x, _ = problem.unpack(result.x)
            voltage_ok = bool(np.min(voltage) >= voltage_bounds[0] and np.max(voltage) <= voltage_bounds[1])
            domain_ok = problem.mapping.contains(r, x)
            valid = bool(result.success and voltage_ok and domain_ok and np.isfinite(loss))
            record.update(loss=loss, success=valid, solver_success=bool(result.success),
                status=int(result.status), message=str(result.message), nfev=int(result.nfev),
                optimality=float(result.optimality), physical_voltage_feasible=voltage_ok,
                ratio_domain_feasible=domain_ok,
                active_bound_count=int(np.count_nonzero(result.active_mask)),
                attained_state=problem.state_record(result.x))
            retained.append((valid, loss, result, record))
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            record.update(loss=None, solver_success=False, status=-99,
                          message=type(error).__name__+": "+str(error), nfev=None)
        restarts.append(record)
    base = problem.base; mapping = problem.mapping
    diagnostics = dict(objective="conditional_known_diagonal_Gaussian_output_given_measured_PQ",
        measurement_channels=list(OUTPUT_CHANNELS), source_node=1,
        source_spatial_location_shared=True, source_joint_voltage_and_master_estimation=True,
        source_anchored=True, root_voltage_known=True, line_parameters_fixed=line_fixed,
        fixed_lines=line_fixed, P_Q_inputs_fixed=True, P_Q_measurement_error_propagated=False,
        P_Q_likelihood_included=False, input_EV_already_in_total_PQ=True,
        source_sign="positive_consumption", impedance_units="pu", known_x_over_r=False,
        bias_policy="known_calibrated", line_domain="original_box_intersect_X_over_R_cone",
        ratio_constraint_enforced=True, ratio_bounds=[mapping.rho_low, mapping.rho_high],
        r_bounds_original=[mapping.rlo.tolist(), mapping.rhi.tolist()],
        x_bounds_original=[mapping.xlo.tolist(), mapping.xhi.tolist()],
        effective_r_bounds=[mapping.lower.tolist(), mapping.upper.tolist()],
        parameterization="fixed_physical_R_X_source_only" if line_fixed else "same_ev_rx_fit_R_alpha_box_cone_map",
        kink_derivative="right derivative in R at max/min switches",
        shared_parameter_count=problem.shared, source_parameter_count=2*problem.t,
        joint_parameter_count=base.width, optimizer_parameter_count=problem.width,
        residual_count=problem.t*(problem.n+2), output_normalizer=base.normalizer,
        initial_loss=initial_loss, initialization=initialization, restarts=restarts,
        solver="scipy.least_squares_trf_lsmr_CSR_exact_AC_Jacobian",
        fit_derivative_rhs="R/X and source P/Q only; source P/Q only when lines fixed",
        terminal_PQ_fit_derivatives_omitted=True,
        solver_calls=solver_calls, total_nfev=sum(row.get('nfev') or 0 for row in restarts),
        ac_evaluations=base.ac_calls,
        information_spectrum_available=False,
        information_spectrum_reason="omitted: dense full-history SVD not required by this local fitter",
        local_identifiability_full_column_rank=None,
        global_optimality_certified=False, global_loss_lower_bound=None,
        loglikelihood_is_topology_probability=False,
        ac_solution_branch="root_initialized_fixed_point; other equilibria not enumerated")
    biases = [_compact_bias(base.bias[key]) for key in ('p', 'q', 'v')]
    if not retained:
        r, x, _ = problem.unpack(problem.start)
        diagnostics.update(failure_kind="computation", seconds=perf_counter()-began)
        return lk.FitResult(candidate.name, r.copy(), x.copy(), base.p.copy(), base.q.copy(),
            np.full(problem.t, np.nan), np.full(problem.t, np.nan), *biases,
            np.inf, -np.inf, False, -99, "all numerical restarts failed", 0, np.inf, {}, diagnostics)
    valid, loss, result, record = min(retained, key=lambda row:(not row[0], row[1]))
    problem.evaluate(result.x); phy = base.cache['physical']
    r, x, source = problem.unpack(result.x)
    finite_losses = [row['loss'] for row in restarts if row.get('loss') is not None]
    diagnostics.update(physical_voltage_feasible=record['physical_voltage_feasible'],
        ratio_constraints=mapping.constraint_record(r, x),
        optimizer_alpha=None if line_fixed else problem.expand(result.x)[problem.m:problem.shared].tolist(),
        min_voltage=float(np.min(abs(phy['voltage']))), max_voltage=float(np.max(abs(phy['voltage']))),
        ac_equation_residual=float(phy['ac_equation_residual']),
        power_balance_residual=float(phy['power_balance_residual']),
        active_bound_count=record['active_bound_count'],
        restart_loss_spread=float(max(finite_losses)-min(finite_losses)),
        failure_kind=None if valid else ('physical_domain' if not record['physical_voltage_feasible']
            or not record['ratio_domain_feasible'] else 'computation'),
        seconds=perf_counter()-began)
    predicted = dict(phy, p=base.p+base.bias['p'], q=base.q+base.bias['q'],
                     v=phy['v']+base.bias['v'], physical_v=phy['v'])
    return lk.FitResult(candidate.name, r.copy(), x.copy(), base.p.copy(), base.q.copy(),
        source[:, 0].copy(), source[:, 1].copy(), *biases,
        loss, -.5*(loss+base.normalizer), bool(valid), int(result.status), str(result.message),
        int(result.nfev), float(result.optimality), predicted, diagnostics)


__all__ = ['fit_rx_inputs', 'history_linear_initial']
