"""Independent two-variable full-AC source fits at fixed physical lines.

This is a numerical alternative for the existing conditional-output objective:
terminal P/Q readings are fixed inputs and a source P/Q pair is fitted at the
same physical interface in each period.  No temporal prior or smoothing is
introduced.  Each period and restart has its own ``max_nfev`` budget; this is
explicitly different from one budget for a batched 2*T-variable TRF solve.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import least_squares

from .. import likelihood as lk
from ..ev_fixed_train_fit import _FixedInputProblem, _compact_bias, OUTPUT_CHANNELS
from ..ev_rx_fit import _RxMap
from ..reference_benchmark.derivatives import fit_response_derivatives


class _SourceRow:
    """One period, with exact implicit AC derivatives and a last-state cache."""

    def __init__(self, base, time, counter=None):
        self.base, self.time, self.counter = base, int(time), counter
        self.cache = {}
        self.ac_calls = 0
        self.ac_seconds = 0.
        self.derivative_seconds = 0.

    def evaluate(self, source):
        source = np.asarray(source, float)
        if source.shape != (2,):
            raise ValueError("a period has exactly two source coordinates")
        if "source" in self.cache and np.array_equal(source, self.cache["source"]):
            return self.cache["residual"]
        time, base = self.time, self.base
        self.ac_calls += 1
        if self.counter is not None:
            self.counter["ac_calls"] = self.counter.get("ac_calls", 0) + 1
            self.counter["ac_time_states"] = self.counter.get("ac_time_states", 0) + 1
        began = perf_counter()
        try:
            physical = lk.ac_predict(base.candidate, base.fixed_r, base.fixed_x,
                base.p[time:time+1], base.q[time:time+1],
                source_p=source[:1], source_q=source[1:],
                root_v=base.observations.root_v[time:time+1])
        finally:
            self.ac_seconds += perf_counter() - began
        residual = np.r_[
            (physical["v"][0] + base.bias["v"][time] - base.observations.v[time]) /
            base.sigmas["v"][time],
            (physical["master_p"][0] - base.observations.master_p[time]) /
            base.sigmas["master_p"][time],
            (physical["master_q"][0] - base.observations.master_q[time]) /
            base.sigmas["master_q"][time]]
        if not np.isfinite(residual).all():
            raise FloatingPointError("nonfinite standardized source residual")
        self.cache = dict(source=source.copy(), residual=residual, physical=physical)
        return residual

    def jacobian(self, source):
        self.evaluate(source)
        began = perf_counter()
        try:
            derivatives = fit_response_derivatives(self.base.candidate,
                self.cache["physical"], include_lines=False)
            time, base = self.time, self.base
            return np.vstack((
                derivatives["v"][0] / base.sigmas["v"][time, :, None],
                derivatives["master_p"][0] / base.sigmas["master_p"][time],
                derivatives["master_q"][0] / base.sigmas["master_q"][time]))
        finally:
            self.derivative_seconds += perf_counter() - began


def _valid_physical(physical, voltage_bounds):
    voltage = np.abs(physical["voltage"])
    return bool(np.min(voltage) >= voltage_bounds[0] and
                np.max(voltage) <= voltage_bounds[1])


def _gradient_mapping(source, gradient, lower, upper, scale):
    """Box gradient mappings; their zero set is the box KKT condition."""
    physical = source - np.clip(source-gradient, lower, upper)
    scaled_source = (source-lower)/scale
    scaled_gradient = scale*gradient
    scaled = scaled_source - np.clip(scaled_source-scaled_gradient, 0.,
                                     (upper-lower)/scale)
    return physical, scaled


def fit_fixed_sources(candidate, observations, model, *, fixed_r, fixed_x,
                      initial=None, r_bounds, x_bounds, ratio_bounds,
                      source_p_bounds, source_q_bounds, voltage_bounds,
                      multistart=1, max_nfev=250, tol=1e-8, seed=0,
                      call_counter=None):
    """Fit each source P/Q pair independently, returning the usual FitResult.

    ``success`` requires a converged, physical retained solution in *every*
    period, followed by one full-history AC replay.  An unsuccessful period
    retains its finite attained state/loss when available.  No global MLE or
    global minimum of any two-dimensional nonlinear fit is asserted.

    FitResult.nfev is the sum for selected per-period restarts.  Diagnostics
    additionally preserve the maximum, all-restart totals and all row statuses.
    It must not be compared as if it were a batched residual evaluation count.
    """
    began = perf_counter()
    for value, name in ((multistart, "multistart"), (max_nfev, "max_nfev")):
        if int(value) != value or value < 1:
            raise ValueError(name + " must be a positive integer")
    if not np.isfinite(tol) or tol <= np.finfo(float).eps:
        raise ValueError("tol must exceed machine precision")
    mapping = _RxMap(candidate.m, r_bounds, x_bounds, ratio_bounds)
    if not mapping.contains(np.asarray(fixed_r, float), np.asarray(fixed_x, float)):
        raise ValueError("fixed R/X must lie in original box and ratio cone")
    base = _FixedInputProblem(candidate, observations, model, initial=initial,
        fixed_r=fixed_r, fixed_x=fixed_x, r_bounds=r_bounds, x_bounds=x_bounds,
        source_p_bounds=source_p_bounds, source_q_bounds=source_q_bounds,
        voltage_bounds=voltage_bounds, call_counter=call_counter)
    if call_counter is not None:
        call_counter["fit_calls"] = call_counter.get("fit_calls", 0) + 1
    setup_seconds = perf_counter()-began
    starts = [base.start.reshape(base.t, 2).copy()]
    rng = np.random.default_rng(seed)
    lower, upper = base.lower.reshape(base.t, 2), base.upper.reshape(base.t, 2)
    margin, scale = base.margin.reshape(base.t, 2), base.scale.reshape(base.t, 2)
    for _ in range(1, int(multistart)):
        trial = starts[0] + rng.normal(0., .01, starts[0].shape)
        starts.append(np.clip(trial, lower+margin, upper-margin))

    rows, selected_states, selected_optimalities = [], [], []
    selected_nfev, all_nfev, solver_calls, total_ac_calls = [], 0, 0, 0
    gradients = np.full((base.t, 2), np.nan)
    pg = np.full_like(gradients, np.nan); scaled_pg = np.full_like(gradients, np.nan)
    source_ac_seconds = derivative_seconds = solver_seconds = 0.
    failures = []
    for time in range(base.t):
        row = _SourceRow(base, time, call_counter)
        restarts, attained = [], []
        for index, initial_sources in enumerate(starts):
            start = initial_sources[time].copy()
            item = dict(index=index, success=False, solver_success=False,
                requested_initial_source=start.tolist(), initial_ac_trials=[])
            initial_attainment = None
            try:
                original_start = start.copy()
                for power in range(13):
                    factor = .5**power
                    if power:
                        start = np.clip(factor*original_start,
                                        lower[time]+margin[time], upper[time]-margin[time])
                    try:
                        residual = row.evaluate(start)
                        item["initial_ac_trials"].append(dict(source_factor=factor, ac_success=True))
                        break
                    except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                        item["initial_ac_trials"].append(dict(source_factor=factor,
                            ac_success=False, error=type(error).__name__+": "+str(error)))
                else:
                    raise RuntimeError("all same-domain source initializations failed AC")
                item.update(initial_source=start.tolist(), initial_loss=float(residual@residual))
                initial_attainment = (start.copy(), float(residual@residual), row.cache["physical"])
                solver_calls += 1
                if call_counter is not None:
                    call_counter["solver_calls"] = call_counter.get("solver_calls", 0) + 1
                solver_began = perf_counter()
                try:
                    result = least_squares(row.evaluate, start, jac=row.jacobian,
                        bounds=(lower[time], upper[time]), x_scale=scale[time],
                        method="trf", tr_solver="exact", ftol=tol, xtol=tol, gtol=tol,
                        max_nfev=int(max_nfev))
                finally:
                    solver_seconds += perf_counter()-solver_began
                all_nfev += int(result.nfev)
                residual = row.evaluate(result.x)
                physical = row.cache["physical"]
                loss = float(residual@residual)
                source_ok = bool(np.all(result.x >= lower[time]) and np.all(result.x <= upper[time]))
                physical_ok = _valid_physical(physical, voltage_bounds)
                valid = bool(result.success and physical_ok and source_ok and np.isfinite(loss))
                item.update(loss=loss, success=valid, solver_success=bool(result.success),
                    status=int(result.status), message=str(result.message), nfev=int(result.nfev),
                    optimality=float(result.optimality), source_box_feasible=source_ok,
                    physical_voltage_feasible=physical_ok,
                    active_bound_count=int(np.count_nonzero(result.active_mask)),
                    attained_source=result.x.tolist())
                attained.append(dict(success=valid, loss=loss, source=result.x.copy(),
                    physical=physical, index=index, nfev=int(result.nfev),
                    optimality=float(result.optimality), status=int(result.status)))
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                item.update(status=-99, message=type(error).__name__+": "+str(error), nfev=None)
                # Preserve a legal attained state even if a later trial failed.
                witness = None
                if row.cache:
                    current = row.cache["source"]
                    if np.all(current >= lower[time]) and np.all(current <= upper[time]):
                        witness = (current.copy(), float(row.cache["residual"]@row.cache["residual"]),
                                   row.cache["physical"])
                if witness is None:
                    witness = initial_attainment
                if witness is not None:
                    source, loss, physical = witness
                    item.update(loss=loss, attained_source=source.tolist(),
                                physical_voltage_feasible=_valid_physical(physical, voltage_bounds),
                                source_box_feasible=True, attainment_origin="failed_solver_feasible_cache")
                    attained.append(dict(success=False, loss=loss, source=source,
                        physical=physical, index=index, nfev=0, optimality=np.inf, status=-99))
            restarts.append(item)
        if attained:
            chosen = min(attained, key=lambda item: (not item["success"], item["loss"]))
            source = chosen["source"]
            selected_states.append(source)
            selected_optimalities.append(chosen["optimality"])
            selected_nfev.append(chosen["nfev"])
            try:
                residual = row.evaluate(source)
                gradients[time] = row.jacobian(source).T@residual
                pg[time], scaled_pg[time] = _gradient_mapping(source, gradients[time],
                    lower[time], upper[time], scale[time])
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                chosen["success"] = False
                restarts[chosen["index"]]["final_gradient_error"] = type(error).__name__+": "+str(error)
            row_record = dict(time=time, success=bool(chosen["success"]),
                selected_restart=int(chosen["index"]), loss=chosen["loss"],
                source=source.tolist(), restarts=restarts,
                selected_nfev=int(chosen["nfev"]), selected_status=int(chosen["status"]),
                optimizer_optimality=float(chosen["optimality"]))
        else:
            selected_states.append(np.full(2, np.nan))
            selected_optimalities.append(np.inf); selected_nfev.append(0)
            row_record = dict(time=time, success=False, selected_restart=None,
                              loss=None, source=[None, None], restarts=restarts,
                              selected_nfev=0, selected_status=-99,
                              optimizer_optimality=np.inf)
        if not row_record["success"]:
            failures.append(time)
        row_record.update(ac_calls=row.ac_calls, ac_seconds=row.ac_seconds,
                          derivative_seconds=row.derivative_seconds)
        total_ac_calls += row.ac_calls
        source_ac_seconds += row.ac_seconds; derivative_seconds += row.derivative_seconds
        rows.append(row_record)

    source = np.asarray(selected_states)
    full_replay = dict(attempted=False, passed=False)
    replay_seconds = 0.; physical = None; loss = np.inf
    if np.isfinite(source).all():
        full_replay["attempted"] = True
        replay_began = perf_counter()
        try:
            if call_counter is not None:
                call_counter["ac_time_states"] = call_counter.get("ac_time_states", 0)+base.t
            residual = base.evaluate(source.ravel())
            total_ac_calls += base.ac_calls
            physical = base.cache["physical"]
            loss = float(residual@residual)
            source_feasible = bool(np.all(source >= lower) and np.all(source <= upper))
            voltage_feasible = _valid_physical(physical, voltage_bounds)
            full_replay.update(passed=bool(source_feasible and voltage_feasible and np.isfinite(loss)),
                source_box_feasible=source_feasible, physical_voltage_feasible=voltage_feasible,
                ac_equation_residual=float(physical["ac_equation_residual"]),
                power_balance_residual=float(physical["power_balance_residual"]))
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            full_replay["error"] = type(error).__name__+": "+str(error)
            total_ac_calls += base.ac_calls
        replay_seconds = perf_counter()-replay_began
    success = bool(not failures and full_replay["passed"])
    if success:
        status, message = 1, "all independent source periods converged and full AC replay passed"
    elif any(row["selected_status"] == 0 for row in rows):
        status, message = 0, "one or more independent source periods exhausted their per-period budget"
    else:
        status, message = -99, "one or more independent source periods or full AC replay failed"
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mapping.constraint_record(base.fixed_r, base.fixed_x)
    diagnostics = dict(
        objective="conditional_known_diagonal_Gaussian_output_given_measured_PQ",
        measurement_channels=list(OUTPUT_CHANNELS), source_node=1,
        source_spatial_location_shared=True, source_joint_voltage_and_master_estimation=True,
        source_anchored=True, root_voltage_known=True, line_parameters_fixed=True, fixed_lines=True,
        P_Q_inputs_fixed=True, P_Q_measurement_error_propagated=False,
        P_Q_likelihood_included=False, input_EV_already_in_total_PQ=True,
        source_sign="positive_consumption", impedance_units="pu", bias_policy="known_calibrated",
        line_domain="original_box_intersect_X_over_R_cone", ratio_constraint_enforced=True,
        ratio_bounds=[mapping.rho_low, mapping.rho_high], ratio_constraints=ratio,
        parameterization="fixed_physical_R_X_independent_source_pairs",
        shared_parameter_count=0, source_parameter_count=2*base.t,
        joint_parameter_count=2*base.t, optimizer_parameter_count_per_period=2,
        residual_count=base.t*(base.n+2), output_normalizer=base.normalizer,
        solver="scipy.least_squares_trf_exact_independent_source_2d",
        fit_derivative_rhs="source P/Q only; exact full-AC implicit derivatives",
        terminal_PQ_fit_derivatives_omitted=True,
        source_periods_independent=True, temporal_source_prior=False,
        max_nfev_per_period_per_restart=int(max_nfev), multistart_per_period=int(multistart),
        budget_scope="independent max_nfev per period per restart, not one batched budget",
        legacy_global_budget_equivalent=False,
        status_scope="aggregate independent-period outcome; per-period SciPy statuses are retained",
        nfev_scope="sum of selected per-period restarts; not batched full-history calls",
        nfev_selected_sum=int(sum(selected_nfev)),
        nfev_selected_max=int(max(selected_nfev, default=0)), total_nfev=int(all_nfev),
        solver_calls=solver_calls, ac_evaluations=total_ac_calls,
        ac_time_states=int(sum(row["ac_calls"] for row in rows)+base.t*full_replay["attempted"]),
        failed_periods=failures, failed_period_count=len(failures), periods=rows,
        full_history_replay=full_replay,
        gradient=gradients.tolist(), gradient_inf=float(np.max(abs(gradients))),
        projected_gradient_inf=float(np.max(abs(pg))),
        scaled_projected_gradient_inf=float(np.max(abs(scaled_pg))),
        optimizer_optimality_max=float(max(selected_optimalities, default=np.inf)),
        information_spectrum_available=False, local_identifiability_full_column_rank=None,
        global_optimality_certified=False, global_loss_lower_bound=None,
        loglikelihood_is_topology_probability=False,
        ac_solution_branch="root_initialized_fixed_point; other equilibria not enumerated",
        timings=dict(setup_seconds=setup_seconds, solver_inclusive_seconds=solver_seconds,
            row_AC_seconds=source_ac_seconds, row_derivative_seconds=derivative_seconds,
            full_replay_seconds=replay_seconds),
        failure_kind=None if success else "retained_period_or_replay_failure",
        seconds=perf_counter()-began)
    if physical is None:
        predicted = dict(v=np.full_like(base.p, np.nan), p=observations.p.copy(),
                         q=observations.q.copy())
        diagnostics.update(physical_voltage_feasible=False)
    else:
        predicted = dict(physical, p=base.p+base.bias["p"], q=base.q+base.bias["q"],
                         v=physical["v"]+base.bias["v"], physical_v=physical["v"])
        diagnostics.update(physical_voltage_feasible=full_replay.get("physical_voltage_feasible", False),
            min_voltage=float(np.min(abs(physical["voltage"]))),
            max_voltage=float(np.max(abs(physical["voltage"]))),
            ac_equation_residual=float(physical["ac_equation_residual"]),
            power_balance_residual=float(physical["power_balance_residual"]))
    biases = [_compact_bias(base.bias[key]) for key in ("p", "q", "v")]
    return lk.FitResult(candidate.name, base.fixed_r.copy(), base.fixed_x.copy(),
        base.p.copy(), base.q.copy(), source[:, 0].copy(), source[:, 1].copy(), *biases,
        loss, -.5*(loss+base.normalizer), success, status, message,
        int(sum(selected_nfev)), float(max(selected_optimalities, default=np.inf)),
        predicted, diagnostics)


__all__ = ["fit_fixed_sources"]
