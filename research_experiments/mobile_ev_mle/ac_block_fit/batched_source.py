"""Vectorized, period-independent source LM with exact full-AC responses.

Only fixed-line P/Q source pairs are optimized.  AC, derivative and 2x2 linear
algebra calls are batched, while damping, acceptance, stopping and budgets are
independent for every period.  Completed periods are frozen.  The nine active
sets below solve a two-variable *convex step*; they are unrelated to EV actions.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .. import likelihood as lk
from ..ev_fixed_train_fit import _FixedInputProblem, _compact_bias, OUTPUT_CHANNELS
from ..ev_rx_fit import _RxMap
from ..reference_benchmark.derivatives import fit_response_derivatives
from .fixed_source import _gradient_mapping


def solve_source_boxes(residual, jacobian, lower, upper, damping=0., *, kkt_tol=1e-10):
    """Solve independent damped, two-variable box least-squares steps.

    For each period minimize 1/2||r+J*d||^2 + damping/2||d||^2.  The interior
    solution uses an augmented square-root QR, not inversion of J'J.  Four edge
    minima and four corners complete all active sets of this convex rectangle.
    Rank-deficient QR rows use SVD least squares.  A full box-gradient mapping
    is checked at the selected solution, including when the best point is flat.
    """
    residual, jacobian = np.asarray(residual, float), np.asarray(jacobian, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if jacobian.ndim != 3 or jacobian.shape[-1] != 2 or residual.shape != jacobian.shape[:2]:
        raise ValueError("expected residual[T,k] and Jacobian[T,k,2]")
    t = len(residual)
    if lower.shape != (t, 2) or upper.shape != (t, 2) or np.any(lower > upper):
        raise ValueError("step boxes must have shape (T,2) and lower <= upper")
    damping = np.broadcast_to(np.asarray(damping, float), (t,)).copy()
    if not all(np.isfinite(a).all() for a in (residual, jacobian, lower, upper, damping)) or \
            np.any(damping < 0) or not np.isfinite(kkt_tol) or kkt_tol <= 0:
        raise ValueError("finite data, nonnegative damping and positive KKT tolerance required")
    aug = np.concatenate((jacobian, np.sqrt(damping)[:, None, None]*np.eye(2)[None]), axis=1)
    target = np.concatenate((-residual, np.zeros((t, 2))), axis=1)
    q, r = np.linalg.qr(aug, mode="reduced")
    projected_target = np.einsum("tki,tk->ti", q, target)
    diagonal = np.abs(np.diagonal(r, axis1=1, axis2=2))
    rank_cutoff = np.finfo(float).eps*max(aug.shape[1:])*np.max(abs(r), axis=(1, 2))
    regular = np.all(diagonal > rank_cutoff[:, None], axis=1)
    unconstrained = np.zeros((t, 2))
    if np.any(regular):
        unconstrained[regular] = np.linalg.solve(r[regular], projected_target[regular, :, None])[..., 0]
    for time in np.flatnonzero(~regular):
        unconstrained[time] = np.linalg.lstsq(aug[time], target[time], rcond=None)[0]
    h = np.einsum("tki,tkj->tij", jacobian, jacobian)
    h[:, 0, 0] += damping; h[:, 1, 1] += damping
    g = np.einsum("tki,tk->ti", jacobian, residual)
    candidates = [unconstrained]
    for fixed, free in ((0, 1), (1, 0)):
        for bound in (lower[:, fixed], upper[:, fixed]):
            step = np.zeros((t, 2)); step[:, fixed] = bound
            numerator = -(g[:, free]+h[:, free, fixed]*bound)
            step[:, free] = np.divide(numerator, h[:, free, free],
                out=np.zeros(t), where=h[:, free, free] > 0)
            step[:, free] = np.clip(step[:, free], lower[:, free], upper[:, free])
            candidates.append(step)
    for first in (lower[:, 0], upper[:, 0]):
        for second in (lower[:, 1], upper[:, 1]):
            candidates.append(np.column_stack((first, second)))
    candidates = np.stack(candidates, axis=1)
    model = residual[:, None, :]+np.einsum("tki,tai->tak", jacobian, candidates)
    values = .5*np.sum(model**2, axis=2)+.5*damping[:, None]*np.sum(candidates**2, axis=2)
    feasible = np.all((candidates >= lower[:, None, :]) & (candidates <= upper[:, None, :]), axis=2)
    values = np.where(feasible, values, np.inf)
    chosen = np.argmin(values, axis=1)
    step = candidates[np.arange(t), chosen]
    linear_residual = residual+np.einsum("tki,ti->tk", jacobian, step)
    grad = np.einsum("tki,tk->ti", jacobian, linear_residual)+damping[:, None]*step
    projected = step-np.clip(step-grad, lower, upper)
    kkt = np.max(abs(projected), axis=1)
    return dict(step=step, objective=values[np.arange(t), chosen], kkt=kkt,
                success=kkt <= kkt_tol, active_set=chosen,
                rank_deficient=~regular, gradient=grad,
                kkt_tolerance=float(kkt_tol))


class _BatchSourceProblem:
    def __init__(self, base, counter=None):
        self.base, self.counter = base, counter
        self.ac_calls = self.ac_states = 0
        self.ac_seconds = self.derivative_seconds = self.linear_seconds = 0.
        self.voltage = np.full((base.t, base.m), np.nan+1j*np.nan)
        self.loads = np.full((base.t, base.m), np.nan+1j*np.nan)
        self.zbus = (base.candidate.B*(base.fixed_r+1j*base.fixed_x))@base.candidate.B.T

    def predict(self, ids, source):
        ids = np.asarray(ids, int)
        if not len(ids):
            raise ValueError("nonempty period set required")
        base = self.base
        self.ac_calls += 1; self.ac_states += len(ids)
        if self.counter is not None:
            self.counter["ac_calls"] = self.counter.get("ac_calls", 0)+1
            self.counter["ac_time_states"] = self.counter.get("ac_time_states", 0)+len(ids)
        began = perf_counter()
        try:
            physical = lk.ac_predict(base.candidate, base.fixed_r, base.fixed_x,
                base.p[ids], base.q[ids], source[:, 0], source[:, 1],
                base.observations.root_v[ids])
        finally:
            self.ac_seconds += perf_counter()-began
        residual = np.column_stack((
            (physical["v"]+base.bias["v"][ids]-base.observations.v[ids])/base.sigmas["v"][ids],
            (physical["master_p"]-base.observations.master_p[ids])/base.sigmas["master_p"][ids],
            (physical["master_q"]-base.observations.master_q[ids])/base.sigmas["master_q"][ids]))
        if not np.isfinite(residual).all():
            raise FloatingPointError("nonfinite source residual")
        return residual, physical

    def remember(self, ids, physical, positions=None):
        if positions is None:
            self.voltage[ids], self.loads[ids] = physical["voltage"], physical["loads"]
        else:
            self.voltage[ids] = physical["voltage"][positions]
            self.loads[ids] = physical["loads"][positions]

    def jacobian(self, ids):
        base = self.base; began = perf_counter()
        try:
            physical = dict(voltage=self.voltage[ids], loads=self.loads[ids], zbus=self.zbus,
                            root_v=base.observations.root_v[ids])
            raw = fit_response_derivatives(base.candidate, physical, include_lines=False)
            return np.concatenate((
                raw["v"]/base.sigmas["v"][ids, :, None],
                raw["master_p"][:, None, :]/base.sigmas["master_p"][ids, None, None],
                raw["master_q"][:, None, :]/base.sigmas["master_q"][ids, None, None]), axis=1)
        finally:
            self.derivative_seconds += perf_counter()-began


_NUMERICAL_ERRORS = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


def _fit_batch_start(base, start, max_nfev, tol, counter):
    """Independent state machines sharing only batched numerical kernels."""
    problem = _BatchSourceProblem(base, counter)
    source = start.copy(); t, k = base.t, base.n+2
    lower, upper, scale = (a.reshape(t, 2) for a in (base.lower, base.upper, base.scale))
    margin = base.margin.reshape(t, 2)
    residual = np.full((t, k), np.nan)
    jac = np.full((t, k, 2), np.nan)
    status = np.zeros(t, int); active = np.ones(t, bool); refresh = np.ones(t, bool)
    nfev = np.zeros(t, int); njev = np.zeros(t, int); damping = np.full(t, np.nan)
    rejects = np.zeros(t, int); losses = np.full(t, np.inf)
    optimality = np.full(t, np.inf); gradients = np.full((t, 2), np.nan)
    traces = [[] for _ in range(t)]; initial_trials = [[] for _ in range(t)]
    messages = ["per-period AC evaluation budget reached" for _ in range(t)]
    initial_source = start.copy()
    ids = np.arange(t)
    try:
        values, physical = problem.predict(ids, source)
        residual[:] = values; problem.remember(ids, physical)
        for time in ids:
            initial_trials[time].append(dict(source_factor=1., ac_success=True))
    except _NUMERICAL_ERRORS:
        # Isolate a bad row rather than marking all periods as physically failed.
        for time in ids:
            for power in range(13):
                factor = .5**power
                trial = np.clip(factor*initial_source[time], lower[time]+margin[time], upper[time]-margin[time])
                try:
                    values, physical = problem.predict([time], trial[None])
                    source[time] = trial; residual[time] = values[0]
                    problem.remember([time], physical)
                    initial_trials[time].append(dict(source_factor=factor, ac_success=True))
                    break
                except _NUMERICAL_ERRORS as error:
                    initial_trials[time].append(dict(source_factor=factor, ac_success=False,
                        error=type(error).__name__+": "+str(error)))
            else:
                source[time] = np.nan; status[time] = -99; active[time] = False
                messages[time] = "all same-domain source initializations failed AC"
    nfev[active] = 1
    losses[active] = .5*np.sum(residual[active]**2, axis=1)

    while np.any(active):
        changed = np.flatnonzero(active & refresh)
        if len(changed):
            try:
                jac[changed] = problem.jacobian(changed)
                njev[changed] += 1; refresh[changed] = False
            except _NUMERICAL_ERRORS as error:
                # Derivative failures are isolated, too; no row can borrow success.
                for time in changed:
                    try:
                        jac[time] = problem.jacobian([time])[0]
                        njev[time] += 1; refresh[time] = False
                    except _NUMERICAL_ERRORS as row_error:
                        active[time] = False; status[time] = -98
                        messages[time] = "AC derivative failed: "+repr(row_error)
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        gradients[ids] = np.einsum("tki,tk->ti", jac[ids], residual[ids])
        _, scaled_pg = _gradient_mapping(source[ids], gradients[ids], lower[ids], upper[ids], scale[ids])
        optimality[ids] = np.max(abs(scaled_pg), axis=1)
        converged = ids[optimality[ids] <= tol]
        active[converged] = False; status[converged] = 1
        for time in converged:
            messages[time] = "per-period scaled projected-gradient tolerance reached"
        budget = np.flatnonzero(active & (nfev >= max_nfev))
        active[budget] = False
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        scaled_jac = jac[ids]*scale[ids, None, :]
        first = ~np.isfinite(damping[ids])
        if np.any(first):
            first_ids = ids[first]
            norms2 = np.sum(scaled_jac[first]**2, axis=1)
            damping[first_ids] = np.maximum(1e-8, 1e-3*np.max(norms2, axis=1))
        began = perf_counter()
        step = solve_source_boxes(residual[ids], scaled_jac,
            (lower[ids]-source[ids])/scale[ids], (upper[ids]-source[ids])/scale[ids],
            damping[ids], kkt_tol=max(1e-10, .1*tol))
        problem.linear_seconds += perf_counter()-began
        bad = ~step["success"]
        for position in np.flatnonzero(bad):
            time = ids[position]
            status[time] = -4; active[time] = False
            messages[time] = "two-dimensional convex step did not meet full box KKT tolerance"
            traces[time].append(dict(nfev=int(nfev[time]), inner_kkt=float(step["kkt"][position]),
                accepted=False, inner_success=False))
        positions = np.flatnonzero(~bad)
        if not len(positions):
            continue
        ids = ids[positions]
        delta = scale[ids]*step["step"][positions]
        trial_source = np.clip(source[ids]+delta, lower[ids], upper[ids])
        predicted = residual[ids]+np.einsum("tki,ti->tk", jac[ids], trial_source-source[ids])
        prediction = losses[ids]-.5*np.sum(predicted**2, axis=1)
        step_small = np.linalg.norm(delta, axis=1) <= tol*(tol+np.linalg.norm(source[ids], axis=1))
        trial_residual = np.full((len(ids), k), np.nan)
        trial_voltage = np.full((len(ids), base.m), np.nan+1j*np.nan)
        trial_loads = np.full_like(trial_voltage, np.nan+1j*np.nan)
        errors = [None]*len(ids)
        try:
            trial_residual, physical = problem.predict(ids, trial_source)
            trial_voltage[:], trial_loads[:] = physical["voltage"], physical["loads"]
        except _NUMERICAL_ERRORS as error:
            for position, time in enumerate(ids):
                try:
                    value, physical = problem.predict([time], trial_source[position:position+1])
                    trial_residual[position] = value[0]
                    trial_voltage[position], trial_loads[position] = physical["voltage"][0], physical["loads"][0]
                except _NUMERICAL_ERRORS as row_error:
                    errors[position] = type(row_error).__name__+": "+str(row_error)
        nfev[ids] += 1
        trial_loss = .5*np.sum(trial_residual**2, axis=1)
        actual = losses[ids]-trial_loss
        gain = np.divide(actual, prediction, out=np.full(len(ids), -np.inf), where=prediction > 0)
        # A zero rounded objective change at a tiny step can still improve the
        # returned stationarity.  It does not increase the evaluated AC loss.
        equal_tiny = (actual == 0) & step_small
        accepted = np.isfinite(trial_loss) & (((actual > 0) & (gain >= 1e-4)) | equal_tiny)
        for position, time in enumerate(ids):
            record = dict(nfev=int(nfev[time]), loss=2*float(losses[time]),
                damping=float(damping[time]), inner_kkt=float(step["kkt"][positions[position]]),
                inner_success=True, accepted=bool(accepted[position]),
                predicted_decrease=float(prediction[position]), actual_decrease=float(actual[position]),
                gain=float(gain[position]), ac_error=errors[position],
                step_norm=float(np.linalg.norm(delta[position])),
                objective_equal_tiny_step=bool(equal_tiny[position]))
            traces[time].append(record)
        good_positions = np.flatnonzero(accepted)
        if len(good_positions):
            good_ids = ids[good_positions]
            old_loss = losses[good_ids].copy()
            source[good_ids] = trial_source[good_positions]
            residual[good_ids] = trial_residual[good_positions]
            problem.voltage[good_ids] = trial_voltage[good_positions]
            problem.loads[good_ids] = trial_loads[good_positions]
            losses[good_ids] = trial_loss[good_positions]
            refresh[good_ids] = True; rejects[good_ids] = 0
            damping[good_ids] = np.where(gain[good_positions] > .75,
                np.maximum(1e-18, damping[good_ids]*.3),
                np.where(gain[good_positions] < .25, damping[good_ids]*2., damping[good_ids]))
            xtol = step_small[good_positions]
            # A relative SSE decrease alone can stop substantially before the
            # source gradient is small.  Complete one more physical tiny step
            # or meet the explicit per-period projected-gradient tolerance.
            for local in np.flatnonzero(xtol):
                time = good_ids[local]
                status[time] = 3; active[time] = False
                messages[time] = "per-period step tolerance at new accepted state"
        rejected_positions = np.flatnonzero(~accepted)
        if len(rejected_positions):
            bad_ids = ids[rejected_positions]
            damping[bad_ids] = np.minimum(1e30, damping[bad_ids]*10.)
            rejects[bad_ids] += 1
            # As in a step-tolerance stop, keep the old accepted state.  Its
            # gradient is still reported; this is not a stationarity claim.
            stagnant = np.isfinite(trial_loss[rejected_positions]) & step_small[rejected_positions]
            for local in np.flatnonzero(stagnant):
                time = bad_ids[local]; status[time] = 3; active[time] = False
                messages[time] = "per-period step tolerance at retained accepted state"
            for time in bad_ids[(rejects[bad_ids] >= 20) & active[bad_ids]]:
                status[time] = -5; active[time] = False
                messages[time] = "per-period damping retry budget exhausted"

    finite = np.isfinite(source).all(axis=1)
    ids = np.flatnonzero(finite)
    if len(ids):
        try:
            jac[ids] = problem.jacobian(ids)
            gradients[ids] = np.einsum("tki,tk->ti", jac[ids], residual[ids])
            _, final_pg = _gradient_mapping(source[ids], gradients[ids], lower[ids], upper[ids], scale[ids])
            optimality[ids] = np.max(abs(final_pg), axis=1)
        except _NUMERICAL_ERRORS as error:
            for time in ids:
                try:
                    raw = problem.jacobian([time])[0]
                    gradients[time] = raw.T@residual[time]
                    _, final_pg = _gradient_mapping(source[time], gradients[time], lower[time], upper[time], scale[time])
                    optimality[time] = float(np.max(abs(final_pg)))
                except _NUMERICAL_ERRORS as row_error:
                    status[time] = -98; messages[time] = "final gradient replay failed: "+repr(row_error)
    # Tiny source Newton steps can reach floating-point objective resolution
    # while still reducing stationarity.  A bounded polish
    # checks *both* the full-AC objective allowance and the new exact gradient;
    # it never substitutes a predicted gradient or changes failed/budget rows.
    for _ in range(3):
        ids = np.flatnonzero(finite & (status > 0) & (optimality > tol) & (nfev < max_nfev))
        if not len(ids):
            break
        scaled_jac = jac[ids]*scale[ids, None, :]
        began = perf_counter()
        step = solve_source_boxes(residual[ids], scaled_jac,
            (lower[ids]-source[ids])/scale[ids], (upper[ids]-source[ids])/scale[ids],
            0., kkt_tol=max(1e-10, .1*tol))
        problem.linear_seconds += perf_counter()-began
        good = step["success"]
        ids, delta = ids[good], step["step"][good]
        if not len(ids):
            break
        trial = np.clip(source[ids]+scale[ids]*delta, lower[ids], upper[ids])
        nfev[ids] += 1
        try:
            trial_residual, physical = problem.predict(ids, trial)
            began = perf_counter()
            raw = fit_response_derivatives(base.candidate, physical, include_lines=False)
            trial_jac = np.concatenate((raw["v"]/base.sigmas["v"][ids, :, None],
                raw["master_p"][:, None, :]/base.sigmas["master_p"][ids, None, None],
                raw["master_q"][:, None, :]/base.sigmas["master_q"][ids, None, None]), axis=1)
            problem.derivative_seconds += perf_counter()-began
            trial_gradient = np.einsum("tki,tk->ti", trial_jac, trial_residual)
            _, trial_pg = _gradient_mapping(trial, trial_gradient, lower[ids], upper[ids], scale[ids])
            trial_optimality = np.max(abs(trial_pg), axis=1)
            trial_loss = .5*np.sum(trial_residual**2, axis=1)
            allowance = 256*np.finfo(float).eps*(1.+losses[ids]+trial_loss)
            accept = (trial_loss <= losses[ids]+allowance) & (trial_optimality < optimality[ids])
            for position, time in enumerate(ids):
                traces[time].append(dict(phase="bounded_stationarity_polish",
                    nfev=int(nfev[time]), accepted=bool(accept[position]),
                    actual_decrease=float(losses[time]-trial_loss[position]),
                    objective_allowance=float(allowance[position]),
                    old_projected_gradient=float(optimality[time]),
                    new_projected_gradient=float(trial_optimality[position])))
            positions = np.flatnonzero(accept)
            if not len(positions):
                break
            accepted_ids = ids[positions]
            source[accepted_ids] = trial[positions]
            residual[accepted_ids] = trial_residual[positions]
            jac[accepted_ids] = trial_jac[positions]
            gradients[accepted_ids] = trial_gradient[positions]
            losses[accepted_ids] = trial_loss[positions]
            optimality[accepted_ids] = trial_optimality[positions]
            problem.remember(accepted_ids, physical, positions)
            for time in accepted_ids[optimality[accepted_ids] <= tol]:
                status[time] = 1
                messages[time] = "per-period scaled projected-gradient tolerance after bounded polish"
        except _NUMERICAL_ERRORS as error:
            # This optional refinement cannot erase a previously retained fit.
            for time in ids:
                traces[time].append(dict(phase="bounded_stationarity_polish", accepted=False,
                                         error=type(error).__name__+": "+str(error)))
            break
    rows = []
    for time in range(t):
        physical_ok = bool(finite[time] and
            np.min(abs(problem.voltage[time])) >= base.voltage_bounds[0] and
            np.max(abs(problem.voltage[time])) <= base.voltage_bounds[1])
        rows.append(dict(time=time, success=bool(status[time] > 0 and physical_ok),
            solver_success=bool(status[time] > 0), status=int(status[time]), message=messages[time],
            source=source[time].tolist(), requested_initial_source=initial_source[time].tolist(),
            loss=2*float(losses[time]), nfev=int(nfev[time]), njev=int(njev[time]),
            optimality=float(optimality[time]), physical_voltage_feasible=physical_ok,
            initial_ac_trials=initial_trials[time], trace=traces[time]))
    return source, rows, problem


def fit_batched_sources(candidate, observations, model, *, fixed_r, fixed_x,
                        initial=None, r_bounds, x_bounds, ratio_bounds,
                        source_p_bounds, source_q_bounds, voltage_bounds,
                        multistart=1, max_nfev=250, tol=1e-8, seed=0,
                        call_counter=None):
    """Fixed-line source fitting with independent stops and batched numerics."""
    began = perf_counter()
    for value, name in ((multistart, "multistart"), (max_nfev, "max_nfev")):
        if int(value) != value or value < 1:
            raise ValueError(name+" must be a positive integer")
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
        call_counter["fit_calls"] = call_counter.get("fit_calls", 0)+1
    lower, upper, margin = (a.reshape(base.t, 2) for a in (base.lower, base.upper, base.margin))
    starts = [base.start.reshape(base.t, 2).copy()]
    rng = np.random.default_rng(seed)
    for _ in range(1, int(multistart)):
        starts.append(np.clip(starts[0]+rng.normal(0., .01, starts[0].shape), lower+margin, upper-margin))
    runs, kernels = [], []
    for index, start in enumerate(starts):
        if call_counter is not None:
            call_counter["solver_calls"] = call_counter.get("solver_calls", 0)+1
            call_counter["source_period_solver_calls"] = call_counter.get("source_period_solver_calls", 0)+base.t
        source, rows, kernel = _fit_batch_start(base, start, int(max_nfev), tol, call_counter)
        for row in rows:
            row["restart"] = index
        runs.append((source, rows)); kernels.append(kernel)
    selected = []
    source = np.full((base.t, 2), np.nan)
    for time in range(base.t):
        eligible = [(index, rows[time]) for index, (_, rows) in enumerate(runs)
                    if np.isfinite(rows[time]["loss"])]
        if not eligible:
            index, row = 0, runs[0][1][time]
        else:
            index, row = min(eligible, key=lambda pair: (not pair[1]["success"], pair[1]["loss"]))
        source[time] = runs[index][0][time]
        selected.append(dict(**row, selected_restart=index,
            restarts=[rows[time] for _, rows in runs]))
    failed = [row["time"] for row in selected if not row["success"]]
    full_replay = dict(attempted=False, passed=False)
    physical = None; loss = np.inf; gradient = np.full_like(source, np.nan)
    pg = np.full_like(source, np.nan); scaled_pg = np.full_like(source, np.nan)
    replay_began = perf_counter()
    if np.isfinite(source).all():
        full_replay["attempted"] = True
        try:
            if call_counter is not None:
                call_counter["ac_time_states"] = call_counter.get("ac_time_states", 0)+base.t
            residual = base.evaluate(source.ravel())
            physical = base.cache["physical"]
            loss = float(residual@residual)
            raw = fit_response_derivatives(candidate, physical, include_lines=False)
            jac = np.concatenate((raw["v"]/base.sigmas["v"][:, :, None],
                raw["master_p"][:, None, :]/base.sigmas["master_p"][:, None, None],
                raw["master_q"][:, None, :]/base.sigmas["master_q"][:, None, None]), axis=1)
            gradient = np.einsum("tki,tk->ti", jac, residual.reshape(base.t, base.n+2))
            pg, scaled_pg = _gradient_mapping(source, gradient, lower, upper, base.scale.reshape(base.t, 2))
            voltage = abs(physical["voltage"])
            voltage_ok = bool(np.min(voltage) >= voltage_bounds[0] and np.max(voltage) <= voltage_bounds[1])
            source_ok = bool(np.all(source >= lower) and np.all(source <= upper))
            full_replay.update(passed=bool(voltage_ok and source_ok and np.isfinite(loss)),
                source_box_feasible=source_ok, physical_voltage_feasible=voltage_ok,
                ac_equation_residual=float(physical["ac_equation_residual"]),
                power_balance_residual=float(physical["power_balance_residual"]))
        except _NUMERICAL_ERRORS as error:
            full_replay["error"] = type(error).__name__+": "+str(error)
    success = bool(not failed and full_replay["passed"])
    if success:
        status, message = 1, "all independent batched source periods converged and full AC replay passed"
    elif any(row["status"] == 0 for row in selected):
        status, message = 0, "one or more source periods exhausted their independent budget"
    else:
        status, message = -99, "one or more source periods or full AC replay failed"
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_record = mapping.constraint_record(base.fixed_r, base.fixed_x)
    selected_nfev = [row["nfev"] for row in selected]
    selected_opt = [row["optimality"] for row in selected]
    diagnostics = dict(objective="conditional_known_diagonal_Gaussian_output_given_measured_PQ",
        measurement_channels=list(OUTPUT_CHANNELS), source_node=1,
        source_spatial_location_shared=True, source_joint_voltage_and_master_estimation=True,
        source_anchored=True, root_voltage_known=True, line_parameters_fixed=True, fixed_lines=True,
        P_Q_inputs_fixed=True, P_Q_measurement_error_propagated=False, P_Q_likelihood_included=False,
        input_EV_already_in_total_PQ=True, source_sign="positive_consumption", impedance_units="pu",
        bias_policy="known_calibrated", line_domain="original_box_intersect_X_over_R_cone",
        ratio_constraint_enforced=True, ratio_bounds=[mapping.rho_low, mapping.rho_high],
        ratio_constraints=ratio_record, shared_parameter_count=0, source_parameter_count=2*base.t,
        joint_parameter_count=2*base.t, optimizer_parameter_count_per_period=2,
        residual_count=base.t*(base.n+2), output_normalizer=base.normalizer,
        solver="period_independent_batched_source_LM_square_root_QR_2d_box_steps",
        full_global_Jacobian_formed=False, source_periods_independent=True, temporal_source_prior=False,
        converged_periods_frozen=True, damping_and_acceptance_per_period=True,
        stopping_policy="per-period projected gradient or tiny step; SSE decrease alone does not stop",
        stationarity_polish="up to three undamped box steps; full-AC loss within a declared FP64-sized acceptance tolerance and exact projected gradient decreases",
        convex_step_active_sets=9, convex_step_active_sets_are_EV_actions=False,
        max_nfev_per_period_per_restart=int(max_nfev), multistart_per_period=int(multistart),
        budget_scope="independent max_nfev per period per restart, not one batched budget",
        legacy_global_budget_equivalent=False,
        status_scope="aggregate period outcome; row-specific numerical stop reasons retained",
        nfev_scope="sum of selected per-period restarts; not batched full-history calls",
        nfev_selected_sum=int(sum(selected_nfev)), nfev_selected_max=int(max(selected_nfev, default=0)),
        total_nfev=int(sum(row["nfev"] for _, rows in runs for row in rows)),
        solver_calls=int(multistart), source_period_solver_calls=int(multistart)*base.t,
        ac_evaluations=sum(kernel.ac_calls for kernel in kernels)+base.ac_calls,
        ac_time_states=sum(kernel.ac_states for kernel in kernels)+base.t*base.ac_calls,
        failed_periods=failed, failed_period_count=len(failed), periods=selected,
        full_history_replay=full_replay, gradient=gradient.tolist(),
        gradient_inf=float(np.max(abs(gradient))), projected_gradient_inf=float(np.max(abs(pg))),
        scaled_projected_gradient_inf=float(np.max(abs(scaled_pg))),
        optimizer_optimality_max=float(max(selected_opt, default=np.inf)),
        information_spectrum_available=False, global_optimality_certified=False,
        global_loss_lower_bound=None, loglikelihood_is_topology_probability=False,
        local_identifiability_full_column_rank=None,
        ac_solution_branch="root_initialized_fixed_point; other equilibria not enumerated",
        timings=dict(row_batch_AC_seconds=sum(kernel.ac_seconds for kernel in kernels),
            row_batch_derivative_seconds=sum(kernel.derivative_seconds for kernel in kernels),
            row_batch_linear_seconds=sum(kernel.linear_seconds for kernel in kernels),
            full_replay_seconds=perf_counter()-replay_began),
        failure_kind=None if success else "retained_period_or_replay_failure",
        seconds=perf_counter()-began)
    if physical is None:
        predicted = dict(v=np.full_like(base.p, np.nan), p=observations.p.copy(), q=observations.q.copy())
        diagnostics["physical_voltage_feasible"] = False
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
        loss, -.5*(loss+base.normalizer), success, status, message, int(sum(selected_nfev)),
        float(max(selected_opt, default=np.inf)), predicted, diagnostics)


__all__ = ["solve_source_boxes", "fit_batched_sources"]
