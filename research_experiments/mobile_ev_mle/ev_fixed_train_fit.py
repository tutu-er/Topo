"""Fixed measured P/Q inputs and one two-component physical source per time.

Training estimates shared R/X; validation and future fits can fix those arrays
exactly. Only voltage magnitude and root P/Q contribute to the conditional
Gaussian output likelihood. P/Q meter errors are deliberately not propagated
in this first model. All retained optima are local attained fits.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from . import likelihood as lk


OUTPUT_CHANNELS = ("v", "master_p", "master_q")


def _indices(observations, indices):
    value = np.arange(observations.t)[indices]
    value = np.asarray(value)
    if value.ndim != 1 or not value.size:
        raise ValueError("indices must select at least one time in one dimension")
    return value


def subset_observations(observations, indices):
    """Slice all observed channels with the same explicit time selection."""
    if not isinstance(observations, lk.Observations):
        raise TypeError("observations must be Observations")
    ids = _indices(observations, indices)
    values = {}
    for name in ("p", "q", "v", "root_v", "master_p", "master_q", "source_p", "source_q"):
        value = getattr(observations, name)
        values[name] = None if value is None else np.asarray(value)[ids].copy()
    return lk.Observations(**values)


def subset_model(model, observations, indices):
    """Broadcast meter arrays on the ORIGINAL record before selecting times.

    In particular a T-vector root sigma and a (T,n) voltage sigma cannot be
    accidentally reinterpreted using the shorter record's T. Bias corrections
    also become explicit (selected T,n) arrays. Calibration data are not time
    series operating records and are preserved here; the fitter rejects a
    joint-bias/calibration model because its likelihood is outside this model.
    """
    if not isinstance(model, lk.MeasurementModel) or not isinstance(observations, lk.Observations):
        raise TypeError("MeasurementModel and Observations are required")
    ids = _indices(observations, indices)
    changes = {}
    for name in ("p", "q", "v", "master_p", "master_q", "source_p", "source_q"):
        value = getattr(model, name + "_sigma")
        if value is not None:
            shape = (observations.t, observations.n) if name in ("p", "q", "v") else (observations.t,)
            changes[name + "_sigma"] = lk._array(value, shape, name + "_sigma")[ids].copy()
    for name in ("p", "q", "v"):
        changes["bias_" + name] = lk._array(getattr(model, "bias_" + name),
            (observations.t, observations.n), "bias_" + name)[ids].copy()
    return replace(model, **changes)


def _read_initial(initial, name, default=None):
    if initial is None:
        return default
    return initial.get(name, default) if isinstance(initial, Mapping) else getattr(initial, name, default)


def _compact_bias(values):
    return values[0].copy() if np.array_equal(values, np.broadcast_to(values[0], values.shape)) else values.copy()


class _FixedInputProblem:
    """Time-major residual/Jacobian used by the fitter and derivative tests."""
    def __init__(self, candidate, observations, model, *, initial=None, fixed_r=None, fixed_x=None,
                 r_bounds=(1e-6, .1), x_bounds=(1e-6, .1), source_p_bounds=(-.5, .5),
                 source_q_bounds=(-.5, .5), voltage_bounds=(.85, 1.15), call_counter=None):
        if not isinstance(observations, lk.Observations) or not isinstance(model, lk.MeasurementModel):
            raise TypeError("Observations and MeasurementModel are required")
        if observations.n != candidate.n:
            raise ValueError("candidate and observations terminal counts differ")
        if model.fit_bias or model.calibration is not None:
            raise ValueError("fixed-input fits require known calibrated biases; no joint bias/calibration likelihood")
        if observations.master_p is None or observations.master_q is None:
            raise ValueError("joint source estimation requires root master_p and master_q observations")
        if any(getattr(observations, k) is not None or getattr(model, k + "_sigma") is not None
               for k in ("source_p", "source_q")):
            raise ValueError("this fixed-input model uses root P/Q; separate source telemetry is not supported")
        if (fixed_r is None) != (fixed_x is None):
            raise ValueError("fixed_r and fixed_x must be supplied together")
        self.candidate, self.observations, self.model = candidate, observations, model
        self.t, self.n, self.m = observations.t, observations.n, candidate.m
        self.line_fixed = fixed_r is not None
        self.shared = 0 if self.line_fixed else 2 * self.m
        self.width = self.shared + 2 * self.t
        self.voltage_bounds = np.asarray(voltage_bounds, float)
        if (self.voltage_bounds.shape != (2,) or not np.isfinite(self.voltage_bounds).all()
                or self.voltage_bounds[0] <= 0 or self.voltage_bounds[0] >= self.voltage_bounds[1]):
            raise ValueError("voltage_bounds must be finite positive lower < upper")
        self.counter, self.ac_calls, self.cache = call_counter, 0, {}
        self.bias = {k: lk._array(getattr(model, "bias_" + k), (self.t, self.n), "bias_" + k)
                     for k in ("p", "q", "v")}
        self.p = observations.p - self.bias["p"]
        self.q = observations.q - self.bias["q"]
        # P/Q sigmas document ignored meter precision but do not enter residuals.
        for k in ("p", "q"):
            sigma = lk._array(getattr(model, k + "_sigma"), (self.t, self.n), k + "_sigma")
            if (sigma <= 0).any():
                raise ValueError("supplied P/Q standard deviations must be positive")
        self.sigmas = {}
        for k in OUTPUT_CHANNELS:
            value = getattr(model, k + "_sigma")
            if value is None:
                raise ValueError(k + "_sigma is required")
            shape = (self.t, self.n) if k == "v" else (self.t,)
            sigma = lk._array(value, shape, k + "_sigma")
            if (sigma <= 0).any():
                raise ValueError("all output standard deviations must be positive")
            self.sigmas[k] = sigma
        self.normalizer = sum(float(np.log(2*np.pi*s*s).sum()) for s in self.sigmas.values())
        rlo, rhi = lk._bounds(r_bounds, (self.m,), "r_bounds")
        xlo, xhi = lk._bounds(x_bounds, (self.m,), "x_bounds")
        if (rlo < 0).any() or (xlo < 0).any():
            raise ValueError("R/X bounds must be nonnegative")
        if self.line_fixed:
            self.fixed_r = lk._array(fixed_r, (self.m,), "fixed_r")
            self.fixed_x = lk._array(fixed_x, (self.m,), "fixed_x")
            if np.any(self.fixed_r < rlo) or np.any(self.fixed_r > rhi) or np.any(self.fixed_x < xlo) or np.any(self.fixed_x > xhi):
                raise ValueError("fixed R/X must lie inside the declared line bounds")
        sp_lo, sp_hi = lk._bounds(source_p_bounds, (self.t,), "source_p_bounds")
        sq_lo, sq_hi = lk._bounds(source_q_bounds, (self.t,), "source_q_bounds")
        self.lower = np.r_[[] if self.line_fixed else np.r_[rlo, xlo], np.c_[sp_lo, sq_lo].ravel()]
        self.upper = np.r_[[] if self.line_fixed else np.r_[rhi, xhi], np.c_[sp_hi, sq_hi].ravel()]
        self.margin = np.minimum(1e-10*np.maximum(self.upper-self.lower, 1), .01*(self.upper-self.lower))
        r0 = lk._array(_read_initial(initial, "r", np.full(self.m, .012)), (self.m,), "initial.r")
        x0 = lk._array(_read_initial(initial, "x", np.full(self.m, .006)), (self.m,), "initial.x")
        source = np.c_[observations.master_p-self.p.sum(axis=1), observations.master_q-self.q.sum(axis=1)]
        for col, k in enumerate(("source_p", "source_q")):
            value = _read_initial(initial, k)
            if value is not None:
                value = np.asarray(value, float)
                if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
                    raise ValueError("warm-start source must be a nonempty finite time vector")
                kept = min(self.t, len(value))
                source[:kept, col] = value[:kept]
        self.start = np.r_[[] if self.line_fixed else np.r_[r0, x0], source.ravel()]
        self.start = np.clip(self.start, self.lower+self.margin, self.upper-self.margin)
        self.scale = np.r_[[] if self.line_fixed else np.r_[np.maximum((rhi-rlo)*.1, 1e-4),
            np.maximum((xhi-xlo)*.1, 1e-4)], np.full(2*self.t, .01)]

    def unpack(self, vector):
        vector = np.asarray(vector, float)
        if vector.shape != (self.width,):
            raise ValueError("packed parameter vector has wrong width")
        if self.line_fixed:
            r, x = self.fixed_r, self.fixed_x
        else:
            r, x = vector[:self.m], vector[self.m:2*self.m]
        return r, x, vector[self.shared:].reshape(self.t, 2)

    def evaluate(self, vector):
        if "vector" in self.cache and np.array_equal(vector, self.cache["vector"]):
            return self.cache["residual"]
        r, x, source = self.unpack(vector)
        self.ac_calls += 1
        if self.counter is not None:
            self.counter["ac_calls"] = self.counter.get("ac_calls", 0) + 1
        phy = lk.ac_predict(self.candidate, r, x, self.p, self.q, source[:, 0], source[:, 1], self.observations.root_v)
        residual = np.column_stack(((phy["v"]+self.bias["v"]-self.observations.v)/self.sigmas["v"],
            (phy["master_p"]-self.observations.master_p)/self.sigmas["master_p"],
            (phy["master_q"]-self.observations.master_q)/self.sigmas["master_q"])).ravel()
        self.cache = dict(vector=np.asarray(vector).copy(), residual=residual, physical=phy)
        return residual

    def jacobian(self, vector):
        self.evaluate(vector)
        derivatives = lk._response_derivatives(self.candidate, self.cache["physical"], True)
        answer = np.zeros((self.t*(self.n+2), self.width))
        for time in range(self.t):
            for name, offset, count in (("v", 0, self.n), ("master_p", self.n, 1), ("master_q", self.n+1, 1)):
                block = derivatives[name][time].reshape(count, 2*self.m+2*self.n+2)
                sigma = self.sigmas[name][time]
                block = block/np.asarray(sigma).reshape(-1, 1)
                rows = slice(time*(self.n+2)+offset, time*(self.n+2)+offset+count)
                if not self.line_fixed:
                    answer[rows, :self.shared] = block[:, :2*self.m]
                answer[rows, self.shared+2*time:self.shared+2*time+2] = block[:, -2:]
        return answer

    def state_record(self, vector):
        r, x, s = self.unpack(vector)
        return dict(r=r.tolist(), x=x.tolist(), source_p=s[:, 0].tolist(), source_q=s[:, 1].tolist())


def fit_fixed_inputs(candidate, observations, model, *, initial=None, fixed_r=None, fixed_x=None,
                     r_bounds=(1e-6, .1), x_bounds=(1e-6, .1), source_p_bounds=(-.5, .5),
                     source_q_bounds=(-.5, .5), voltage_bounds=(.85, 1.15), multistart=2,
                     max_nfev=250, tol=1e-10, seed=0, call_counter=None):
    """Best retained local conditional-output fit with exactly fixed input P/Q.

    With both fixed_r/fixed_x supplied the optimizer has exactly 2*T source
    variables. Otherwise shared R/X add 2*m variables. Root and voltage meters
    jointly estimate the source at physical interface1. Failed restarts and
    their attained states remain in diagnostics; no global optimum is claimed.
    """
    began = perf_counter()
    for value, name in ((multistart, "multistart"), (max_nfev, "max_nfev")):
        if int(value) != value or value < 1:
            raise ValueError(name + " must be a positive integer")
    if not np.isfinite(tol) or tol <= np.finfo(float).eps:
        raise ValueError("tol must be finite and greater than machine epsilon")
    problem = _FixedInputProblem(candidate, observations, model, initial=initial, fixed_r=fixed_r, fixed_x=fixed_x,
        r_bounds=r_bounds, x_bounds=x_bounds, source_p_bounds=source_p_bounds, source_q_bounds=source_q_bounds,
        voltage_bounds=voltage_bounds, call_counter=call_counter)
    if call_counter is not None:
        call_counter["fit_calls"] = call_counter.get("fit_calls", 0) + 1
    rng = np.random.default_rng(seed)
    starts = [problem.start]
    for _ in range(1, int(multistart)):
        trial = problem.start.copy()
        if not problem.line_fixed:
            trial[:problem.shared] *= np.exp(rng.normal(0, .5, problem.shared))
        trial[problem.shared:] += rng.normal(0, .01, 2*problem.t)
        starts.append(np.clip(trial, problem.lower+problem.margin, problem.upper-problem.margin))
    restarts, retained, initial_loss, solver_calls = [], [], None, 0
    for index, start in enumerate(starts):
        record = dict(index=index, initial_state=problem.state_record(start), success=False)
        try:
            residual = problem.evaluate(start)
            record["initial_loss"] = float(residual @ residual)
            if initial_loss is None:
                initial_loss = record["initial_loss"]
            if call_counter is not None:
                call_counter["solver_calls"] = call_counter.get("solver_calls", 0) + 1
            solver_calls += 1
            result = least_squares(problem.evaluate, start, jac=problem.jacobian, bounds=(problem.lower, problem.upper),
                x_scale=problem.scale, method="trf", tr_solver="exact", ftol=tol, xtol=tol, gtol=tol, max_nfev=int(max_nfev))
            residual = problem.evaluate(result.x)
            loss = float(residual @ residual)
            voltage = abs(problem.cache["physical"]["voltage"])
            physical = bool(np.min(voltage) >= problem.voltage_bounds[0] and np.max(voltage) <= problem.voltage_bounds[1])
            valid = bool(result.success and physical and np.isfinite(loss))
            record.update(loss=loss, success=valid, solver_success=bool(result.success), status=int(result.status),
                message=str(result.message), nfev=int(result.nfev), optimality=float(result.optimality),
                physical_voltage_feasible=physical, active_bound_count=int(np.count_nonzero(result.active_mask)),
                attained_state=problem.state_record(result.x))
            retained.append((valid, loss, result, record))
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            record.update(loss=None, solver_success=False, status=-99, message=str(error), nfev=None)
        restarts.append(record)
    base_diagnostics = dict(objective="conditional_known_diagonal_Gaussian_output_given_measured_PQ",
        measurement_channels=list(OUTPUT_CHANNELS), source_node=1, source_spatial_location_shared=True,
        source_joint_voltage_and_master_estimation=True, source_anchored=True, root_voltage_known=True,
        line_parameters_fixed=problem.line_fixed, fixed_lines=problem.line_fixed, P_Q_inputs_fixed=True,
        P_Q_measurement_error_propagated=False, P_Q_likelihood_included=False, input_EV_already_in_total_PQ=True,
        source_sign="positive_consumption", impedance_units="pu", known_x_over_r=False,
        bias_policy="known_calibrated", shared_parameter_count=problem.shared,
        source_parameter_count=2*problem.t, joint_parameter_count=problem.width,
        residual_count=problem.t*(problem.n+2), output_normalizer=problem.normalizer,
        initial_loss=initial_loss, restarts=restarts, solver="scipy.least_squares_trf_exact",
        solver_calls=solver_calls,
        total_nfev=sum(r.get("nfev") or 0 for r in restarts), ac_evaluations=problem.ac_calls,
        global_optimality_certified=False, global_loss_lower_bound=None,
        loglikelihood_is_topology_probability=False,
        ac_solution_branch="root_initialized_fixed_point; other equilibria not enumerated")
    biases = [_compact_bias(problem.bias[k]) for k in ("p", "q", "v")]
    if not retained:
        r = problem.fixed_r.copy() if problem.line_fixed else np.full(problem.m, np.nan)
        x = problem.fixed_x.copy() if problem.line_fixed else np.full(problem.m, np.nan)
        base_diagnostics.update(failure_kind="computation", seconds=perf_counter()-began)
        return lk.FitResult(candidate.name, r, x, problem.p.copy(), problem.q.copy(), np.full(problem.t, np.nan),
            np.full(problem.t, np.nan), *biases, np.inf, -np.inf, False, -99, "all numerical restarts failed", 0,
            np.inf, {}, base_diagnostics)
    valid, loss, result, record = min(retained, key=lambda item: (not item[0], item[1]))
    problem.evaluate(result.x)
    full_jac = problem.jacobian(result.x)
    singular = np.linalg.svd(full_jac*problem.scale[None, :], compute_uv=False)
    rank_tol = 1e-7*singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > rank_tol))
    nuisance = full_jac[:, problem.shared:]
    left, values, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = int(np.count_nonzero(values > (1e-10*values[0] if len(values) else 0)))
    shared = full_jac[:, :problem.shared]*problem.scale[None, :problem.shared]
    shared = shared-left[:, :nuisance_rank]@(left[:, :nuisance_rank].T@shared)
    shared_singular = np.linalg.svd(shared, compute_uv=False)
    shared_rank = int(np.count_nonzero(shared_singular > (1e-7*shared_singular[0] if len(shared_singular) else 0)))
    phy = problem.cache["physical"]
    predicted = dict(phy, p=problem.p+problem.bias["p"], q=problem.q+problem.bias["q"],
        v=phy["v"]+problem.bias["v"], physical_v=phy["v"])
    r, x, source = problem.unpack(result.x)
    finite_losses = [row["loss"] for row in restarts if row.get("loss") is not None]
    base_diagnostics.update(ac_evaluations=problem.ac_calls, physical_voltage_feasible=record["physical_voltage_feasible"],
        min_voltage=float(np.min(abs(phy["voltage"]))), max_voltage=float(np.max(abs(phy["voltage"]))),
        ac_equation_residual=float(phy["ac_equation_residual"]), power_balance_residual=float(phy["power_balance_residual"]),
        local_jacobian_rank=rank, local_jacobian_rank_tolerance=float(rank_tol),
        local_scaled_singular_values=singular.tolist(), local_shared_rank=shared_rank,
        local_projected_shared_singular_values=shared_singular.tolist(), local_source_rank=nuisance_rank,
        local_identifiability_full_column_rank=bool(rank == problem.width), rank_is_global_identifiability_certificate=False,
        active_bound_count=record["active_bound_count"],
        restart_loss_spread=float(max(finite_losses)-min(finite_losses)),
        failure_kind=None if valid else ("physical_domain" if not record["physical_voltage_feasible"] else "computation"),
        seconds=perf_counter()-began)
    return lk.FitResult(candidate.name, r.copy(), x.copy(), problem.p.copy(), problem.q.copy(),
        source[:, 0].copy(), source[:, 1].copy(), *biases, loss, -.5*(loss+problem.normalizer), valid,
        int(result.status), str(result.message), int(result.nfev), float(result.optimality), predicted, base_diagnostics)


__all__ = ["fit_fixed_inputs", "subset_observations", "subset_model", "OUTPUT_CHANNELS"]
