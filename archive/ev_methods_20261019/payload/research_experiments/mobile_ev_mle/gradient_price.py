"""Full AC profile-loss prices for short, single-site EV experiments.

History is fixed. Each reference prediction uses only its historical fit;
all candidates refit one shared R/X vector and all H+N latent total P/Q and
same-interface source states. Prices differentiate additive future readings,
not historical meters. The retained fits are local solutions, not global MLEs.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import signature
from time import perf_counter

import numpy as np

from .design import FutureForecast
from . import likelihood as _likelihood
from .likelihood import (MeasurementModel, Observations, _response_derivatives,
                         ac_predict, concat_observations, fit_candidate)

CHANNEL_ORDER = ("p", "q", "v", "master_p", "master_q", "source_p", "source_q")


def _array(value, shape, name):
    try:
        result = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def measurement_channels(observations):
    return tuple(k for k in CHANNEL_ORDER if getattr(observations, k) is not None)


def profile_margin_gradient(jacobian_white_kw, residual_reference, residual_competitor):
    """Half-SSE margin price for code residuals prediction minus observation."""
    j = np.asarray(jacobian_white_kw, float)
    ra, rb = np.asarray(residual_reference, float), np.asarray(residual_competitor, float)
    if j.ndim != 2 or ra.shape != (j.shape[0],) or rb.shape != ra.shape:
        raise ValueError("Jacobian and future residual dimensions differ")
    if not all(np.isfinite(x).all() for x in (j, ra, rb)):
        raise ValueError("Jacobian and residuals must be finite")
    return j.T@(ra-rb)


def whitened_residual(fit, observations, model):
    """Return time-major (prediction-observation)/sigma, exactly as the fit."""
    blocks = []
    for name in measurement_channels(observations):
        value = np.asarray(getattr(observations, name), float)
        sigma = _array(getattr(model, name+"_sigma"), value.shape, name+"_sigma")
        if np.any(sigma <= 0):
            raise ValueError("measurement sigmas must be positive")
        prediction = _array(fit.predicted[name], value.shape, "fitted."+name)
        blocks.append(((prediction-value)/sigma).reshape(observations.t, -1))
    return np.concatenate(blocks, axis=1).ravel()


def _copy_observations(observations):
    values = {k: (None if getattr(observations, k) is None else
                  np.asarray(getattr(observations, k), float).copy())
              for k in (*CHANNEL_ORDER, "root_v")}
    return Observations(**values)


def _observation_hash(observations):
    digest = sha256()
    for name in (*CHANNEL_ORDER, "root_v"):
        value = getattr(observations, name)
        digest.update(name.encode())
        if value is not None:
            array = np.asarray(value, dtype="<f8")
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _parameter_domain(value, shape, name):
    try:
        low, high = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} requires a lower/upper pair") from error
    low, high = _array(low, shape, name+".lower"), _array(high, shape, name+".upper")
    if np.any(low >= high):
        raise ValueError(f"{name} requires lower < upper")
    if name in ("r_bounds", "x_bounds") and np.any(low < 0):
        raise ValueError(f"{name} requires nonnegative lower bounds")
    return low, high


def _check_parameter_point(value, domain, name):
    low, high = domain
    point = _array(value, low.shape, name)
    tolerance = 1e-10*np.maximum(1., np.maximum(abs(low), abs(high)))
    if np.any(point < low-tolerance) or np.any(point > high+tolerance):
        raise ValueError(f"{name} violates declared joint-fit parameter bounds")
    return point


def _models_for_future(history, model, horizon, future_model):
    """Explicitly assemble H/N sigmas; never silently reuse H-shaped arrays."""
    if model.fit_bias or model.calibration is not None:
        raise ValueError("gradient design requires fixed independently calibrated biases")
    future = model if future_model is None else future_model
    if future.fit_bias or future.calibration is not None:
        raise ValueError("future biases must be fixed")
    for name in ("p", "q", "v"):
        if not np.array_equal(np.broadcast_to(getattr(model, "bias_"+name), (history.n,)),
                              np.broadcast_to(getattr(future, "bias_"+name), (history.n,))):
            raise ValueError("historical and future calibrated biases must agree")
    future_values, joint_values = {}, {}
    for name in CHANNEL_ORDER:
        field = name+"_sigma"
        if getattr(history, name) is None:
            if getattr(model, field) is not None or getattr(future, field) is not None:
                raise ValueError(f"sigma supplied for absent {name} channel")
            future_values[field] = joint_values[field] = None
            continue
        tail = (history.n,) if name in ("p", "q", "v") else ()
        old = _array(getattr(model, field), (history.t, *tail), field)
        new = _array(getattr(future, field), (horizon, *tail), "future."+field)
        if np.any(old <= 0) or np.any(new <= 0):
            raise ValueError("all measurement sigmas must be positive")
        future_values[field] = new
        joint_values[field] = np.concatenate((old, new))
    return replace(model, **future_values), replace(model, **joint_values)


class GradientPriceEvaluator:
    """Predict/re-fit ordered topology margins and their physical EV prices.

    ``evaluate(site, power_kw)`` returns all ordered pairs. Its score is the
    minimum pair's mean improvement of half-SSE likelihood margin relative
    to history. Fixed ``design_noise_samples`` are whitened integration draws
    [sample, time, channel-width], not the simulator's future noise.

    The mean-prediction default is a plug-in approximation, not an exact
    expectation. Passing a future_model is necessary for time-dependent
    sigmas that cannot broadcast from H to N. Forecasts, if supplied, must
    already be made from H and declared assumptions only.
    """

    def __init__(self, history, candidates, historical_fits, model, *, horizon,
                 base_kva=20., q_over_p=0., forecasts=None,
                 known_history_ev_kw=None, future_model=None,
                 design_noise_samples=None, fit_options=None,
                 voltage_limits=(.9, 1.1), seed=0):
        self.history = _copy_observations(history)
        self.history_hash = _observation_hash(self.history)
        self.candidates = tuple(candidates)
        if len(self.candidates) < 2 or len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError("at least two candidates with unique names are required")
        if any(c.n != history.n for c in self.candidates):
            raise ValueError("candidate terminal dimensions differ")
        if int(horizon) != horizon or horizon < 1:
            raise ValueError("positive integer horizon required")
        if not np.isfinite(base_kva) or base_kva <= 0 or not np.isfinite(q_over_p):
            raise ValueError("positive base_kva and finite q_over_p required")
        self.horizon, self.base_kva, self.q_over_p = int(horizon), float(base_kva), float(q_over_p)
        self.model = model
        self.future_model, self.joint_model = _models_for_future(history, model, self.horizon, future_model)
        self.channels = measurement_channels(history)
        self.width = sum(history.n if k in ("p", "q", "v") else 1 for k in self.channels)
        self.fit_options = dict(multistart=2, max_nfev=150, linear_solver="exact", tol=1e-10)
        self.fit_options.update(fit_options or {})
        if "initial" in self.fit_options or "seed" in self.fit_options:
            raise ValueError("initial and seed are managed by the evaluator")
        self.seed = int(seed)
        defaults = signature(fit_candidate).parameters
        self.estimate_source = self.fit_options.get("estimate_source", defaults["estimate_source"].default)
        if not isinstance(self.estimate_source, (bool, np.bool_)):
            raise ValueError("estimate_source must be boolean")
        self.estimate_source = bool(self.estimate_source)
        self.parameter_domains, self.history_domain_checks = {}, {}
        self.current_parameter_bounds = {}
        joint_t = self.history.t+self.horizon
        for candidate in self.candidates:
            domains = {}
            for name, shape in (("r", (candidate.m,)), ("x", (candidate.m,)),
                                ("p", (joint_t, history.n)), ("q", (joint_t, history.n)),
                                ("source", (joint_t, 2))):
                field = name+"_bounds"
                domains[name] = _parameter_domain(
                    self.fit_options.get(field, defaults[field].default), shape, field)
            self.parameter_domains[candidate.name] = domains
            self.current_parameter_bounds[candidate.name] = {
                name:dict(lower=lo.tolist(), upper=hi.tolist()) for name, (lo, hi) in domains.items()}
        self.voltage_limits = tuple(voltage_limits)
        if len(self.voltage_limits) != 2 or not 0 < self.voltage_limits[0] < self.voltage_limits[1]:
            raise ValueError("positive ordered prediction voltage limits required")
        self.historical_fits = dict(historical_fits)
        self.history_half_loss, self.forecasts = {}, {}
        previous = (np.zeros_like(history.p) if known_history_ev_kw is None else
                    _array(known_history_ev_kw, history.p.shape, "known_history_ev_kw")/self.base_kva)
        for candidate in self.candidates:
            fitted = self.historical_fits.get(candidate.name)
            if fitted is None or not fitted.success or fitted.p.shape != history.p.shape:
                raise ValueError(f"successful H-only fit required for {candidate.name}")
            domain = self.parameter_domains[candidate.name]
            _check_parameter_point(fitted.r, domain["r"], "historical.r")
            _check_parameter_point(fitted.x, domain["x"], "historical.x")
            for field in ("p", "q"):
                _check_parameter_point(getattr(fitted, field),
                    tuple(b[:history.t] for b in domain[field]), "historical."+field)
            historic_source = np.c_[fitted.source_p, fitted.source_q]
            _check_parameter_point(historic_source,
                tuple(b[:history.t] for b in domain["source"]), "historical.source")
            count = fitted.diagnostics.get("joint_parameter_count")
            count_fixed = 2*candidate.m+history.t*(2*history.n)
            count_free = count_fixed+2*history.t
            if count not in (count_fixed, count_free):
                raise ValueError("historical estimate_source mode cannot be inferred from parameter count")
            historic_mode = count == count_free
            if historic_mode != self.estimate_source:
                raise ValueError("historical estimate_source mode differs from declared joint fit")
            if not self.estimate_source and np.any(abs(historic_source) > 1e-12):
                raise ValueError("fixed-zero source mode requires zero historical source states")
            self.history_domain_checks[candidate.name] = dict(point_feasible=True,
                estimate_source=historic_mode, source_mode_inference="joint_parameter_count",
                historical_original_bounds_known=False,
                historical_minimum_over_current_domain_certified=False)
            residual = whitened_residual(fitted, self.history, model)
            recomputed = float(residual@residual)
            if not np.isfinite(fitted.loss) or abs(recomputed-fitted.loss) > 1e-6*max(1., abs(fitted.loss)):
                raise ValueError("historical fit does not match these meters and sigmas")
            self.history_half_loss[candidate.name] = .5*recomputed
            provided = None if forecasts is None else forecasts.get(candidate.name)
            if provided is None:
                default = FutureForecast.from_history(self.history, fitted, self.horizon)
                forecast = FutureForecast(
                    np.tile(np.mean(fitted.p-previous, axis=0), (self.horizon, 1)),
                    np.tile(np.mean(fitted.q-self.q_over_p*previous, axis=0), (self.horizon, 1)),
                    default.source_p, default.source_q, default.root_v)
            else:
                forecast = provided
            # Defensive copies ensure later evaluation cannot update the forecast.
            self.forecasts[candidate.name] = FutureForecast(
                _array(forecast.p, (self.horizon, history.n), "forecast.p"),
                _array(forecast.q, (self.horizon, history.n), "forecast.q"),
                _array(forecast.source_p, (self.horizon,), "forecast.source_p"),
                _array(forecast.source_q, (self.horizon,), "forecast.source_q"),
                _array(forecast.root_v, (self.horizon,), "forecast.root_v"))
        if design_noise_samples is None:
            samples = np.zeros((1, self.horizon, self.width))
            self.prediction_mode = "mean_prediction"
        else:
            samples = np.asarray(design_noise_samples, float)
            if samples.ndim == 2:
                samples = samples[None]
            if samples.ndim != 3 or samples.shape[1:] != (self.horizon, self.width) or not len(samples):
                raise ValueError("design_noise_samples must have shape (S,horizon,channel_width)")
            if not np.isfinite(samples).all():
                raise ValueError("design integration samples must be finite")
            self.prediction_mode = "fixed_sample_average"
        self.design_noise_samples = samples.copy()
        self.sample_hash = sha256(samples.astype("<f8").tobytes()).hexdigest()
        self.fit_calls = self.ac_evaluations = self.response_ac_calls = 0

    def future_response(self, reference, site, power_kw):
        """Return future meter means and exact white Jacobian w.r.t. kW."""
        if not isinstance(site, (int, np.integer)) or not 0 <= site < self.history.n:
            raise ValueError("site is outside terminal set")
        u = _array(power_kw, (self.horizon,), "power_kw")
        candidate = next((c for c in self.candidates if c.name == reference), None)
        if candidate is None:
            raise ValueError("unknown reference candidate")
        fitted, forecast = self.historical_fits[reference], self.forecasts[reference]
        p, q = forecast.p.copy(), forecast.q.copy()
        p[:, site] += u/self.base_kva
        q[:, site] += self.q_over_p*u/self.base_kva
        domain = self.parameter_domains[reference]
        _check_parameter_point(fitted.r, domain["r"], "reference.r")
        _check_parameter_point(fitted.x, domain["x"], "reference.x")
        for name, point in (("p", p), ("q", q)):
            _check_parameter_point(point, tuple(b[self.history.t:] for b in domain[name]), "future."+name)
        source = np.c_[forecast.source_p, forecast.source_q]
        _check_parameter_point(source,
            tuple(b[self.history.t:] for b in domain["source"]), "future.source")
        if not self.estimate_source and np.any(abs(source) > 1e-12):
            raise ValueError("fixed-zero source mode requires zero forecast source states")
        self.response_ac_calls += 1
        physical = ac_predict(candidate, fitted.r, fitted.x, p, q,
                              forecast.source_p, forecast.source_q, forecast.root_v)
        voltage = abs(physical["voltage"])
        if voltage.min() < self.voltage_limits[0] or voltage.max() > self.voltage_limits[1]:
            raise ValueError("reference predicted voltage violates declared limits")
        derivatives = _response_derivatives(candidate, physical, True)
        values, jacobians, slices, offset = {}, [], {}, 0
        for name in self.channels:
            raw = np.asarray(physical[name], float)
            if name in ("p", "q", "v"):
                raw = raw+_array(getattr(self.model, "bias_"+name), raw.shape, "bias_"+name)
            values[name] = raw
            width = self.history.n if name in ("p", "q", "v") else 1
            sigma = _array(getattr(self.future_model, name+"_sigma"), raw.shape, name+"_sigma").reshape(self.horizon, width)
            effect = np.zeros((self.horizon, width))
            if name in ("p", "q"):
                effect[:, site] = (1. if name == "p" else self.q_over_p)/self.base_kva
            elif name not in ("source_p", "source_q"):
                deriv = derivatives[name].reshape(self.horizon, width, -1)
                effect = (deriv[:, :, 2*candidate.m+site] + self.q_over_p*
                          deriv[:, :, 2*candidate.m+self.history.n+site])/self.base_kva
            white = effect/sigma
            jacobian = np.zeros((self.horizon, width, self.horizon))
            for time in range(self.horizon):
                jacobian[time, :, time] = white[time]
            jacobians.append(jacobian)
            slices[name] = slice(offset, offset+width); offset += width
        observations = Observations(root_v=forecast.root_v.copy(), **values)
        return dict(observations=observations, physical=physical,
                    jacobian=np.concatenate(jacobians, axis=1).reshape(self.horizon*self.width, self.horizon),
                    channel_slices=slices, min_voltage=float(voltage.min()), max_voltage=float(voltage.max()))

    def _noisy_observations(self, observations, sample):
        values, offset = {}, 0
        for name in self.channels:
            value = getattr(observations, name)
            width = self.history.n if name in ("p", "q", "v") else 1
            noise = sample[:, offset:offset+width].reshape(np.asarray(value).shape)
            sigma = _array(getattr(self.future_model, name+"_sigma"), np.asarray(value).shape, name+"_sigma")
            values[name] = np.asarray(value)+sigma*noise
            offset += width
        return Observations(root_v=observations.root_v.copy(), **values)

    def evaluate(self, site, power_kw, *, with_gradient=True, initial_fits=None):
        """Evaluate every necessary reference/rival; retain all fit failures.

        ``initial_fits`` optionally maps (reference, sample, candidate) to a
        warm fit. The seed is fixed per key, independently of evaluated u.
        No failure is removed from the minimum. Finite unsuccessful fits are
        recorded but do not supply accepted scores or gradient prices.
        """
        began = perf_counter()
        if _observation_hash(self.history) != self.history_hash:
            raise RuntimeError("fixed historical observations were modified")
        u = _array(power_kw, (self.horizon,), "power_kw")
        warm = dict(initial_fits or {})
        pairs, failures, records, fit_results = [], [], [], {}
        start_counts = (self.fit_calls, self.ac_evaluations, self.response_ac_calls)
        for ref_index, reference in enumerate(self.candidates):
            try:
                response = self.future_response(reference.name, site, u)
            except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures.append(dict(reference=reference.name, kind="prediction", error=str(error)))
                for rival in self.candidates:
                    if rival.name != reference.name:
                        pairs.append(dict(reference=reference.name, competitor=rival.name,
                                          success=False, score=None, gradient=None, samples=[]))
                continue
            sample_records = []
            for sample_index, noise in enumerate(self.design_noise_samples):
                bank = {}
                try:
                    future = self._noisy_observations(response["observations"], noise)
                    combined = concat_observations(self.history, future)
                except (ValueError, RuntimeError) as error:
                    failures.append(dict(reference=reference.name, sample=sample_index, kind="design_sample", error=str(error)))
                    sample_records.append(bank); continue
                for candidate_index, candidate in enumerate(self.candidates):
                    key = (reference.name, sample_index, candidate.name)
                    try:
                        self.fit_calls += 1
                        original_ac, call_count = _likelihood.ac_predict, [0]
                        def counted_ac(*args, **kwargs):
                            call_count[0] += 1
                            return original_ac(*args, **kwargs)
                        # This evaluator is serial. Restore the module function even
                        # when an optimizer/AC call raises before diagnostics exist.
                        _likelihood.ac_predict = counted_ac
                        try:
                            fitted = fit_candidate(candidate, combined, self.joint_model,
                                initial=warm.get(key, self.historical_fits[candidate.name]),
                                seed=self.seed+10000*ref_index+100*sample_index+candidate_index,
                                **self.fit_options)
                        finally:
                            _likelihood.ac_predict = original_ac
                            self.ac_evaluations += call_count[0]
                        fit_results[key] = fitted
                        row = dict(reference=reference.name, sample=sample_index,
                                   topology=candidate.name,
                                   initial_source="explicit_warm_fit" if key in warm else "historical_fit_and_new_meters",
                                   counted_ac_calls=call_count[0], **fitted.summary())
                        records.append(row)
                        if not fitted.success or not np.isfinite(fitted.loss):
                            failures.append(dict(reference=reference.name, sample=sample_index,
                                topology=candidate.name, kind="joint_fit", summary=fitted.summary()))
                            continue
                        rho = whitened_residual(fitted, combined, self.joint_model)
                        future_rho = rho.reshape(combined.t, self.width)[self.history.t:].ravel()
                        historical_rho = rho.reshape(combined.t, self.width)[:self.history.t].ravel()
                        bank[candidate.name] = dict(half_loss=.5*fitted.loss,
                            future_half_loss=.5*float(future_rho@future_rho),
                            history_half_loss=.5*float(historical_rho@historical_rho), residual=future_rho)
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                        failures.append(dict(reference=reference.name, sample=sample_index,
                                             topology=candidate.name, kind="joint_fit_exception", error=str(error)))
                sample_records.append(bank)
            for rival in self.candidates:
                if rival.name == reference.name:
                    continue
                samples = []
                for sample_index, bank in enumerate(sample_records):
                    if reference.name not in bank or rival.name not in bank:
                        continue
                    ref, comp = bank[reference.name], bank[rival.name]
                    historic_margin = self.history_half_loss[rival.name]-self.history_half_loss[reference.name]
                    score = comp["half_loss"]-ref["half_loss"]-historic_margin
                    delta = ref["residual"]-comp["residual"]
                    gradient = profile_margin_gradient(response["jacobian"], ref["residual"], comp["residual"])
                    components = {}
                    for channel, sl in response["channel_slices"].items():
                        mask = np.zeros((self.horizon, self.width), bool); mask[:, sl] = True
                        components[channel] = response["jacobian"][mask.ravel()].T@delta[mask.ravel()]
                    samples.append(dict(sample=sample_index, score=float(score), gradient=gradient,
                        pq_only_gradient=components["p"]+components["q"], channel_gradients=components,
                        history_margin_change=comp["history_half_loss"]-ref["history_half_loss"]-historic_margin,
                        future_margin=comp["future_half_loss"]-ref["future_half_loss"],
                        joint_half_cost_reference=ref["half_loss"], joint_half_cost_competitor=comp["half_loss"],
                        joint_history_half_cost_reference=ref["history_half_loss"],
                        joint_history_half_cost_competitor=comp["history_half_loss"],
                        future_half_cost_reference=ref["future_half_loss"],
                        future_half_cost_competitor=comp["future_half_loss"]))
                success = len(samples) == len(self.design_noise_samples)
                pair = dict(reference=reference.name, competitor=rival.name, success=success,
                    pair_id=reference.name+"->"+rival.name,
                    score=float(np.mean([s["score"] for s in samples])) if success else None,
                    gradient=np.mean([s["gradient"] for s in samples], axis=0) if success and with_gradient else None,
                    pq_only_gradient=np.mean([s["pq_only_gradient"] for s in samples], axis=0) if success and with_gradient else None,
                    channel_gradients={k:np.mean([s["channel_gradients"][k] for s in samples], axis=0)
                                       for k in self.channels} if success and with_gradient else {}, samples=samples,
                    historical_margin=self.history_half_loss[rival.name]-self.history_half_loss[reference.name],
                    history_half_cost_reference=self.history_half_loss[reference.name],
                    history_half_cost_competitor=self.history_half_loss[rival.name])
                pair["gradient_kw"] = pair["gradient"]
                pair["pq_only_gradient_kw"] = pair["pq_only_gradient"]
                if success and with_gradient:
                    pair["gradient_pq"] = pair["pq_only_gradient"]
                    pair["gradient_voltage"] = pair["channel_gradients"]["v"]
                    pair["gradient_master"] = sum((pair["channel_gradients"].get(k, np.zeros(self.horizon))
                                                   for k in ("master_p", "master_q")), np.zeros(self.horizon))
                else:
                    pair.update(gradient_pq=None, gradient_voltage=None, gradient_master=None)
                for field in ("joint_half_cost_reference", "joint_half_cost_competitor",
                              "joint_history_half_cost_reference", "joint_history_half_cost_competitor",
                              "future_half_cost_reference", "future_half_cost_competitor"):
                    pair[field] = float(np.mean([s[field] for s in samples])) if success else None
                pairs.append(pair)
        success = bool(pairs and all(p["success"] for p in pairs) and not failures)
        score = min(p["score"] for p in pairs) if success else None
        return dict(success=success, score=score, pairs=pairs, failures=failures,
                    fit_records=records, fit_results=fit_results, site=int(site), power_kw=u.copy(),
                    prediction_mode=self.prediction_mode, n_design_samples=len(self.design_noise_samples),
                    design_sample_hash=self.sample_hash, history_hash=self.history_hash,
                    current_parameter_bounds=self.current_parameter_bounds,
                    historical_domain_checks=self.history_domain_checks,
                    estimate_source=self.estimate_source,
                    prediction_uncertainty_scope="fixed physical forecast + meter noise",
                    gradient_target="retained_local_half_SSE_likelihood_margin",
                    global_optimality_certified=False,
                    fit_calls=self.fit_calls-start_counts[0], ac_evaluations=self.ac_evaluations-start_counts[1],
                    response_ac_calls=self.response_ac_calls-start_counts[2],
                    total_ac_calls=self.ac_evaluations+self.response_ac_calls-start_counts[1]-start_counts[2],
                    half_SSE=True, serial_evaluator_required=True, seconds=perf_counter()-began)


__all__ = ["GradientPriceEvaluator", "whitened_residual", "measurement_channels", "profile_margin_gradient"]
