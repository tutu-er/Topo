"""H-only AC information centers and public metering-quality controls.

This module adds a new experiment context without changing the frozen profile
price implementation. History support is a computational loss gate. A local
flat Jacobian direction and retained numerical explanations are not confidence
sets. Public sampling counts are independent of topology truth and future noise.
"""
from __future__ import annotations
from dataclasses import fields, replace
from inspect import signature
from time import perf_counter
import numpy as np
from . import likelihood
from .likelihood import MeasurementModel, ac_predict, _response_derivatives
from .gradient_price import _models_for_future, whitened_residual, measurement_channels
from .single_site_milp import solve_site_step


SIGMA_CHANNELS = ("p", "q", "v", "master_p", "master_q", "source_p", "source_q")
TARGET_SUPPORTS = ((0, 1), (0, 2), (1, 2))
FIT_DEFAULTS = signature(likelihood.fit_candidate).parameters


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def meter_to_record(model):
    """Preserve every known sigma and calibrated bias as ordinary JSON values."""
    if not isinstance(model, MeasurementModel):
        raise TypeError("MeasurementModel required")
    if model.fit_bias or model.calibration is not None:
        raise ValueError("this context requires independently calibrated fixed biases")
    return _json({field.name: getattr(model, field.name) for field in fields(model)})


def meter_from_record(record):
    """Restore a fixed Gaussian model, preserving array shape and absent meters."""
    names = {field.name for field in fields(MeasurementModel)}
    if set(record) != names:
        raise ValueError("measurement-model fields differ from the saved schema")
    if record["fit_bias"] or record["calibration"] is not None:
        raise ValueError("free calibration likelihood is outside this context")
    values = {key: np.asarray(value, float) if isinstance(value, list) else value
              for key, value in record.items()}
    return MeasurementModel(**values)


model_from_record = meter_from_record


def quality_models(history, history_model, kind="uniform", *, horizon=4,
                   row_permutation=None):
    """Return future model, joint model and independent public sampling metadata.

    History sigma is a single-reading sigma. Any array-valued historical sigma
    must be constant in time; otherwise the underlying raw sigma is unspecified.
    Averaging acts on random error only. Fixed calibration offsets stay identical.
    """
    if history.n != 3 or horizon != 4:
        raise ValueError("the declared quality experiment requires 3 terminals and 4 slots")
    if kind == "uniform":
        raw_counts = np.full((4, 3), 5, int)
    elif kind == "temporal":
        raw_counts = np.tile(np.array([1, 9, 9, 1])[:, None], (1, 3))
    elif kind == "spatial":
        raw_counts = np.array([[9, 1, 1], [1, 9, 1], [1, 1, 9], [9, 9, 9]])
    else:
        raise ValueError("quality kind must be uniform, temporal or spatial")
    order = np.arange(4) if row_permutation is None else np.asarray(row_permutation)
    if order.shape != (4,) or not np.isin(order, np.arange(4)).all() or len(set(order.tolist())) != 4:
        raise ValueError("row_permutation must be a permutation of 0,1,2,3")
    order = order.astype(int)
    counts = raw_counts[order].copy()
    values, all_counts = {}, {}
    for name in SIGMA_CHANNELS:
        sigma = getattr(history_model, name + "_sigma")
        observed = getattr(history, name)
        if observed is None:
            if sigma is not None:
                raise ValueError("sigma provided for absent " + name + " meter")
            values[name + "_sigma"] = None
            continue
        tail = (3,) if name in ("p", "q", "v") else ()
        old = np.broadcast_to(np.asarray(sigma, float), (history.t, *tail))
        if not np.isfinite(old).all() or np.any(old <= 0) or not np.allclose(old, old[0], rtol=0., atol=0.):
            raise ValueError("quality model requires positive time-constant raw " + name + " sigma")
        repeated = counts if name == "v" else np.full((horizon, *tail), 5, int)
        values[name + "_sigma"] = np.broadcast_to(old[0], repeated.shape) / np.sqrt(repeated)
        all_counts[name] = repeated.tolist()
    future = replace(history_model, **values)
    future, joint = _models_for_future(history, history_model, horizon, future)
    record = dict(kind=kind, row_permutation=order.tolist(), v_repeat_counts=counts.tolist(),
        v_total_raw_reads=int(counts.sum()), per_channel_repeat_counts=all_counts,
        random_error_rule="sigma_single/sqrt(independent_repeat_count)",
        bias_rule="fixed calibrated offsets unchanged by averaging",
        public_plan_uses_truth=False, public_plan_uses_future_noise=False,
        history_model=meter_to_record(history_model), future_model=meter_to_record(future),
        joint_model=meter_to_record(joint))
    return future, joint, record


def draw_measurement_noise(model, n_slots, n_terminals, rng):
    """Generate independent actual measurement noise under the saved model."""
    noise = {}
    for name in SIGMA_CHANNELS:
        sigma = getattr(model, name + "_sigma")
        if sigma is None:
            continue
        shape = (n_slots, n_terminals) if name in ("p", "q", "v") else (n_slots,)
        scale = np.broadcast_to(np.asarray(sigma, float), shape)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("positive finite measurement sigma required")
        noise[name] = rng.normal(size=shape) * scale
    return noise


def cheap_quality_plan(kind, quality_record, constraints, *, n_sites=3,
                       time_limit=5., random_seed=None):
    """Optimize a public precision-price control, or sample a feasible site.

    Global/local quality score is sum_t m_t u_t (read-count times pu). Bounds
    certify only this linear operational objective, not AC information. Random
    site uses zero-score operation problems and no metering-response information.
    """
    counts = np.asarray(quality_record["v_repeat_counts"], float)
    if counts.ndim != 2 or counts.shape[1] != n_sites or np.any(counts <= 0):
        raise ValueError("quality record dimensions differ from public device")
    if kind not in ("global", "local", "random"):
        raise ValueError("quality rule must be global, local or random")
    horizon = len(counts)
    site_results = {}
    for site in range(n_sites):
        gradient = (counts.mean(axis=1) if kind == "global" else counts[:, site]
                    if kind == "local" else np.zeros(horizon))
        site_results[site] = solve_site_step(site, horizon, n_sites, constraints,
            intercepts=[0.], gradients=gradient[None], time_limit=time_limit)
    feasible = [site for site, result in site_results.items() if result["success"]]
    site_records = {str(site): {key: value for key, value in result.items() if key != "plan"}
                    for site, result in site_results.items()}
    if not feasible:
        return dict(success=False, plan=None, selected_site=None, site_records=site_records,
            failures=list(site_records), incumbent=None, best_bound=None,
            bound_scope="public linear quality operations only", kind=kind)
    if kind == "random":
        selected = int(np.random.default_rng(random_seed).choice(feasible))
    else:
        selected = min(feasible, key=lambda site: (-site_results[site]["incumbent"], site))
    result = site_results[selected]
    bounds = [item["best_bound"] for item in site_results.values()]
    bound = max(bounds) if all(value is not None for value in bounds) else None
    return dict(success=True, plan=result["plan"], selected_site=selected,
        site_records=site_records, feasible_sites=feasible,
        failures=[str(site) for site in site_results if site not in feasible],
        incumbent=result["incumbent"], best_bound=bound,
        absolute_gap=None if bound is None else bound-result["incumbent"],
        bound_scope="public linear quality operations only", kind=kind,
        uses_truth=False, uses_future_noise=False)


def history_white_jacobian(candidate, fit, history, model, *, estimate_source=True):
    """Complete time-major white derivative, matching fit_candidate columns."""
    if model.fit_bias or model.calibration is not None:
        raise ValueError("fixed independently calibrated biases required")
    physical = ac_predict(candidate, fit.r, fit.x, fit.p, fit.q,
        fit.source_p, fit.source_q, history.root_v)
    derivatives = _response_derivatives(candidate, physical, estimate_source)
    t, n, m = history.t, history.n, candidate.m
    local_width = 2*n + (2 if estimate_source else 0)
    channels = measurement_channels(history)
    width = sum(n if name in ("p", "q", "v") else 1 for name in channels)
    full = np.zeros((t*width, 2*m+t*local_width))
    offset = 0
    for name in channels:
        channel_width = n if name in ("p", "q", "v") else 1
        sigma = np.broadcast_to(np.asarray(getattr(model, name+"_sigma"), float),
            (t, n) if channel_width == n else (t,)).reshape(t, channel_width)
        derivative = np.zeros((t, channel_width, 2*m+local_width))
        if name in ("p", "q"):
            for j in range(n):
                derivative[:, j, 2*m+j+(0 if name == "p" else n)] = 1.
        elif name in ("source_p", "source_q"):
            if estimate_source:
                derivative[:, 0, 2*m+2*n+(name == "source_q")] = 1.
        else:
            derivative = derivatives[name].reshape(t, channel_width, 2*m+local_width)
        derivative = derivative/sigma[:, :, None]
        for slot in range(t):
            rows = slice(slot*width+offset, slot*width+offset+channel_width)
            full[rows, :2*m] = derivative[slot, :, :2*m]
            cols = slice(2*m+slot*local_width, 2*m+(slot+1)*local_width)
            full[rows, cols] = derivative[slot, :, 2*m:]
        offset += channel_width
    return full, physical


def _boxes(candidate, history, fit_options):
    default = FIT_DEFAULTS
    shapes = dict(r=(candidate.m,), x=(candidate.m,), p=(history.t, history.n),
                  q=(history.t, history.n), source=(history.t, 2))
    result = {}
    for name, shape in shapes.items():
        low, high = fit_options.get(name+"_bounds", default[name+"_bounds"].default)
        low = np.broadcast_to(np.asarray(low, float), shape).copy()
        high = np.broadcast_to(np.asarray(high, float), shape).copy()
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("ordered finite bounds required for " + name)
        result[name] = (low, high)
    return result


def _point(fit):
    return dict(r=np.asarray(fit.r), x=np.asarray(fit.x), p=np.asarray(fit.p),
        q=np.asarray(fit.q), source=np.c_[fit.source_p, fit.source_q])


def _active(point, boxes, tolerance=1e-7):
    active = []
    for name, value in point.items():
        low, high = boxes[name]
        tol = tolerance*np.maximum(1., high-low)
        if value.shape != low.shape or not np.isfinite(value).all() or np.any(value < low-tol) or np.any(value > high+tol):
            raise ValueError("historical point violates declared original " + name + " box")
        for side, bound in (("lower", low), ("upper", high)):
            for index in np.argwhere(abs(value-bound) <= tol):
                at = tuple(index)
                active.append(dict(parameter=name, index=index.tolist(), side=side,
                    value=float(value[at]), bound=float(bound[at])))
    return active


def embed_star(star_candidate, star_fit, cherry_candidate):
    """Embed identical AC states in a cherry whose added internal edge is zero."""
    if any(1 < len(support) < star_candidate.n for support in star_candidate.supports):
        raise ValueError("a reduced star anchor is required")
    old = {tuple(support): j for j, support in enumerate(star_candidate.supports)}
    r, x = np.zeros(cherry_candidate.m), np.zeros(cherry_candidate.m)
    inserted = []
    for index, support in enumerate(cherry_candidate.supports):
        support = tuple(support)
        if support in old:
            r[index], x[index] = star_fit.r[old[support]], star_fit.x[old[support]]
        else:
            if len(support) != 2 or cherry_candidate.n != 3:
                raise ValueError("only a single n3 cherry edge may be inserted")
            inserted.append(support)
    if len(inserted) != 1 or set(old) != {tuple(s) for s in cherry_candidate.supports if tuple(s) in old}:
        raise ValueError("cherry must preserve every star support and insert one pair")
    return r, x, inserted[0]


def _verify_fit_loss(candidate, fit, history, model):
    physical = likelihood.ac_predict(candidate, fit.r, fit.x, fit.p, fit.q,
        fit.source_p, fit.source_q, history.root_v)
    blocks = []
    for name in measurement_channels(history):
        predicted = physical[name]
        if name in ("p", "q", "v"):
            predicted = predicted + getattr(model, "bias_"+name)
        sigma = np.broadcast_to(np.asarray(getattr(model, name+"_sigma"), float), np.asarray(predicted).shape)
        blocks.append(((predicted-getattr(history, name))/sigma).ravel())
    residual = np.concatenate(blocks)
    actual = float(residual@residual)
    if not np.isfinite(actual) or abs(actual-fit.loss) > 1e-6*max(1., abs(fit.loss)):
        raise ValueError("refitted historical loss disagrees with independent AC recomputation")
    return actual


def build_centres(history, candidates, historical_fits, history_model, *,
                  horizon=4, fit_options=None, gate_half_sse=6., max_anchors=2,
                  perturb_scale_pu=.003, seed=0, refit=True):
    """H-only star anchors, two flat-direction refits and three zero-edge centers.

    Returns (InformationCenter list, diagnostic record). The first successful
    star fit is an anchor if it is within the declared historical numerical gate.
    Attempts with converged but duplicate/unsupported points remain recorded.
    """
    from .structured_information import InformationCenter, FutureForecast
    began = perf_counter()
    if history.n != 3 or int(horizon) != horizon or horizon < 1:
        raise ValueError("three-terminal context and positive integer horizon required")
    if not 1 <= max_anchors <= 2 or gate_half_sse < 0 or perturb_scale_pu <= 0:
        raise ValueError("valid gate, positive perturbation and at most two anchors required")
    options = dict(multistart=1, max_nfev=200, linear_solver="exact", tol=1e-10)
    options.update(fit_options or {})
    if "initial" in options or "seed" in options:
        raise ValueError("initial and seed managed by context")
    estimate_source = options.get("estimate_source", True)
    star = next((candidate for candidate in candidates if not any(1 < len(s) < 3 for s in candidate.supports)), None)
    record = dict(activated=False, gate_half_sse=gate_half_sse, attempts=[], anchor_records=[],
        source_uses_H_only=True, uses_truth=False, uses_future_noise=False,
        zero_edge_scope="contraction centers for local planning only; final fits retain original R/X bounds",
        support_is_confidence_set=False, original_parameter_box_shrunk=False,
        ac_calls=0, fit_calls=0, anchor_count=0)
    good = [(candidate, historical_fits.get(candidate.name)) for candidate in candidates]
    good = [(candidate, fit) for candidate, fit in good if fit is not None and fit.success and np.isfinite(fit.loss)]
    if star is None or star.name not in historical_fits or not historical_fits[star.name].success or not good:
        record.update(reason="no_successful_history_star", seconds=perf_counter()-began)
        return [], record
    best_half = min(.5*fit.loss for _, fit in good)
    fitted = historical_fits[star.name]
    record.update(star_name=star.name, historical_best_half_loss=best_half,
        star_half_loss=.5*fitted.loss, star_gap_half_sse=.5*fitted.loss-best_half)
    if .5*fitted.loss-best_half > gate_half_sse:
        record.update(reason="history_star_outside_computational_support", seconds=perf_counter()-began)
        return [], record
    boxes = _boxes(star, history, options)
    record["original_parameter_boxes"] = {name: dict(lower=lo.tolist(), upper=hi.tolist()) for name, (lo, hi) in boxes.items()}
    record["declared_refit_fit_options"] = _json(options)
    record["declared_voltage_domain"] = _json(options.get("voltage_bounds", FIT_DEFAULTS["voltage_bounds"].default))
    record["historical_original_boxes_verified_from_fit_metadata"] = False
    active = _active(_point(fitted), boxes)
    rho = whitened_residual(fitted, history, history_model)
    if abs(float(rho@rho)-fitted.loss) > 1e-6*max(1., abs(fitted.loss)):
        raise ValueError("historical star fit does not match saved meter model")
    count = fitted.diagnostics.get("joint_parameter_count")
    expected = 2*star.m+history.t*(2*history.n+(2 if estimate_source else 0))
    if count is not None and count != expected:
        raise ValueError("historical source estimation mode differs from declared refits")
    anchors = [fitted]
    record["ac_calls"] += 1  # Count the attempted alias call before any AC exception.
    jac, physical = history_white_jacobian(star, fitted, history, history_model, estimate_source=estimate_source)
    shared, local = jac[:, :2*star.m], jac[:, 2*star.m:]
    compensation = np.linalg.lstsq(local, shared, rcond=1e-10)[0]
    profiled = shared-local@compensation
    scales = np.r_[np.maximum(.1*(boxes["r"][1]-boxes["r"][0]), 1e-4),
                   np.maximum(.1*(boxes["x"][1]-boxes["x"][0]), 1e-4)]
    _, singular, right = np.linalg.svd(profiled*scales[None, :], full_matrices=False)
    direction = scales*right[-1]
    direction = direction/max(np.max(abs(direction)), 1e-300)
    record["flat_direction"] = dict(shared_direction=direction.tolist(), scaled_shared_singular_values=singular.tolist(),
        local_nuisance_rank=int(np.linalg.matrix_rank(local)), compensation_rcond=1e-10, method="unconstrained_local_profile_SVD",
        ac_equation_residual=physical["ac_equation_residual"], active_bounds=active,
        interpretation="a local computational perturbation; not a confidence direction")
    if refit:
        for index, sign in enumerate((1., -1.)):
            delta = sign*perturb_scale_pu*direction
            state_delta = (-compensation@delta).reshape(history.t, -1)
            initial = dict(r=fitted.r+delta[:star.m], x=fitted.x+delta[star.m:],
                p=fitted.p+state_delta[:, :history.n], q=fitted.q+state_delta[:, history.n:2*history.n],
                source_p=fitted.source_p+(state_delta[:, 2*history.n] if estimate_source else 0),
                source_q=fitted.source_q+(state_delta[:, 2*history.n+1] if estimate_source else 0))
            initial_point = dict(r=initial["r"], x=initial["x"], p=initial["p"], q=initial["q"],
                source=np.c_[initial["source_p"], initial["source_q"]])
            clip_count = 0
            for name, value in initial_point.items():
                lo, hi = boxes[name]
                margin = 1e-10*np.maximum(1., hi-lo)
                corrected = np.clip(value, lo+margin, hi-margin)
                clip_count += int(np.count_nonzero(corrected != value))
                initial_point[name] = corrected
            initial.update(r=initial_point["r"], x=initial_point["x"], p=initial_point["p"], q=initial_point["q"],
                source_p=initial_point["source"][:, 0], source_q=initial_point["source"][:, 1])
            attempt = dict(index=index, sign=sign, initial=_json(initial), clipped_coordinate_count=clip_count,
                perturbation_scale_pu=perturb_scale_pu, retained=False, success=False)
            record["attempts"].append(attempt)
            original, calls = likelihood.ac_predict, [0]
            def counted(*args, **kwargs):
                calls[0] += 1
                return original(*args, **kwargs)
            try:
                likelihood.ac_predict = counted
                record["fit_calls"] += 1
                trial_options = {**options, "multistart": 1}
                alternative = likelihood.fit_candidate(star, history, history_model,
                    initial=initial, seed=int(seed)+100+index, **trial_options)
                attempt.update(fit=_json(alternative.summary()), success=bool(alternative.success and np.isfinite(alternative.loss)))
                if attempt["success"]:
                    independent_loss = _verify_fit_loss(star, alternative, history, history_model)
                    attempt["independent_ac_loss"] = independent_loss
                    boundary = _active(_point(alternative), boxes)
                    gap = .5*alternative.loss-best_half
                    difference = min(float(np.max(abs(np.r_[alternative.r-anchor.r, alternative.x-anchor.x]))) for anchor in anchors)
                    different = difference > 1e-5
                    supported = gap <= gate_half_sse
                    attempt.update(history_gap_half_sse=float(gap), different_shared_parameters=different,
                        closest_shared_parameter_distance_inf_pu=difference, active_bounds=boundary,
                        numerical_support_passed=bool(supported))
                    if different and supported and len(anchors) < max_anchors:
                        anchors.append(alternative)
                        attempt.update(retained=True, reason="distinct_supported_refit")
                    else:
                        attempt["reason"] = "duplicate_explanation" if not different else "outside_history_gate" if not supported else "anchor_budget"
                else:
                    attempt["reason"] = "optimization_failure"
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                attempt.update(success=False, reason="refit_exception", error=type(error).__name__+":"+str(error))
            finally:
                likelihood.ac_predict = original
                attempt["ac_calls"] = calls[0]
                record["ac_calls"] += calls[0]
    centers = []
    for index, anchor in enumerate(anchors):
        anchor_id = "history_star_"+str(index)
        record["anchor_records"].append(dict(id=anchor_id, half_loss=.5*anchor.loss,
            history_gap_half_sse=.5*anchor.loss-best_half, r=anchor.r.tolist(), x=anchor.x.tolist(),
            active_bounds=_active(_point(anchor), boxes)))
        future = FutureForecast(p=np.tile(np.mean(anchor.p, axis=0), (horizon, 1)),
            q=np.tile(np.mean(anchor.q, axis=0), (horizon, 1)),
            source_p=np.full(horizon, np.mean(anchor.source_p)), source_q=np.full(horizon, np.mean(anchor.source_q)),
            root_v=np.full(horizon, np.mean(history.root_v)))
        for support in TARGET_SUPPORTS:
            candidate = next((candidate for candidate in candidates if [tuple(s) for s in candidate.supports if 1 < len(s) < 3] == [support]), None)
            if candidate is None:
                raise ValueError("complete three-cherry repair set required")
            r, x, inserted = embed_star(star, anchor, candidate)
            centers.append(InformationCenter(id=anchor_id+"_"+"".join(map(str, inserted)),
                anchor_id=anchor_id, target_support=support, candidate=candidate, r=r.copy(), x=x.copy(),
                history_p=anchor.p.copy(), history_q=anchor.q.copy(), history_source_p=anchor.source_p.copy(),
                history_source_q=anchor.source_q.copy(), history_root_v=history.root_v.copy(), future=future))
    record.update(activated=True, reason="history_supported_star_contractions", anchor_count=len(anchors),
        center_count=len(centers), seconds=perf_counter()-began,
        distinct_explanation_threshold_inf_pu=1e-5, refit_attempt_count=len(record["attempts"]))
    return centers, record


build_centers = build_centres

__all__ = ["build_centres", "build_centers", "embed_star", "history_white_jacobian",
    "quality_models", "meter_to_record", "meter_from_record", "model_from_record",
    "draw_measurement_noise", "cheap_quality_plan"]