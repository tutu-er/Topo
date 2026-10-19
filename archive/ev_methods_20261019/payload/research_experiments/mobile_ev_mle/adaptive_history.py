"""History-only contexts and joint-control/parameter frozen AC surrogates.

For a topology/centre/scenario pair, r=[r_Hg;r_Hh;mu_Ng-mu_Nh]. Joint Taylor
expansion at (u0,z=0) gives r=c+B u+A z, with c_H retained and B_H=0. Control
u is time-major kW; physical EV P is tracking_gain*u/base_kva, once. Historical
R/X and states remain explicit variables, with their physical/declared local
boxes. Thus a boundary fit or nonzero historical gradient is not silently
replaced by delta_beta' H delta_beta. F is empty in the first implementation.

The history profile H and flat vectors are diagnostics/exploration geometry,
not confidence regions. Multiple centres are actual historical AC refits or
explicitly marked physically evaluated low-loss seeds, never new MLE claims.
All matrices/boxes are fixed within one proxy. Nonlinear back-substitution is
separate; these finite local domains do not cover the full AC likelihood.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, product
from time import perf_counter
from types import SimpleNamespace

import numpy as np

try:
    from .likelihood import fit_candidate
    from .symmetric_ac_local import (CHANNELS, mean_and_jacobian,
        historical_profile_geometry, whiten_observations)
except ImportError:
    from research_experiments.mobile_ev_mle.likelihood import fit_candidate
    from research_experiments.mobile_ev_mle.symmetric_ac_local import (CHANNELS,
        mean_and_jacobian, historical_profile_geometry, whiten_observations)


def _array(value, shape, name):
    try:
        out = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _bound(pair, name):
    value = np.asarray(pair, float)
    if value.shape != (2,) or not np.isfinite(value).all() or value[0] >= value[1]:
        raise ValueError(f"{name} requires finite lower < upper")
    return tuple(value)


def _control_array(value, t, n, name):
    out = np.asarray(value, float)
    if out.shape == (t*n,):
        out = out.reshape(t, n)
    return _array(out, (t, n), name)


def _state(fit):
    return np.c_[fit.p, fit.q, fit.source_p, fit.source_q]


def _hash_arrays(*arrays):
    digest = sha256()
    for value in arrays:
        out = np.asarray(value, dtype="<f8")
        digest.update(str(out.shape).encode()); digest.update(out.tobytes())
    return digest.hexdigest()


def _predict_path(values, future_count, mode):
    values = np.asarray(values, float)
    if mode == "mean":
        return np.broadcast_to(values.mean(axis=0), (future_count, *values.shape[1:])).copy()
    if mode != "ols_trend" or len(values) < 2:
        raise ValueError("forecast mode must be mean or ols_trend with at least two history slots")
    design = np.c_[np.ones(len(values)), np.arange(len(values))]
    coefficients = np.linalg.lstsq(design, values.reshape(len(values), -1), rcond=None)[0]
    future = np.c_[np.ones(future_count), np.arange(len(values), len(values)+future_count)]@coefficients
    return future.reshape(future_count, *values.shape[1:])


def _forecast_specs(forecast_support):
    if forecast_support is None:
        return [dict(name="mean", baseline="mean", source_baseline="mean", tracking_gain=1.,
                     source_shift_p=0., source_shift_q=0., provenance="historical_mean_and_declared_unit_tracking")]
    specs = forecast_support.get("scenarios") if isinstance(forecast_support, dict) else forecast_support
    if specs is None or not len(specs):
        raise ValueError("forecast_support must contain explicit nonempty scenario specifications")
    result = []
    for index, given in enumerate(specs):
        spec = dict(name=f"scenario_{index}", baseline="mean", source_baseline="mean",
                    tracking_gain=1., source_shift_p=0., source_shift_q=0.,
                    provenance="declared_history_only_forecast")
        spec.update(given)
        allowed = {"name", "baseline", "source_baseline", "tracking_gain", "source_shift_p", "source_shift_q", "provenance"}
        if any(key not in allowed for key in spec):
            raise ValueError("forecast specifications may not contain evaluation truth or future noise")
        if spec["baseline"] not in ("mean", "ols_trend") or spec["source_baseline"] not in ("mean", "ols_trend"):
            raise ValueError("unsupported history-only forecast method")
        if not np.isfinite(spec["tracking_gain"]) or spec["tracking_gain"] <= 0:
            raise ValueError("tracking_gain must be finite and strictly positive")
        for key in ("source_shift_p", "source_shift_q"):
            if not np.isfinite(spec[key]):
                raise ValueError("declared source shifts must be finite")
        result.append(spec)
    if len({s["name"] for s in result}) != len(result):
        raise ValueError("scenario names must be unique")
    return result


def _center(candidate, fitted, history, model, channels, bounds, flat_scale_floor):
    if (fitted.p.shape != history.p.shape or fitted.q.shape != history.q.shape
            or np.asarray(fitted.source_p).shape != (history.t,)
            or np.asarray(fitted.source_q).shape != (history.t,)):
        raise ValueError("a supplied historical fit must contain H only; longer H+future fits are forbidden")
    geometry = mean_and_jacobian(candidate, fitted.r, fitted.x, fitted.p, fitted.q,
        fitted.source_p, fitted.source_q, history.root_v, model, channels=channels)
    voltage = abs(geometry["physical"]["voltage"])
    if voltage.min() < bounds["voltage"][0] or voltage.max() > bounds["voltage"][1]:
        raise ValueError("historical expansion point violates voltage physical domain")
    white_y = whiten_observations(history, model, channels)
    residual = geometry["mean"]-white_y
    beta, state = np.r_[fitted.r, fitted.x], _state(fitted)
    beta_lo = np.r_[np.full(candidate.m, bounds["r"][0]), np.full(candidate.m, bounds["x"][0])]
    beta_hi = np.r_[np.full(candidate.m, bounds["r"][1]), np.full(candidate.m, bounds["x"][1])]
    state_lo = np.tile(np.r_[np.full(candidate.n, bounds["p"][0]), np.full(candidate.n, bounds["q"][0]),
                              np.full(2, bounds["source"][0])], (history.t, 1))
    state_hi = np.tile(np.r_[np.full(candidate.n, bounds["p"][1]), np.full(candidate.n, bounds["q"][1]),
                              np.full(2, bounds["source"][1])], (history.t, 1))
    if (np.any(beta < beta_lo-1e-12) or np.any(beta > beta_hi+1e-12)
            or np.any(state < state_lo-1e-12) or np.any(state > state_hi+1e-12)):
        raise ValueError("historical fitted state or line violates the declared physical box")
    profile = historical_profile_geometry(geometry["line_jacobian"], geometry["state_jacobian"])
    coordinate_scale = np.maximum(beta, flat_scale_floor)
    scaled = coordinate_scale[:, None]*profile["precision"]*coordinate_scale[None, :]
    eigenvalues, vectors = np.linalg.eigh((scaled+scaled.T)/2)
    directions = (coordinate_scale[:, None]*vectors).T
    flat_compensation = (profile["state_compensation"]@directions.T).T.reshape(len(directions), history.t, 2*history.n+2)
    values = np.r_[beta, state.ravel()]
    lo, hi = np.r_[beta_lo, state_lo.ravel()], np.r_[beta_hi, state_hi.ravel()]
    # These are reconstructed active faces, with the tolerance recorded. The
    # original solver's status/optimality is preserved in the attempt ledger.
    face_tolerance = 1e-7*np.maximum(hi-lo, 1e-4)
    active = np.where(values-lo <= face_tolerance, -1, np.where(hi-values <= face_tolerance, 1, 0))
    gradient = 2*np.c_[geometry["line_jacobian"], geometry["state_jacobian"]].T@residual
    violation = np.where(active == -1, np.minimum(gradient, 0.),
                         np.where(active == 1, np.maximum(gradient, 0.), gradient))
    return dict(candidate=candidate, fit=fitted, beta=beta, history_state=state,
        geometry=geometry, history_residual=residual, history_loss=float(residual@residual),
        profile=profile, flat_eigenvalues=eigenvalues, flat_directions=directions,
        flat_source_compensation=flat_compensation[:, :, 2*history.n:],
        flat_coordinate_scale=coordinate_scale, active_faces=active,
        reconstructed_kkt_violation=float(np.max(abs(violation), initial=0.)),
        historical_gradient=gradient, active_face_tolerance=face_tolerance,
        beta_physical_lower=beta_lo, beta_physical_upper=beta_hi,
        state_physical_lower=state_lo, state_physical_upper=state_hi)


def build_history_context(history, candidates, model, forecast_support=None, *, horizon=8,
                          base_kva=20., historical_fits=None, max_centers=2,
                          max_attempts_per_candidate=3, center_loss_slack=12.,
                          retain_feasible_flat_seeds=True,
                          flat_search_fraction=.5, flat_scale_floor=.002,
                          center_separation=.02, line_trust_fraction=.30,
                          source_delta_bound=.02, history_state_radius=None,
                          r_bounds=(1e-6, .1), x_bounds=(1e-6, .1),
                          p_bounds=(-1., 1.), q_bounds=(-1., 1.), source_bounds=(-.5, .5),
                          voltage_limits=(.85, 1.15), known_history_ev_kw=None,
                          q_over_p=0., multistart=2, max_nfev=150, seed=0,
                          passive_metadata=None):
    """Build actual H-only fit centres and declared H-only predictive scenarios.

    max_centers bounds a finite explored family; loss_slack is a descriptive
    inclusion rule, not confidence coverage. Default history-state step boxes
    retain the original physical bounds. An optional nonnegative radius is
    another explicitly declared local domain, never a meter error guarantee.
    known_history_ev_kw is known executed terminal EV input, not an uncertain
    command. Command/realized-input discrepancies require a declared scenario.
    """
    started = perf_counter()
    candidates = list(candidates)
    if len(candidates) < 2 or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("at least two unique candidates required")
    if model.fit_bias:
        raise ValueError("stage A requires independently calibrated known fixed bias")
    if history.master_p is None or history.master_q is None:
        raise ValueError("stage A requires master P/Q at the known root")
    if (int(horizon) != horizon or horizon < 1 or not np.isfinite(base_kva) or base_kva <= 0
            or int(max_centers) != max_centers or max_centers < 1
            or int(max_attempts_per_candidate) != max_attempts_per_candidate or max_attempts_per_candidate < 1
            or not np.isfinite(center_loss_slack) or center_loss_slack < 0
            or not 0 <= line_trust_fraction < 1 or not np.isfinite(source_delta_bound) or source_delta_bound < 0
            or not np.isfinite(flat_search_fraction) or flat_search_fraction < 0
            or not np.isfinite(flat_scale_floor) or flat_scale_floor <= 0
            or not np.isfinite(center_separation) or center_separation < 0
            or not np.isfinite(q_over_p)):
        raise ValueError("invalid context budget/domain/unit specification")
    channels = CHANNELS + tuple(key for key in ("source_p", "source_q") if getattr(history, key) is not None)
    bounds = {name:_bound(value, name) for name, value in
              (("r", r_bounds), ("x", x_bounds), ("p", p_bounds), ("q", q_bounds),
               ("source", source_bounds), ("voltage", voltage_limits))}
    specs = _forecast_specs(forecast_support)
    baseline_p = history.p-_array(model.bias_p, history.p.shape, "bias_p")
    baseline_q = history.q-_array(model.bias_q, history.q.shape, "bias_q")
    if known_history_ev_kw is not None:
        previous_control = _array(known_history_ev_kw, history.p.shape, "known_history_ev_kw")/base_kva
        baseline_p -= previous_control
        baseline_q -= q_over_p*previous_control
    scenarios = []
    for spec in specs:
        scenarios.append(dict(spec, p0=_predict_path(baseline_p, horizon, spec["baseline"]),
            q0=_predict_path(baseline_q, horizon, spec["baseline"]),
            root_v=np.full(horizon, np.mean(history.root_v))))
    attempts, centres, failures = [], {}, []
    supplied = dict(historical_fits or {})
    for candidate_index, candidate in enumerate(candidates):
        valid = []
        attempts_for_candidate = 0

        def attempt(initial, kind, direction=None):
            nonlocal attempts_for_candidate
            number = attempts_for_candidate
            attempts_for_candidate += 1
            row = dict(candidate=candidate.name, attempt=number, kind=kind,
                       seed=seed+1000*candidate_index+number, retained=False,
                       centre_kind="optimizer_fit", optimized=True)
            if direction is not None:
                row["flat_seed_beta_step"] = direction.tolist()
            try:
                fitted = (supplied[candidate.name] if kind == "supplied_history_fit" else
                    fit_candidate(candidate, history, model, initial=initial, multistart=multistart,
                        seed=row["seed"], max_nfev=max_nfev, r_bounds=bounds["r"], x_bounds=bounds["x"],
                        p_bounds=bounds["p"], q_bounds=bounds["q"], source_bounds=bounds["source"],
                        voltage_bounds=bounds["voltage"]))
                row.update(success=bool(fitted.success), loss=float(fitted.loss) if np.isfinite(fitted.loss) else None,
                    status=int(fitted.status), message=str(fitted.message), nfev=int(fitted.nfev),
                    optimality=float(fitted.optimality), diagnostics=fitted.diagnostics)
                if fitted.success and np.isfinite(fitted.loss):
                    centre = _center(candidate, fitted, history, model, channels, bounds, flat_scale_floor)
                    if abs(centre["history_loss"]-fitted.loss) > 1e-6*max(1., abs(fitted.loss)):
                        raise ValueError("supplied fit loss does not match these historical meters")
                    centre["attempt"] = len(attempts)
                    centre["centre_kind"] = "optimizer_fit"
                    centre["optimized"] = True
                    valid.append(centre)
                    row["actual_recomputed_history_loss"] = centre["history_loss"]
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                row.update(success=False, error=str(error), failure_kind="history_computation_or_domain")
            attempts.append(row)

        attempt(None, "supplied_history_fit" if candidate.name in supplied else "initial_history_fit")
        if valid and flat_search_fraction:
            initial_center = min(valid, key=lambda c:c["history_loss"])
            for number in range(max_attempts_per_candidate-1):
                direction_number = (number//2) % len(initial_center["flat_directions"])
                sign = -1. if number % 2 == 0 else 1.
                proposed_beta = np.clip(initial_center["beta"] + sign*flat_search_fraction*
                    initial_center["flat_directions"][direction_number],
                    initial_center["beta_physical_lower"], initial_center["beta_physical_upper"])
                step = proposed_beta-initial_center["beta"]
                proposed_state = initial_center["history_state"] + (
                    initial_center["profile"]["state_compensation"]@step).reshape(initial_center["history_state"].shape)
                proposed_state = np.clip(proposed_state, initial_center["state_physical_lower"], initial_center["state_physical_upper"])
                initial = dict(r=proposed_beta[:candidate.m], x=proposed_beta[candidate.m:],
                    p=proposed_state[:, :history.n], q=proposed_state[:, history.n:2*history.n],
                    source_p=proposed_state[:, 2*history.n], source_q=proposed_state[:, 2*history.n+1])
                if retain_feasible_flat_seeds and max_centers > 1:
                    seed_row = dict(candidate=candidate.name, attempt=attempts_for_candidate,
                        kind="flat_seed_actual_AC_evaluation", centre_kind="feasible_low_loss_seed",
                        optimized=False, retained=False, nfev=0, flat_seed_beta_step=step.tolist())
                    try:
                        actual_seed = mean_and_jacobian(candidate, initial["r"], initial["x"], initial["p"], initial["q"],
                            initial["source_p"], initial["source_q"], history.root_v, model, channels=channels)
                        seed_residual = actual_seed["mean"]-whiten_observations(history, model, channels)
                        seed_loss = float(seed_residual@seed_residual)
                        seed_fit = SimpleNamespace(**initial, success=True, loss=seed_loss, status=0,
                            message="physically evaluated historical seed; not an optimizer/MLE result", nfev=0,
                            optimality=np.nan, diagnostics=dict(centre_kind="feasible_low_loss_seed", optimized=False))
                        seed_centre = _center(candidate, seed_fit, history, model, channels, bounds, flat_scale_floor)
                        seed_centre.update(attempt=len(attempts), centre_kind="feasible_low_loss_seed", optimized=False)
                        valid.append(seed_centre)
                        seed_row.update(success=True, loss=seed_loss, actual_recomputed_history_loss=seed_loss,
                            ac_equation_residual=actual_seed["physical"]["ac_equation_residual"],
                            power_balance_residual=actual_seed["physical"]["power_balance_residual"])
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                        seed_row.update(success=False, error=str(error), failure_kind="seed_physical_or_computation")
                    attempts.append(seed_row)
                attempt(initial, "flat_direction_actual_refit", step)
        ordered = sorted(valid, key=lambda c:(not c["optimized"], c["history_loss"]))
        best_actual_loss = min((c["history_loss"] for c in valid), default=np.inf)
        kept = []
        for centre in ordered:
            if centre["history_loss"] > best_actual_loss+center_loss_slack:
                continue
            if any(np.linalg.norm((centre["beta"]-old["beta"])/old["flat_coordinate_scale"]) < center_separation for old in kept):
                continue
            centre["id"] = f"{candidate.name}_centre_{len(kept)}"
            attempts[centre["attempt"]]["retained"] = True
            kept.append(centre)
            if len(kept) == max_centers:
                break
        centres[candidate.name] = kept
        if not kept:
            failures.append(dict(candidate=candidate.name, reason="no_valid_history_centre"))
    config = dict(base_kva=float(base_kva), horizon=int(horizon), line_trust_fraction=float(line_trust_fraction),
        source_delta_bound=float(source_delta_bound), history_state_radius=history_state_radius,
        q_over_p=float(q_over_p), max_centers=int(max_centers), max_attempts_per_candidate=int(max_attempts_per_candidate),
        center_loss_slack=float(center_loss_slack), center_separation=float(center_separation),
        retain_feasible_flat_seeds=bool(retain_feasible_flat_seeds),
        bounds=bounds, source_node=1, historical_state_order="p[0:n],q[0:n],sourceP,sourceQ")
    audit = dict(wzzt_default_source_variables_active=False,
        reason="candidates.propose does not pass source nominal; None reverts to ordinary WZZT",
        wzzt_is_independent_extra_history_evidence=False, same_historical_meters_used=True,
        source_location_shared_node1=True, passive_metadata_supplied=passive_metadata is not None)
    summary = dict(status="history_fit_failure" if failures else "ready", config=config,
        centre_counts={name:len(values) for name, values in centres.items()}, attempts=attempts, failures=failures,
        history_hash=_hash_arrays(*[getattr(history, k) for k in (*channels, "root_v")]),
        scenario_specs=specs, forecast_is_coverage_guarantee=False,
        known_history_ev_semantics="known actual executed EV input; uncertain commands cannot be exactly subtracted",
        known_history_ev_subtracted_from_forecast_only=known_history_ev_kw is not None,
        constant_prediction_time_symmetry=all(s["baseline"] == "mean" and s["source_baseline"] == "mean" for s in specs),
        time_symmetry_conditions="constant forecasts, time-invariant meter covariance and physical domains; operational feasibility can break symmetry",
        wzzt_audit=audit, seconds=perf_counter()-started,
        centres={name:[dict(id=c["id"], history_loss=c["history_loss"],
            centre_kind=c["centre_kind"], optimized=c["optimized"],
            beta=c["beta"].tolist(), profile_precision=c["profile"]["precision"].tolist(),
            flat_eigenvalues=c["flat_eigenvalues"].tolist(), flat_directions=c["flat_directions"].tolist(),
            flat_source_compensation=c["flat_source_compensation"].tolist(),
            compensation=c["profile"]["state_compensation"].tolist(), active_faces=c["active_faces"].tolist(),
            reconstructed_kkt_violation=c["reconstructed_kkt_violation"],
            profile_is_confidence_region=False) for c in values] for name, values in centres.items()})
    return dict(history=history, candidates=candidates, model=model, channels=channels,
        scenarios=scenarios, centres=centres, config=config, summary=summary,
        status=summary["status"], attempts=attempts)


def _future_for_center(context, centre, scenario, control_kw):
    config, candidate = context["config"], centre["candidate"]
    gain = scenario["tracking_gain"]
    p = scenario["p0"]+gain*control_kw/config["base_kva"]
    q = scenario["q0"]+config["q_over_p"]*gain*control_kw/config["base_kva"]
    sp = _predict_path(centre["fit"].source_p, config["horizon"], scenario["source_baseline"])+scenario["source_shift_p"]
    sq = _predict_path(centre["fit"].source_q, config["horizon"], scenario["source_baseline"])+scenario["source_shift_q"]
    geometry = mean_and_jacobian(candidate, centre["fit"].r, centre["fit"].x, p, q, sp, sq,
        scenario["root_v"], context["model"], channels=context["channels"])
    physical = geometry["physical"]
    for key, bound in (("p", "p"), ("q", "q"), ("source_p", "source"), ("source_q", "source")):
        if np.min(physical[key]) < config["bounds"][bound][0]-1e-12 or np.max(physical[key]) > config["bounds"][bound][1]+1e-12:
            raise ValueError(f"required forecast/expansion violates {key} physical domain")
    voltage = abs(physical["voltage"])
    if voltage.min() < config["bounds"]["voltage"][0] or voltage.max() > config["bounds"]["voltage"][1]:
        raise ValueError("required forecast/expansion violates voltage domain")
    local = 2*candidate.n+2
    columns_p = (np.arange(len(p))[:, None]*local+np.arange(candidate.n)).ravel()
    columns_q = columns_p+candidate.n
    control = (geometry["state_jacobian"][:, columns_p]+config["q_over_p"]*
               geometry["state_jacobian"][:, columns_q])*gain/config["base_kva"]
    return dict(geometry=geometry, p=p, q=q, source_p=sp, source_q=sq, control_jacobian=control)


def build_frozen_pairs(context, u0=None, *, source_mode="local_refit", line_trust_fraction=None,
                       source_delta_bound=None):
    """Return dictionaries for FrozenPair(**pair), all matrices fixed at u0.

    Every retained cross-topology centre combination and every declared common
    scenario is included. A failed required expansion raises, never drops a
    difficult pair. No complete EV action library is generated or traversed.
    """
    if context["status"] != "ready":
        raise ValueError("cannot freeze a context with failed historical candidates")
    if source_mode not in ("fixed", "local_refit"):
        raise ValueError("source_mode must be fixed or local_refit")
    cfg = context["config"]
    trust = cfg["line_trust_fraction"] if line_trust_fraction is None else float(line_trust_fraction)
    source_radius = cfg["source_delta_bound"] if source_delta_bound is None else float(source_delta_bound)
    if not 0 <= trust < 1 or source_radius < 0:
        raise ValueError("invalid declared local line/source radius")
    n, t = context["history"].n, cfg["horizon"]
    control = np.zeros((t, n)) if u0 is None else _control_array(u0, t, n, "u0_kW")
    if np.any(control < 0):
        raise ValueError("first stage is charging only; control must be nonnegative kW")
    pairs = []
    for left_name, right_name in combinations(context["centres"], 2):
        for left_index, right_index, scenario_index in product(range(len(context["centres"][left_name])),
                range(len(context["centres"][right_name])), range(len(context["scenarios"]))):
            centres = (context["centres"][left_name][left_index], context["centres"][right_name][right_index])
            scenario = context["scenarios"][scenario_index]
            futures = [_future_for_center(context, c, scenario, control) for c in centres]
            history_widths = [len(c["history_residual"]) for c in centres]
            future_width = len(futures[0]["geometry"]["mean"])
            residual_count = sum(history_widths)+future_width
            layout, lo, hi, scale = [], [], [], []
            variable_count = 0
            for centre, future in zip(centres, futures):
                beta = centre["beta"]
                beta_lower = np.maximum(centre["beta_physical_lower"]-beta, -trust*beta)
                beta_upper = np.minimum(centre["beta_physical_upper"]-beta, trust*beta)
                state = centre["history_state"]
                state_lower = centre["state_physical_lower"]-state
                state_upper = centre["state_physical_upper"]-state
                if cfg["history_state_radius"] is not None:
                    radius = _array(cfg["history_state_radius"], state.shape, "history_state_radius")
                    if np.any(radius < 0):
                        raise ValueError("history_state_radius must be nonnegative")
                    state_lower = np.maximum(state_lower, -radius)
                    state_upper = np.minimum(state_upper, radius)
                sizes = (len(beta), state.size, 2*t if source_mode == "local_refit" else 0)
                slices = []
                for size in sizes:
                    slices.append((variable_count, variable_count+size)); variable_count += size
                layout.append(dict(beta=slices[0], history_state=slices[1], future_source=slices[2]))
                lo.extend(beta_lower); hi.extend(beta_upper); scale.extend(np.maximum(beta, 1e-4))
                lo.extend(state_lower.ravel()); hi.extend(state_upper.ravel())
                sigma_p = _array(context["model"].p_sigma, (context["history"].t, n), "p_sigma")
                sigma_q = _array(context["model"].q_sigma, sigma_p.shape, "q_sigma")
                scale.extend(np.c_[sigma_p, sigma_q, np.full((context["history"].t, 2), .02)].ravel())
                if sizes[2]:
                    source = np.c_[future["source_p"], future["source_q"]].ravel()
                    lo.extend(np.maximum(-source_radius, cfg["bounds"]["source"][0]-source))
                    hi.extend(np.minimum(source_radius, cfg["bounds"]["source"][1]-source))
                    scale.extend(np.full(2*t, max(source_radius, 1e-4)))
            c = np.r_[centres[0]["history_residual"], centres[1]["history_residual"],
                      futures[0]["geometry"]["mean"]-futures[1]["geometry"]["mean"]]
            b = np.zeros((residual_count, t*n)); a = np.zeros((residual_count, variable_count))
            future_rows = slice(sum(history_widths), residual_count)
            b[future_rows] = futures[0]["control_jacobian"]-futures[1]["control_jacobian"]
            c -= b@control.ravel()
            row_start = 0
            for sign, centre, future, sizes, width in zip((1., -1.), centres, futures, layout, history_widths):
                hs = slice(row_start, row_start+width); row_start += width
                beta_slice, state_slice, source_slice = [slice(*sizes[key]) for key in ("beta", "history_state", "future_source")]
                a[hs, beta_slice] = centre["geometry"]["line_jacobian"]
                a[hs, state_slice] = centre["geometry"]["state_jacobian"]
                a[future_rows, beta_slice] = sign*future["geometry"]["line_jacobian"]
                if source_slice.stop > source_slice.start:
                    ids = (np.arange(t)[:, None]*(2*n+2)+np.array([2*n, 2*n+1])).ravel()
                    a[future_rows, source_slice] = sign*future["geometry"]["state_jacobian"][:, ids]
            lo, hi, scale = np.asarray(lo), np.asarray(hi), np.asarray(scale)
            if np.any(lo > hi) or not np.isfinite(a).all() or not np.isfinite(c).all():
                raise ValueError("invalid required frozen pair; it may not be discarded")
            free = np.flatnonzero(hi-lo > 1e-15)
            fixed = np.flatnonzero(hi-lo <= 1e-15)
            fixed_values = np.zeros(variable_count)
            fixed_values[fixed] = (lo[fixed]+hi[fixed])/2
            c += a@fixed_values
            name = f"{centres[0]['id']}__{centres[1]['id']}__{scenario['name']}__{source_mode}"
            metadata = dict(candidate_names=[left_name, right_name], centre_indices=[left_index, right_index],
                centre_ids=[c0["id"] for c0 in centres], scenario_index=scenario_index,
                centre_kinds=[c0["centre_kind"] for c0 in centres],
                centre_optimized=[c0["optimized"] for c0 in centres],
                scenario_name=scenario["name"], scenario_provenance=scenario["provenance"],
                u0_kw=control.tolist(), control_order="time-major", control_unit="kW",
                base_kva=cfg["base_kva"], tracking_gain=scenario["tracking_gain"], source_node=1,
                source_mode=source_mode, variable_layout=layout, full_variable_count=variable_count,
                free_indices=free.tolist(), fixed_values=fixed_values.tolist(),
                history_row_widths=history_widths, future_row_start=sum(history_widths),
                line_trust_fraction=trust, source_delta_bound=source_radius,
                historical_center_cost=sum(c0["history_loss"] for c0 in centres),
                objective="two historical fit costs plus future mean discrepancy",
                matrices_depend_on_current_u=False, bounds_independent_of_u=True,
                scenario_coupling="common baseline/tracking; candidate-specific H-only source forecast",
                nonlinear_separation_certificate=False, history_scope="H_only",
                historical_inputs_hash=context["summary"]["history_hash"])
            pairs.append(dict(name=name, c=c, B=b, A=a[:, free], F=np.zeros((0, len(free))),
                lower=lo[free], upper=hi[free], scale=scale[free], metadata=metadata))
    return pairs


def evaluate_frozen_pair_actual(context, pair, u_kw, z):
    """Back-substitute one proxy witness in original AC; no future readings."""
    cfg, md = context["config"], pair["metadata"]
    control = _control_array(u_kw, cfg["horizon"], context["history"].n, "u_kw")
    if np.any(control < 0):
        raise ValueError("charging-only stage cannot evaluate negative EV control")
    reduced = np.asarray(z, float)
    if reduced.shape != pair["lower"].shape or not np.isfinite(reduced).all():
        raise ValueError("proxy witness has wrong shape or is nonfinite")
    if np.any(reduced < pair["lower"]-1e-10) or np.any(reduced > pair["upper"]+1e-10):
        raise ValueError("proxy witness violates its fixed parameter domain")
    full = np.asarray(md["fixed_values"]).copy()
    full[md["free_indices"]] = reduced
    scenario = context["scenarios"][md["scenario_index"]]
    histories, futures, physics = [], [], []
    observed = whiten_observations(context["history"], context["model"], context["channels"])
    for name, index, layout in zip(md["candidate_names"], md["centre_indices"], md["variable_layout"]):
        centre = context["centres"][name][index]
        candidate = centre["candidate"]
        beta = centre["beta"]+full[slice(*layout["beta"])]
        state = centre["history_state"]+full[slice(*layout["history_state"])].reshape(centre["history_state"].shape)
        historic = mean_and_jacobian(candidate, beta[:candidate.m], beta[candidate.m:], state[:, :candidate.n],
            state[:, candidate.n:2*candidate.n], state[:, 2*candidate.n], state[:, 2*candidate.n+1],
            context["history"].root_v, context["model"], channels=context["channels"], with_jacobian=False)
        nominal = _future_for_center(context, centre, scenario, control)
        source_delta = full[slice(*layout["future_source"])]
        source_delta = source_delta.reshape(cfg["horizon"], 2) if len(source_delta) else np.zeros((cfg["horizon"], 2))
        future = mean_and_jacobian(candidate, beta[:candidate.m], beta[candidate.m:], nominal["p"], nominal["q"],
            nominal["source_p"]+source_delta[:, 0], nominal["source_q"]+source_delta[:, 1], scenario["root_v"],
            context["model"], channels=context["channels"], with_jacobian=False)
        histories.append(historic["mean"]-observed); futures.append(future["mean"])
        physics.extend((historic["physical"], future["physical"]))
    actual = np.r_[*histories, futures[0]-futures[1]]
    affine = pair["c"]+pair["B"]@control.ravel()+pair["A"]@reduced
    physical_ok = True
    for item in physics:
        volts = abs(item["voltage"])
        physical_ok &= bool(volts.min() >= cfg["bounds"]["voltage"][0] and volts.max() <= cfg["bounds"]["voltage"][1])
        for key, bound in (("p", "p"), ("q", "q"), ("source_p", "source"), ("source_q", "source")):
            physical_ok &= bool(item[key].min() >= cfg["bounds"][bound][0]-1e-12 and item[key].max() <= cfg["bounds"][bound][1]+1e-12)
    return dict(actual_residual=actual, affine_residual=affine,
        actual_soft_cost=float(actual@actual), affine_soft_cost=float(affine@affine),
        actual_history_costs=[float(r@r) for r in histories],
        actual_future_distance_squared=float(np.sum((futures[0]-futures[1])**2)),
        joint_taylor_error_norm=float(np.linalg.norm(actual-affine)),
        physically_feasible=physical_ok,
        max_ac_equation_residual=max(p["ac_equation_residual"] for p in physics),
        max_power_balance_residual=max(p["power_balance_residual"] for p in physics),
        nonlinear_infimum_computed=False)
