"""Bounded local symmetric AC design with a soft historical GN penalty.

The selector uses historical meters and public actions only. Future terminal
totals are a common calibrated historical forecast plus EV once. Source
forecasts remain at node1, fixed or perturbed in a declared local box. This is
a convex linearized inner QP, not a nonlinear distance/separation certificate.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter
from typing import Mapping

import numpy as np
from scipy.optimize import lsq_linear

try:
    from .likelihood import ac_predict, fit_candidate, _response_derivatives
    from .design import check_action
except ImportError:
    from research_experiments.mobile_ev_mle.likelihood import ac_predict, fit_candidate, _response_derivatives
    from research_experiments.mobile_ev_mle.design import check_action


CHANNELS = ("p", "q", "v", "master_p", "master_q")


def _broadcast(value, shape, name):
    try:
        array = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _channels(history):
    if history.master_p is None or history.master_q is None:
        raise ValueError("local symmetric AC design requires master P/Q anchors")
    return CHANNELS + tuple(name for name in ("source_p", "source_q") if getattr(history, name) is not None)


def whiten_observations(observations, model, channels=CHANNELS):
    blocks = []
    for name in channels:
        value = getattr(observations, name)
        if value is None:
            raise ValueError(f"missing {name} observation")
        sigma = _broadcast(getattr(model, name+"_sigma"), np.asarray(value).shape, name+"_sigma")
        if np.any(sigma <= 0):
            raise ValueError("all included meter standard deviations must be positive")
        blocks.append((np.asarray(value)/sigma).ravel())
    return np.concatenate(blocks)


def baseline_from_history(history, model, n_slots):
    """Common terminal forecast from calibrated meters, never a model's truth."""
    if model.fit_bias:
        raise ValueError("this local stage requires known independently calibrated biases")
    if n_slots < 1:
        raise ValueError("future horizon must be positive")
    p = history.p - _broadcast(model.bias_p, history.p.shape, "bias_p")
    q = history.q - _broadcast(model.bias_q, history.q.shape, "bias_q")
    return dict(p=np.tile(p.mean(axis=0), (n_slots, 1)),
                q=np.tile(q.mean(axis=0), (n_slots, 1)),
                root_v=np.full(n_slots, np.mean(history.root_v)))


def mean_and_jacobian(candidate, r, x, p, q, source_p, source_q, root_v,
                      model, *, channels=CHANNELS, with_jacobian=True):
    """Whiten full AC meter means and analytic line/local-state Jacobians.

    Columns are beta=(all R,all X) and w=(p,q,sP,sQ) per time. State columns
    are arranged time by time. Channel blocks include terminal P/Q as well as
    voltage and master P/Q. Known biases enter means but have no free columns.
    """
    physical = ac_predict(candidate, r, x, p, q, source_p, source_q, root_v)
    t, n, m = len(physical["p"]), candidate.n, candidate.m
    local = 2*n+2
    derivatives = _response_derivatives(candidate, physical, True) if with_jacobian else None
    means, lines, states = [], [], []
    for name in channels:
        value = np.asarray(physical[name], float)
        if name in ("p", "q", "v"):
            value = value + _broadcast(getattr(model, "bias_"+name), value.shape, "bias_"+name)
        sigma = _broadcast(getattr(model, name+"_sigma"), value.shape, name+"_sigma")
        if np.any(sigma <= 0):
            raise ValueError("all included meter standard deviations must be positive")
        width = n if name in ("p", "q", "v") else 1
        means.append((value/sigma).ravel())
        if with_jacobian:
            direct = np.zeros((t, width, 2*m+local))
            if name in ("p", "q"):
                indices = np.arange(n)+(0 if name == "p" else n)
                direct[:, np.arange(n), 2*m+indices] = 1.
            elif name in ("source_p", "source_q"):
                direct[:, 0, 2*m+2*n+(name == "source_q")] = 1.
            else:
                direct = derivatives[name].reshape(t, width, 2*m+local).copy()
            direct /= sigma.reshape(t, width, 1)
            lines.append(direct[:, :, :2*m].reshape(t*width, 2*m))
            block = np.zeros((t*width, t*local))
            for time in range(t):
                block[time*width:(time+1)*width, time*local:(time+1)*local] = direct[time, :, 2*m:]
            states.append(block)
    result = dict(mean=np.concatenate(means), physical=physical)
    if with_jacobian:
        result.update(line_jacobian=np.vstack(lines), state_jacobian=np.vstack(states))
    return result


def historical_profile_geometry(line_jacobian, state_jacobian, *, rcond=1e-10):
    """GN line geometry after permitting historical local-state compensation.

    Rank-revealing SVD projects only the historical state columns. These ranks
    are diagnostics, never an all-direction singular-value design objective.
    """
    jb, jw = np.asarray(line_jacobian, float), np.asarray(state_jacobian, float)
    if (jb.ndim != 2 or jw.ndim != 2 or jb.shape[0] != jw.shape[0]
            or not np.isfinite(jb).all() or not np.isfinite(jw).all()):
        raise ValueError("historical Jacobians must be finite with matching rows")
    u, singular, vt = np.linalg.svd(jw, full_matrices=False)
    rank = int(np.count_nonzero(singular > rcond*singular[0])) if len(singular) and singular[0] else 0
    q = u[:, :rank]
    projected = jb-q@(q.T@jb)
    factor = np.linalg.qr(projected, mode="reduced")[1]
    compensation = -(vt[:rank].T/singular[:rank])@(q.T@jb) if rank else np.zeros((jw.shape[1], jb.shape[1]))
    return dict(precision=projected.T@projected, factor=factor,
                state_compensation=compensation, state_rank=rank,
                projected_line_rank=int(np.linalg.matrix_rank(projected, tol=rcond*max(np.linalg.norm(jb, 2), 1.))),
                projection_rcond=rcond)


def solve_bounded_soft_qp(d, A, history_factor, lower, upper, *, scale=None,
                          tol=1e-10, max_iter=300):
    """Globally solve the declared convex linearized box-constrained inner QP."""
    d, a, factor = np.asarray(d, float), np.asarray(A, float), np.asarray(history_factor, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (d.ndim != 1 or a.ndim != 2 or a.shape[0] != len(d)
            or factor.ndim != 2 or factor.shape[1] != a.shape[1]
            or lower.shape != (a.shape[1],) or upper.shape != lower.shape
            or not all(np.isfinite(v).all() for v in (d, a, factor, lower, upper))
            or np.any(lower >= upper)):
        raise ValueError("bounded soft-QP dimensions/bounds must be finite and consistent")
    scaling = np.ones(len(lower)) if scale is None else np.asarray(scale, float)
    if scaling.shape != lower.shape or not np.isfinite(scaling).all() or np.any(scaling <= 0):
        raise ValueError("positive finite coordinate scales required")
    augmented = np.vstack((a, factor))
    target = -np.r_[d, np.zeros(len(factor))]
    fitted = lsq_linear(augmented*scaling, target,
                       bounds=(lower/scaling, upper/scaling), method="bvls",
                       lsq_solver="exact", tol=tol, max_iter=max_iter)
    step = fitted.x*scaling
    future_residual, history_residual = d+a@step, factor@step
    residual = np.r_[future_residual, history_residual]
    objective = float(residual@residual)
    gradient = 2*augmented.T@residual
    # A tangent of this convex quadratic is a lower affine minorant. Minimize
    # that tangent over the same finite box; this is a numeric QP diagnostic,
    # with no transfer of its bound to the nonlinear AC design problem.
    tangent_corner = np.where(gradient >= 0, lower, upper)
    tangent_bound = objective+float(gradient@(tangent_corner-step))
    lower_bound = min(objective, max(0., tangent_bound))
    success = bool(fitted.success and np.isfinite(residual).all()
                   and np.all(step >= lower-1e-12) and np.all(step <= upper+1e-12))
    return dict(step=step, proxy_score=objective,
                proxy_future_distance_squared=float(future_residual@future_residual),
                proxy_history_penalty=float(history_residual@history_residual),
                success=success, status=int(fitted.status), message=str(fitted.message),
                optimality=float(fitted.optimality), iterations=int(fitted.nit),
                qp_tangent_lower_bound_numeric=lower_bound,
                qp_gap_numeric=max(0., objective-lower_bound),
                qp_gradient_inf_norm=float(np.max(abs(gradient), initial=0.)),
                active_mask=fitted.active_mask.astype(int),
                coordinate_scale=scaling, solver="scipy.lsq_linear/bvls/exact",
                nonlinear_global_certificate=False)


def _physical_valid(geometry, voltage_limits, p_bounds, q_bounds, source_bounds):
    physical = geometry["physical"]
    voltage = abs(physical["voltage"])
    return bool(np.min(voltage) >= voltage_limits[0] and np.max(voltage) <= voltage_limits[1]
                and np.min(physical["p"]) >= p_bounds[0] and np.max(physical["p"]) <= p_bounds[1]
                and np.min(physical["q"]) >= q_bounds[0] and np.max(physical["q"]) <= q_bounds[1]
                and np.min(physical["source_p"]) >= source_bounds[0] and np.max(physical["source_p"]) <= source_bounds[1]
                and np.min(physical["source_q"]) >= source_bounds[0] and np.max(physical["source_q"]) <= source_bounds[1])


def local_symmetric_scores(candidates, history, model, actions, *, historical_fits=None,
                           source_modes=("fixed", "local_refit"), source_mode=None, constraints=None,
                           line_trust_fraction=.30, source_delta_bound=.02,
                           r_bounds=(1e-6, .1), x_bounds=(1e-6, .1),
                           p_bounds=(-1., 1.), q_bounds=(-1., 1.), source_bounds=(-.5, .5),
                           voltage_limits=(.85, 1.15), future_root_v=None,
                           multistart=2, max_nfev=150, seed=0):
    """History-only conditional local symmetric scores for public EV actions.

    Each source forecast is that model's historical fitted mean, at node1.
    local_refit allows a separate +/-source_delta_bound per future time/PQ
    component; this is a declared local predictive domain, not unrestricted
    robust source uncertainty. Any required candidate/pair failure invalidates
    that action rather than silently dropping a rival.
    """
    started = perf_counter()
    channels = _channels(history)
    if model.fit_bias:
        raise ValueError("known calibrated biases are required in this stage")
    candidates = list(candidates)
    if len(candidates) < 2 or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("at least two uniquely named candidates required")
    if not 0 < line_trust_fraction < 1 or source_delta_bound <= 0 or not np.isfinite(source_delta_bound):
        raise ValueError("positive local trust/source bounds required")
    modes = (source_mode,) if source_mode is not None else tuple(source_modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in ("fixed", "local_refit") for m in modes):
        raise ValueError("source_modes must contain fixed and/or local_refit once")
    action_list = list(actions.values()) if isinstance(actions, Mapping) else list(actions)
    if not action_list or len({a.name for a in action_list}) != len(action_list):
        raise ValueError("nonempty uniquely named action library required")
    n_slots = len(action_list[0].p)
    forecast = baseline_from_history(history, model, n_slots)
    if future_root_v is not None:
        forecast["root_v"] = _broadcast(future_root_v, (n_slots,), "future_root_v")
    observed = whiten_observations(history, model, channels)
    fits = dict(historical_fits or {})
    contexts, failures, summaries = {}, [], {}
    for index, candidate in enumerate(candidates):
        try:
            fitted = fits.get(candidate.name)
            if fitted is None:
                fitted = fit_candidate(candidate, history, model, multistart=multistart,
                    max_nfev=max_nfev, seed=seed+index,
                    r_bounds=r_bounds, x_bounds=x_bounds, p_bounds=p_bounds,
                    q_bounds=q_bounds, source_bounds=source_bounds, voltage_bounds=voltage_limits)
            if not fitted.success or not np.isfinite(fitted.loss):
                raise RuntimeError(f"historical fit unsuccessful: {fitted.message}")
            geometry = mean_and_jacobian(candidate, fitted.r, fitted.x, fitted.p, fitted.q,
                fitted.source_p, fitted.source_q, history.root_v, model, channels=channels)
            metric = historical_profile_geometry(geometry["line_jacobian"], geometry["state_jacobian"])
            beta = np.r_[fitted.r, fitted.x]
            physical_lower = np.r_[np.full(candidate.m, r_bounds[0]), np.full(candidate.m, x_bounds[0])]
            physical_upper = np.r_[np.full(candidate.m, r_bounds[1]), np.full(candidate.m, x_bounds[1])]
            if np.any(beta < physical_lower-1e-12) or np.any(beta > physical_upper+1e-12):
                raise ValueError("historical line estimate lies outside declared physical box")
            contexts[candidate.name] = dict(candidate=candidate, fit=fitted, geometry=geometry,
                profile=metric, beta=beta, lower=np.maximum(physical_lower-beta, -line_trust_fraction*beta),
                upper=np.minimum(physical_upper-beta, line_trust_fraction*beta),
                source_p=np.full(n_slots, np.mean(fitted.source_p)),
                source_q=np.full(n_slots, np.mean(fitted.source_q)),
                base_loss=float(np.sum((geometry["mean"]-observed)**2)))
            summaries[candidate.name] = dict(fitted.summary(),
                recomputed_history_loss=contexts[candidate.name]["base_loss"],
                history_state_rank=metric["state_rank"], history_line_profile_rank=metric["projected_line_rank"],
                history_precision=metric["precision"].tolist(),
                source_forecast_p=float(np.mean(fitted.source_p)), source_forecast_q=float(np.mean(fitted.source_q)))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
            failures.append(dict(candidate=candidate.name, error=str(error), kind="historical_computation_or_domain"))
    common = dict(criterion="conditional_symmetric_bounded_local_soft_history_QP",
        forecast=dict(baseline_p=forecast["p"].tolist(), baseline_q=forecast["q"].tolist(), root_v=forecast["root_v"].tolist()),
        historical_fits=summaries, historical_failures=failures,
        covariance_channels=list(channels), source_node=1, source_spatial_location_shared=True,
        line_trust_fraction=line_trust_fraction, source_delta_bound=source_delta_bound,
        r_bounds=list(r_bounds), x_bounds=list(x_bounds), voltage_limits=list(voltage_limits),
        score_is_probability=False, nonlinear_separation_certified=False,
        history_geometry="Gauss_Newton_after_unconstrained_historical_state_projection",
        future_noise_observations_used=False, local_source_domain_is_uniform_robust=False)
    if failures:
        return dict(common, status="historical_fit_failure", variants={}, actions={}, selected=None,
                    source_mode=modes[0] if len(modes) == 1 else None, seconds=perf_counter()-started)

    variants = {mode:dict(actions={}, selected=None, status="pending") for mode in modes}
    for action in action_list:
        action_started = perf_counter()
        initial, prediction_failure = {}, None
        try:
            ep = _broadcast(action.p, (n_slots, history.n), "action.p")
            eq = _broadcast(action.q, ep.shape, "action.q")
            if constraints is not None and not (np.all(ep == 0) and np.all(eq == 0)):
                if not check_action(action, constraints)["feasible"]:
                    raise ValueError("EV action is infeasible")
            total_p, total_q = forecast["p"]+ep, forecast["q"]+eq
            for name, ctx in contexts.items():
                physical = mean_and_jacobian(ctx["candidate"], ctx["fit"].r, ctx["fit"].x,
                    total_p, total_q, ctx["source_p"], ctx["source_q"], forecast["root_v"], model, channels=channels)
                if not _physical_valid(physical, voltage_limits, p_bounds, q_bounds, source_bounds):
                    raise ValueError(f"initial future physical domain violation: {name}")
                initial[name] = physical
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
            prediction_failure = str(error)
        for mode in modes:
            row = dict(action=action.name, success=False, score=None, pairwise=[], failures=[])
            if prediction_failure is not None:
                row["failures"].append(dict(kind="future_prediction_or_action", error=prediction_failure))
            else:
                for left, right in combinations(contexts, 2):
                    try:
                        pair = _score_pair(contexts[left], contexts[right], initial[left], initial[right],
                            total_p, total_q, forecast["root_v"], history, observed, model, channels,
                            mode, source_delta_bound, source_bounds, p_bounds, q_bounds, voltage_limits)
                        row["pairwise"].append(pair)
                        if not pair["success"]:
                            row["failures"].append(dict(kind="bounded_QP_or_nonlinear_domain", pair=[left, right]))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                        row["failures"].append(dict(kind="pair_computation", pair=[left, right], error=str(error)))
                if not row["failures"]:
                    closest = min(row["pairwise"], key=lambda p: p["proxy_score"])
                    row.update(success=True, score=closest["proxy_score"], closest_pair=closest["candidates"])
            row["seconds"] = perf_counter()-action_started
            variants[mode]["actions"][action.name] = row
    for variant in variants.values():
        valid = [row for row in variant["actions"].values() if row["success"]]
        # Selection excludes a zero-energy control whenever charging actions exist.
        charged = [row for row in valid if np.any(next(a for a in action_list if a.name == row["action"]).p != 0)]
        pool = charged or valid
        variant["selected"] = min(pool, key=lambda row: (-row["score"], row["action"]))["action"] if pool else None
        variant["status"] = "selected" if pool else "all_action_fits_failed"
        variant["failed_action_count"] = sum(not row["success"] for row in variant["actions"].values())
    single = variants[modes[0]] if len(modes) == 1 else dict(actions={}, selected=None)
    return dict(common, variants=variants, actions=single["actions"], selected=single["selected"],
                source_mode=modes[0] if len(modes) == 1 else None,
                status="completed", seconds=perf_counter()-started)


def _score_pair(left, right, gl, gr, p, q, root, history, observed, model, channels,
                mode, source_delta_bound, source_bounds, p_bounds, q_bounds, voltage_limits):
    contexts = (left, right)
    geometries = (gl, gr)
    widths = [len(ctx["beta"]) for ctx in contexts]
    t, n = p.shape
    extra = 2*t if mode == "local_refit" else 0
    total = sum(widths)+2*extra
    design = np.zeros((len(gl["mean"]), total))
    factor_rows = sum(len(ctx["profile"]["factor"]) for ctx in contexts)
    penalty = np.zeros((factor_rows, total))
    lower, upper, scaling = [], [], []
    line_slices, source_slices = [], []
    column, history_row = 0, 0
    for sign, ctx, geo, width in zip((1., -1.), contexts, geometries, widths):
        sl = slice(column, column+width)
        line_slices.append(sl)
        design[:, sl] = sign*geo["line_jacobian"]
        f = ctx["profile"]["factor"]
        penalty[history_row:history_row+len(f), sl] = f
        history_row += len(f)
        column += width
        lower.extend(ctx["lower"]); upper.extend(ctx["upper"])
        scaling.extend(np.maximum(ctx["beta"], 1e-6))
    for sign, ctx, geo in zip((1., -1.), contexts, geometries):
        sl = slice(column, column+extra)
        source_slices.append(sl)
        if extra:
            ids = (np.arange(t)[:, None]*(2*n+2)+np.array([2*n, 2*n+1])).ravel()
            design[:, sl] = sign*geo["state_jacobian"][:, ids]
            source = np.c_[ctx["source_p"], ctx["source_q"]].ravel()
            lower.extend(np.maximum(-source_delta_bound, source_bounds[0]-source))
            upper.extend(np.minimum(source_delta_bound, source_bounds[1]-source))
            scaling.extend(np.full(extra, source_delta_bound))
            column += extra
    d = gl["mean"]-gr["mean"]
    solved = solve_bounded_soft_qp(d, design, penalty, lower, upper, scale=scaling)
    step = solved.pop("step")
    solved["active_mask"] = solved["active_mask"].tolist()
    solved["coordinate_scale"] = solved["coordinate_scale"].tolist()
    actual, actual_history, line_steps, source_steps, physical_feasible = [], [], [], [], []
    for ctx, sl, ssl in zip(contexts, line_slices, source_slices):
        db = step[sl]
        ds = step[ssl].reshape(t, 2) if extra else np.zeros((t, 2))
        beta = ctx["beta"]+db
        candidate = ctx["candidate"]
        updated = mean_and_jacobian(candidate, beta[:candidate.m], beta[candidate.m:], p, q,
            ctx["source_p"]+ds[:, 0], ctx["source_q"]+ds[:, 1], root, model,
            channels=channels, with_jacobian=False)
        old = ctx["fit"]
        dw = (ctx["profile"]["state_compensation"]@db).reshape(history.t, 2*n+2)
        updated_history = mean_and_jacobian(candidate, beta[:candidate.m], beta[candidate.m:],
            old.p+dw[:, :n], old.q+dw[:, n:2*n], old.source_p+dw[:, 2*n], old.source_q+dw[:, 2*n+1],
            history.root_v, model, channels=channels, with_jacobian=False)
        actual.append(updated)
        actual_history.append(float(np.sum((updated_history["mean"]-observed)**2)))
        line_steps.append(db.tolist()); source_steps.append(ds.tolist())
        physical_feasible.append(_physical_valid(updated, voltage_limits, p_bounds, q_bounds, source_bounds)
                                 and _physical_valid(updated_history, voltage_limits, p_bounds, q_bounds, source_bounds))
    exact_difference = actual[0]["mean"]-actual[1]["mean"]
    linear_difference = d+design@step
    error = exact_difference-linear_difference
    history_increments = [loss-ctx["base_loss"] for loss, ctx in zip(actual_history, contexts)]
    actual_distance = float(exact_difference@exact_difference)
    return dict(solved, success=bool(solved["success"] and all(physical_feasible)),
        candidates=[ctx["candidate"].name for ctx in contexts], source_mode=mode,
        line_steps=line_steps, source_steps=source_steps,
        initial_future_distance_squared=float(d@d),
        actual_future_distance_squared=actual_distance,
        taylor_difference_norm=float(np.linalg.norm(error)),
        taylor_relative_error=float(np.linalg.norm(error)/max(np.linalg.norm(exact_difference), np.linalg.norm(d), 1.)),
        actual_history_losses=actual_history, actual_history_loss_increments=history_increments,
        actual_soft_cost_at_compensated_states=float(actual_distance+sum(history_increments)),
        nonlinear_physical_feasible=physical_feasible,
        max_ac_equation_residual=max(v["physical"]["ac_equation_residual"] for v in actual),
        max_power_balance_residual=max(v["physical"]["power_balance_residual"] for v in actual),
        actual_cost_is_nonlinear_infimum=False,
        surrogate_error_is_bound=False)
