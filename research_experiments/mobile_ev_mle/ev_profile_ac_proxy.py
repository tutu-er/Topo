"""H-only complete-AC local pair profiles for continuous EV curve design.

The free profile is a local linear-model relaxation of bounded refitting.
Its MILP optimization cannot certify a nonlinear AC or global-MLE result.
No truth or future-noise argument enters this module.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import lsq_linear

from .symmetric_ac_local import (CHANNELS, baseline_from_history,
    mean_and_jacobian, whiten_observations)


def internal_line_bounds(candidate, floor_r=1e-6):
    """Physical R/X limits; the full-set and singleton edges keep old limits."""
    if not np.isfinite(floor_r) or not 1e-6 <= floor_r < .1:
        raise ValueError('internal R floor must lie in [1e-6,.1) pu')
    internal = np.array([1 < len(s) < candidate.n for s in candidate.supports])
    rlo = np.where(internal, floor_r, 1e-6)
    xlo = np.where(internal, max(1e-6, (2./3.)*floor_r), 1e-6)
    return dict(r_bounds=(rlo, np.full(candidate.m, .1)),
                x_bounds=(xlo, np.full(candidate.m, .1)), internal=internal)


def profile_affine(c, B, C, *, rcond=1e-10):
    """Project min_z ||c+B u+C z||^2 and preserve its constant by QR.

    Columns of C are normalized before SVD so unlike parameter units do not
    determine the numerical rank. No residual direction is removed by the QR
    compression: [T,d] has at most horizon+1 columns, including the intercept.
    """
    c, B, C = np.asarray(c, float), np.asarray(B, float), np.asarray(C, float)
    if c.ndim != 1 or B.ndim != 2 or C.ndim != 2 or B.shape[0] != len(c) or C.shape[0] != len(c):
        raise ValueError('c, B, C must have matching residual rows')
    if not all(np.isfinite(v).all() for v in (c, B, C)) or not 0 < rcond < 1:
        raise ValueError('finite local arrays and 0 < rcond < 1 required')
    norm = np.linalg.norm(C, axis=0)
    scale = np.ones(C.shape[1])
    scale[norm > 0] = 1./norm[norm > 0]
    normalized = C*scale
    U, singular, Vh = np.linalg.svd(normalized, full_matrices=False)
    rank = int(np.count_nonzero(singular > rcond*singular[0])) if singular.size and singular[0] else 0
    Q = U[:, :rank]
    projected = np.column_stack((B, c))
    projected = projected-Q@(Q.T@projected)
    _, compressed = np.linalg.qr(projected, mode='reduced')
    # All QR rows retained, even if [PB,Pc] is rank-deficient.
    T, d = compressed[:, :-1], compressed[:, -1]
    pinv_scaled = ((Vh[:rank].T/singular[:rank])@Q.T
                   if rank else np.zeros((C.shape[1], len(c))))
    z0 = -scale*(pinv_scaled@c)
    Zu = -scale[:, None]*(pinv_scaled@B)
    gram_error = float(np.max(np.abs(compressed.T@compressed-projected.T@projected))) if projected.size else 0.
    return dict(d=d, T=T, scale=scale, z0=z0, Zu=Zu,
        rank=rank, normalized_singular_values=singular, rcond=rcond,
        compressed_rows=len(d), residual_rows=len(c), qr_gram_error=gram_error,
        projected_intercept=projected[:, -1], projected_control=projected[:, :-1])


def evaluate_model(model, u):
    """Squared local profile cost; score is (q-offset)/2."""
    u = np.asarray(u, float)
    if u.shape != (np.asarray(model['T']).shape[1],) or not np.isfinite(u).all():
        raise ValueError('finite control vector with the declared horizon required')
    residual = np.asarray(model['d'])+np.asarray(model['T'])@u
    q = float(residual@residual)
    return dict(q=q, score=.5*(q-float(model['offset'])))


def bounded_audit(model, u, *, tol=1e-10, max_iter=2000):
    """Bounded convex local QP, feasible upper and affine-tangent lower bounds.

    Reported bounds are numerical local-QP evidence. A feasible competitor
    explanation gives an upper distance bound, not an EV-benefit lower bound.
    """
    began = perf_counter()
    full = model['full']; u = np.asarray(u, float)
    c, B, C = (np.asarray(full[key], float) for key in ('c', 'B', 'C'))
    if u.shape != (B.shape[1],) or not np.isfinite(u).all():
        raise ValueError('invalid control vector')
    lower, upper, scale = (np.asarray(full[key], float) for key in ('lower', 'upper', 'scale'))
    if C.shape[1] == 0 or lower.shape != (C.shape[1],) or upper.shape != lower.shape or scale.shape != lower.shape:
        raise ValueError('bounded audit requires valid competitor parameter columns')
    if not all(np.isfinite(v).all() for v in (lower, upper, scale)) or np.any(lower >= upper) or np.any(scale <= 0):
        raise ValueError('ordered finite bounds and positive scales required')
    r0 = c+B@u
    solved = lsq_linear(C*scale, -r0, bounds=(lower/scale, upper/scale),
        method='bvls', tol=tol, max_iter=max_iter)
    z = np.asarray(solved.x)*scale
    residual = r0+C@z
    primal = float(residual@residual)
    gradient = 2*C.T@residual
    corner = np.where(gradient >= 0., lower, upper)
    tangent_raw = primal+float(gradient@(corner-z))
    lower_bound = min(primal, max(0., tangent_raw))
    free_step = np.asarray(full['free_z0'])+np.asarray(full['free_Zu'])@u
    free_residual = r0+C@free_step
    free_value = float(free_residual@free_residual)
    violation = float(max(0., np.max(lower-free_step), np.max(free_step-upper)))
    feasible = bool(np.isfinite(z).all() and np.all(z >= lower-1e-10) and np.all(z <= upper+1e-10))
    return dict(success=bool(solved.success and feasible), feasible=feasible,
        status=int(solved.status), message=str(solved.message), iterations=int(solved.nit),
        primal_upper=primal, tangent_lower_numeric=lower_bound,
        tangent_raw_numeric=tangent_raw, numerical_gap=primal-lower_bound,
        free_value=free_value, free_step=free_step, free_bound_violation=violation,
        free_feasible=bool(violation <= 1e-10), step=z,
        parameter=np.asarray(full['parameter_center'])+z,
        active_lower=np.flatnonzero(np.abs(z-lower) <= 1e-8).tolist(),
        active_upper=np.flatnonzero(np.abs(z-upper) <= 1e-8).tolist(),
        optimality=float(solved.optimality), seconds=perf_counter()-began,
        scope='frozen linear bounded competitor QP; not nonlinear AC')


def _state_vector(p, q, sp, sq):
    return np.column_stack((p, q, sp, sq)).ravel()


def _parameter_box(candidate, fit, forecast, floor_r):
    bounds = internal_line_bounds(candidate, floor_r)
    h, n, t = len(fit.p), candidate.n, len(forecast['p'])
    center = np.r_[fit.r, fit.x,
        _state_vector(fit.p, fit.q, fit.source_p, fit.source_q),
        _state_vector(forecast['p'], forecast['q'], forecast['source_p'], forecast['source_q'])]
    state_lo = np.r_[np.full(2*n, -1.), -.5, -.5]
    state_hi = -state_lo
    absolute_lo = np.r_[bounds['r_bounds'][0], bounds['x_bounds'][0], np.tile(state_lo, h+t)]
    absolute_hi = np.r_[bounds['r_bounds'][1], bounds['x_bounds'][1], np.tile(state_hi, h+t)]
    if np.any(center < absolute_lo-1e-9) or np.any(center > absolute_hi+1e-9):
        raise ValueError('historical/reference point violates the declared physical domain')
    return center, absolute_lo-center, absolute_hi-center, bounds


def _public_geometry(geometry):
    # Keep the real original means/Jacobians; complex voltages are unnecessary
    # to reproduce the local residual or to compare with a fresh AC response.
    return {key: np.asarray(geometry[key]).copy()
            for key in ('mean', 'line_jacobian', 'state_jacobian')}


def build_pair_models(history, candidates, fits, hmodel, fmodel, *, site=0,
                      horizon=4, reference_kw=None, floor_r=1e-6, base_kw=20.):
    """Build both directions of the H-best pair and fixed-MLE ablations.

    Returns {success, models, fixed_models, diagnostics}. Required fit/center
    failure returns explicit empty model lists; it never silently drops a
    difficult candidate from the declared bank. Each model['full']['c'] is
    the u=0 intercept; reference_residual is the residual at reference_kw.
    """
    began = perf_counter(); candidates = tuple(candidates)
    diagnostics = dict(ac_calls=0, selector_uses_truth=False,
        selector_uses_future_noise=False, source_node=1, source_interface_shared=True,
        future_total_pq_includes_EV_once=True, failure=None,
        bound_scope='free local profile; bounded audit and nonlinear refit required',
        floor_r=float(floor_r), reference_site=site)
    answer = dict(success=False, models=[], fixed_models=[], diagnostics=diagnostics)
    try:
        if int(horizon) != horizon or horizon < 1 or int(site) != site or not 0 <= site < history.n:
            raise ValueError('valid positive horizon and site index required')
        horizon, site = int(horizon), int(site)
        if not np.isfinite(base_kw) or base_kw <= 0:
            raise ValueError('base_kw must be finite and positive')
        u0 = np.full(horizon, 3.) if reference_kw is None else np.asarray(reference_kw, float)
        if u0.shape != (horizon,) or not np.isfinite(u0).all():
            raise ValueError('reference_kw must be a finite horizon vector')
        if len(candidates) < 2 or len({c.name for c in candidates}) != len(candidates):
            raise ValueError('at least two uniquely named candidates required')
        for candidate in candidates:
            fit = fits.get(candidate.name)
            if candidate.n != history.n or fit is None or not fit.success or not np.isfinite(fit.loss):
                raise ValueError('required historical fit failed: '+candidate.name)
        ranked = sorted(candidates, key=lambda c: (fits[c.name].loss, c.name))
        selected = ranked[:2]
        diagnostics.update(selected_pair=[c.name for c in selected],
            history_ranking=[dict(name=c.name, loss=float(fits[c.name].loss)) for c in ranked],
            reference_kw=u0, base_kw=float(base_kw), horizon=horizon)
        channels = CHANNELS+tuple(name for name in ('source_p', 'source_q') if getattr(history, name) is not None)
        obs = whiten_observations(history, hmodel, channels)
        baseline = baseline_from_history(history, hmodel, horizon)
        geometries, forecasts, boxes = {}, {}, {}
        for candidate in selected:
            fit = fits[candidate.name]
            future = dict(p=baseline['p'].copy(), q=baseline['q'].copy(),
                root_v=baseline['root_v'].copy(),
                source_p=np.full(horizon, np.mean(fit.source_p)),
                source_q=np.full(horizon, np.mean(fit.source_q)))
            future['p'][:, site] += u0/base_kw
            diagnostics['ac_calls'] += 1
            hgeo = mean_and_jacobian(candidate, fit.r, fit.x, fit.p, fit.q,
                fit.source_p, fit.source_q, history.root_v, hmodel, channels=channels)
            diagnostics['ac_calls'] += 1
            ngeo = mean_and_jacobian(candidate, fit.r, fit.x, future['p'], future['q'],
                future['source_p'], future['source_q'], future['root_v'], fmodel, channels=channels)
            reconstructed = float(np.sum((hgeo['mean']-obs)**2))
            if abs(reconstructed-float(fit.loss)) > 1e-6*max(1., abs(float(fit.loss))):
                raise ValueError('historical fit loss differs from complete AC means: '+candidate.name)
            geometries[candidate.name] = dict(history=hgeo, future=ngeo)
            forecasts[candidate.name] = future
            boxes[candidate.name] = _parameter_box(candidate, fit, future, floor_r)
        width = 2*history.n+2
        indices = np.arange(horizon)*width+site
        for source in selected:
            competitor = next(c for c in selected if c.name != source.name)
            a, b = source.name, competitor.name
            ag, bg = geometries[a], geometries[b]
            hrows, nrows = len(obs), len(ag['future']['mean'])
            source_B = ag['future']['state_jacobian'][:, indices]/base_kw
            B = np.vstack((np.zeros((hrows, horizon)), source_B))
            def assemble(target):
                geo = geometries[target]
                jh, jn = geo['history'], geo['future']
                C = np.block([
                    [jh['line_jacobian'], jh['state_jacobian'], np.zeros((hrows, width*horizon))],
                    [-jn['line_jacobian'], np.zeros((nrows, width*history.t)), -jn['state_jacobian']]])
                ref = np.r_[jh['mean']-obs, ag['future']['mean']-jn['mean']]
                c = ref-B@u0
                return c, C, ref, profile_affine(c, B, C)
            self_c, self_C, self_ref, self_profile = assemble(a)
            self_control_norm = float(np.linalg.norm(self_profile['T']))
            self_tolerance = 1e-9*max(1., float(np.linalg.norm(B)))
            if self_control_norm > self_tolerance:
                raise ValueError('self-profile unexpectedly depends on EV control: '+a)
            offset = float(self_profile['d']@self_profile['d'])
            c, C, ref, profile = assemble(b)
            center, lower, upper, bounds = boxes[b]
            model = dict(name=a+'__to__'+b, a=a, b=b, kind='free_profile',
                d=profile['d'], T=profile['T'], offset=offset,
                reference_kw=u0.copy(), site=site, base_kw=float(base_kw),
                channels=list(channels), forecast_a=forecasts[a], forecast_b=forecasts[b],
                geometry_a={key:_public_geometry(value) for key,value in ag.items()},
                geometry_b={key:_public_geometry(value) for key,value in bg.items()},
                full=dict(c=c, B=B.copy(), C=C, reference_residual=ref,
                    parameter_center=center, lower=lower, upper=upper,
                    scale=profile['scale'], free_z0=profile['z0'], free_Zu=profile['Zu'],
                    line_columns=2*competitor.m, history_state_columns=width*history.t,
                    future_state_columns=width*horizon, internal_edge_indices=np.flatnonzero(bounds['internal']).tolist(),
                    parameter_order='all_R,all_X,H_timewise_P_Q_sourceP_sourceQ,N_timewise_P_Q_sourceP_sourceQ'),
                projection={key:profile[key] for key in ('rank','normalized_singular_values','rcond','compressed_rows','residual_rows','qr_gram_error')},
                self_profile=dict(d=self_profile['d'], T=self_profile['T'], q_constant=offset,
                    control_norm=self_control_norm, tolerance=self_tolerance,
                    constant_verified=True, original_history_loss=float(fits[a].loss)),
                includes_positive_competitor_internal_lines=True,
                global_AC_certificate=False)
            fixed_B = np.vstack((np.zeros((hrows, horizon)),
                source_B-bg['future']['state_jacobian'][:, indices]/base_kw))
            fixed_c = ref-fixed_B@u0
            fixed_profile = profile_affine(fixed_c, fixed_B, np.empty((hrows+nrows, 0)))
            fixed = dict(name=a+'__to__'+b+'__fixed', a=a, b=b, kind='fixed_MLE',
                d=fixed_profile['d'], T=fixed_profile['T'], offset=float(fits[a].loss),
                reference_kw=u0.copy(), site=site, base_kw=float(base_kw),
                full=dict(c=fixed_c, B=fixed_B, C=np.empty((len(fixed_c),0))),
                forecast_a=forecasts[a], forecast_b=forecasts[b],
                global_AC_certificate=False)
            self_center, self_lower, self_upper, self_bounds = boxes[a]
            model['self_model'] = dict(name=a+'__self', a=a, b=a,
                kind='free_self_profile', d=self_profile['d'], T=self_profile['T'],
                offset=0., reference_kw=u0.copy(), site=site, base_kw=float(base_kw),
                full=dict(c=self_c, B=B.copy(), C=self_C,
                    reference_residual=self_ref, parameter_center=self_center,
                    lower=self_lower, upper=self_upper, scale=self_profile['scale'],
                    free_z0=self_profile['z0'], free_Zu=self_profile['Zu'],
                    line_columns=2*source.m, history_state_columns=width*history.t,
                    future_state_columns=width*horizon,
                    internal_edge_indices=np.flatnonzero(self_bounds['internal']).tolist()))
            model['forecasts'] = dict(a=forecasts[a], b=forecasts[b],
                common_background=baseline)
            fixed['forecasts'] = model['forecasts']
            answer['models'].append(model); answer['fixed_models'].append(fixed)
        diagnostics.update(channels=list(channels), common_background_forecast=baseline,
            candidate_internal_floors={c.name:internal_line_bounds(c, floor_r) for c in selected},
            seconds=perf_counter()-began)
        answer['success']=True
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        answer['models']=[]; answer['fixed_models']=[]
        diagnostics.update(failure=type(error).__name__+':'+str(error), seconds=perf_counter()-began)
    return answer
