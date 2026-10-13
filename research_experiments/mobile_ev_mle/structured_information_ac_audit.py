"""Independent small-branch AC audit around one history-only star center.

This is a local approximation audit, not a selector or global AC distance
certificate.  The generated H+N means use one perturbed shared branch. Every
star competitor refits all its shared R/X and all local P/Q/source states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from . import likelihood, simulation, symmetric_ac_local, information_context
from .likelihood import Observations, fit_candidate
from .simulation import generate, observe, model, BASE_KW
from .candidates import propose
from .information_context import build_centres, quality_models, meter_to_record
from .symmetric_ac_local import mean_and_jacobian, CHANNELS
from .structured_information import information_from_blocks
from .structured_information_checks import projected_information


HERE = Path(__file__).resolve().parent
SOURCE_NAMES = ('structured_information_ac_audit.py', 'structured_information_checks.py',
    'likelihood.py', 'simulation.py', 'candidates.py', 'information_context.py',
    'structured_information.py', 'symmetric_ac_local.py',
    '../mobile_ev_probing/ac_general_v9.py')


def _serial(value):
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, dict):
        return {str(key): _serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_serial(record), ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    path.write_bytes(gzip.compress(data, mtime=0))


def _hashes():
    return {name: sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def _means_observations(physical, meter):
    values = {name: physical[name]+getattr(meter, 'bias_'+name)
              for name in ('p', 'q', 'v')}
    values.update(root_v=physical['root_v'], master_p=physical['master_p'], master_q=physical['master_q'])
    for name in ('source_p', 'source_q'):
        if getattr(meter, name+'_sigma') is not None:
            values[name] = physical[name]
    return Observations(**values)


def _independent_sse(candidate, fit, observations, meter):
    physical = likelihood.ac_predict(candidate, fit.r, fit.x, fit.p, fit.q,
        fit.source_p, fit.source_q, observations.root_v)
    blocks = []
    for name in CHANNELS+tuple(key for key in ('source_p', 'source_q') if getattr(observations, key) is not None):
        prediction = physical[name]
        if name in ('p', 'q', 'v'):
            prediction = prediction+getattr(meter, 'bias_'+name)
        sigma = np.broadcast_to(np.asarray(getattr(meter, name+'_sigma'), float), prediction.shape)
        blocks.append(((prediction-getattr(observations, name))/sigma).ravel())
    residual = np.concatenate(blocks)
    return float(residual@residual), physical


def _physical_record(physical):
    voltage = abs(physical['voltage'])
    return {name: physical[name] for name in ('ac_equation_residual', 'power_balance_residual',
        'p_balance_max_abs', 'q_balance_max_abs', 'iterations')} | dict(
        min_all_nonroot_voltage=float(voltage.min()), max_all_nonroot_voltage=float(voltage.max()))


def _point_record(fit):
    return dict(**fit.summary(), p=fit.p, q=fit.q, source_p=fit.source_p, source_q=fit.source_q,
                bias_p=fit.bias_p, bias_q=fit.bias_q, bias_v=fit.bias_v)


def _observations_record(observations):
    return {name: getattr(observations, name) for name in
        ('p', 'q', 'v', 'root_v', 'master_p', 'master_q', 'source_p', 'source_q')}


def _combine_geometry(history_geometry, future_geometry, target):
    h_line, n_line = history_geometry['line_jacobian'], future_geometry['line_jacobian']
    h_state, n_state = history_geometry['state_jacobian'], future_geometry['state_jacobian']
    other = [j for j in range(h_line.shape[1]) if j not in target]
    a = np.r_[h_line[:, target], n_line[:, target]]
    b = np.r_[np.c_[h_line[:, other], h_state, np.zeros((len(h_line), n_state.shape[1]))],
              np.c_[n_line[:, other], np.zeros((len(n_line), h_state.shape[1])), n_state]]
    independent = projected_information(a, b, rank_rtol=1e-10)
    two_stage = information_from_blocks(h_line, h_state, n_line, n_state, target,
        rcond=1e-10, rank_guard_factor=3.)
    actual = independent['information']
    staged = two_stage['joint_information']['information']
    return dict(information=actual, coefficients=independent['coefficients'],
        nuisance_rank=independent['nuisance_rank'], nuisance_singular_values=independent['nuisance_singular_values'],
        nuisance_rank_tolerance=independent['rank_tolerance'],
        target_information_eigenvalues=np.linalg.eigvalsh(actual),
        target_information_rank=int(np.linalg.matrix_rank(actual, tol=1e-10*max(float(np.linalg.norm(actual, 2)), 1.))),
        target_columns=list(target), other_line_columns=other,
        history_state_columns=h_state.shape[1], future_state_columns=n_state.shape[1],
        complete_nuisance_columns=b.shape[1], complete_observation_rows=b.shape[0],
        two_stage_information=staged,
        two_stage_max_absolute_error=float(np.max(abs(staged-actual))),
        projection_comparison_passed=bool(np.max(abs(staged-actual)) <= 1e-8*max(1., np.linalg.norm(actual, 2))),
        residual_orthogonality=float(np.linalg.norm(b.T@independent['residual'])),
        history_physical=_physical_record(history_geometry['physical']),
        future_physical=_physical_record(future_geometry['physical']))


def _initial_points(star, center, total_states, geometry, direction, epsilon):
    base = dict(r=np.array([center.r[center.candidate.supports.index(s)] for s in star.supports]),
                x=np.array([center.x[center.candidate.supports.index(s)] for s in star.supports]),
                p=total_states['p'].copy(), q=total_states['q'].copy(),
                source_p=total_states['source_p'].copy(), source_q=total_states['source_q'].copy())
    delta = geometry['coefficients']@(epsilon*np.asarray(direction, float))
    m, h_width = star.m, geometry['history_state_columns']
    projected = {key: value.copy() for key, value in base.items()}
    projected['r'] += delta[:m]
    projected['x'] += delta[m:2*m]
    state_delta = np.r_[delta[2*m:2*m+h_width].reshape(center.history_p.shape[0], -1),
                        delta[2*m+h_width:].reshape(len(center.future.p), -1)]
    n = star.n
    projected['p'] += state_delta[:, :n]
    projected['q'] += state_delta[:, n:2*n]
    projected['source_p'] += state_delta[:, 2*n]
    projected['source_q'] += state_delta[:, 2*n+1]
    params = signature(fit_candidate).parameters
    clipping = {}
    for key, value in projected.items():
        bound_name = 'source_bounds' if key.startswith('source_') else key+'_bounds'
        low, high = params[bound_name].default
        corrected = np.clip(value, low+1e-10, high-1e-10)
        clipping[key] = int(np.count_nonzero(abs(corrected-value) > 1e-14))
        projected[key] = corrected
    return base, projected, dict(predicted_compensation=delta,
        projected_initial_clip_counts=clipping, projected_initial=_serial(projected),
        predicted_shared_compensation_l2_pu=float(np.linalg.norm(delta[:2*m])),
        predicted_state_compensation_l2_pu=float(np.linalg.norm(delta[2*m:])))


def audit(output, *, seed=970001):
    began = perf_counter()
    source_start = _hashes()
    record = dict(version='structured_information_ac_local_scale_audit_v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(), status='running', seed=seed,
        source_sha256_start=source_start, records=[], failures=[],
        config=dict(history_points=12, future_points=4, context_refit=False, max_anchors=1,
            quality='temporal', public_quality_row_order=[0, 1, 2, 3], site=0,
            curves_kw={'constant': [3., 3., 3., 3.], 'public_quality': [0., 6., 6., 0.]},
            positive_directions_pu={'R_X': [.003, .002], 'R_only': [.003, 0.], 'X_only': [0., .003]},
            scales=[1., .5, .25, .125], target_support=[0, 1],
            historical_fit_options=dict(multistart=2, max_nfev=300, tol=1e-12, linear_solver='exact'),
            default_competing_fit_options=dict(multistart=2, max_nfev=400, tol=1e-12, linear_solver='exact'),
            projected_competing_fit_options=dict(multistart=1, max_nfev=400, tol=1e-12, linear_solver='exact'),
            original_r_x_bounds=[1e-6, .1], original_p_q_bounds=[-1., 1.],
            original_source_bounds=[-.5, .5], voltage_domain=[.85, 1.15],
            smallest_scale_ratio_acceptance_band=[.85, 1.15]),
        scope='Independent Taylor-distance audit around one H-only point; noiseless nearby H+N means, not the original fixed noisy history likelihood.',
        local_losses_are_numerical_upper_bounds=True, global_AC_distance_certified=False,
        parameter_boxes_shrunk=False, all_required_structural_centers_retained=True,
        selector_uses_truth=False, selector_uses_future_noise=False)
    ac_calls = [0]
    originals = [(likelihood, 'ac_predict', likelihood.ac_predict),
                 (simulation, 'ac_predict', simulation.ac_predict),
                 (symmetric_ac_local, 'ac_predict', symmetric_ac_local.ac_predict),
                 (information_context, 'ac_predict', information_context.ac_predict)]
    original_ac = likelihood.ac_predict
    def counted(*args, **kwargs):
        ac_calls[0] += 1
        return original_ac(*args, **kwargs)
    for module, name, _ in originals:
        setattr(module, name, counted)
    try:
        path = generate(seed, 'star', history_count=12, future_count=4, truth_family=())
        history, _ = observe(path, slice(0, 12))
        meter = model()
        candidates, proposal = propose(history, run_wzzt=False)
        star = next(c for c in candidates if c.name == 'star')
        hfit = fit_candidate(star, history, meter, seed=seed+1,
            **record['config']['historical_fit_options'])
        record.update(history_observations=_observations_record(history), history_model=meter_to_record(meter),
                      candidate_bank=proposal['bank'], history_fit=_point_record(hfit))
        if not hfit.success:
            raise RuntimeError('necessary H-only star fit failed')
        centers, context = build_centres(history, candidates, {star.name: hfit}, meter,
            horizon=4, fit_options=record['config']['historical_fit_options'],
            refit=False, max_anchors=1, seed=seed+2)
        record['context'] = context
        if len(centers) != 3:
            raise RuntimeError('all three required structural centers unavailable')
        future_meter, joint_meter, quality = quality_models(history, meter, 'temporal', row_permutation=np.arange(4))
        record['quality'] = quality
        record['centers'] = [dict(id=c.id, support=c.target_support, r=c.r, x=c.x,
            history_p=c.history_p, history_q=c.history_q, history_source_p=c.history_source_p,
            history_source_q=c.history_source_q, future_p=c.future.p, future_q=c.future.q,
            future_source_p=c.future.source_p, future_source_q=c.future.source_q) for c in centers]
        center = next(c for c in centers if c.target_support == (0, 1))
        cherry = center.candidate
        edge = cherry.supports.index((0, 1))
        target = (edge, cherry.m+edge)
        expected_columns = [((0 if col < cherry.m else 1), cherry.supports[col % cherry.m])
                            for col in range(2*cherry.m) if col not in target]
        star_columns = [(mode, support) for mode in (0, 1) for support in star.supports]
        if expected_columns != star_columns:
            raise RuntimeError('star and nuisance shared-column mapping differs')
        history_geometry = mean_and_jacobian(cherry, center.r, center.x,
            center.history_p, center.history_q, center.history_source_p, center.history_source_q,
            center.history_root_v, meter)
        projections = []
        for curve_index, (curve, powers) in enumerate(record['config']['curves_kw'].items()):
            curve_started = perf_counter()
            p, q = center.future.p.copy(), center.future.q.copy()
            p[:, 0] += np.asarray(powers)/BASE_KW
            # This audit has zero reactive EV injection; P contains EV once.
            states = dict(p=np.r_[center.history_p, p], q=np.r_[center.history_q, q],
                source_p=np.r_[center.history_source_p, center.future.source_p],
                source_q=np.r_[center.history_source_q, center.future.source_q],
                root_v=np.r_[center.history_root_v, center.future.root_v])
            future_geometry = mean_and_jacobian(cherry, center.r, center.x, p, q,
                center.future.source_p, center.future.source_q, center.future.root_v, future_meter)
            geometry = _combine_geometry(history_geometry, future_geometry, target)
            projection_record = {key: value for key, value in geometry.items() if key != 'coefficients'}
            projection_record.update(curve=curve, power_kw=powers)
            projections.append(projection_record)
            if not geometry['projection_comparison_passed']:
                record['failures'].append(dict(kind='independent_projection_disagreement', curve=curve))
            for direction_index, (direction_name, direction) in enumerate(record['config']['positive_directions_pu'].items()):
                direction = np.asarray(direction, float)
                direction_squared = float(direction@geometry['information']@direction)
                for scale_index, epsilon in enumerate(record['config']['scales']):
                    row_started, before = perf_counter(), ac_calls[0]
                    row = dict(curve=curve, power_kw=powers, direction=direction_name,
                        direction_pu=direction, epsilon=epsilon,
                        predicted_full_sse=epsilon**2*direction_squared,
                        information_rank=geometry['target_information_rank'],
                        nuisance_rank=geometry['nuisance_rank'], fits=[], success=False,
                        optimizer_is_local=True, global_distance_lower_bound=None)
                    try:
                        r, x = center.r.copy(), center.x.copy()
                        r[edge], x[edge] = epsilon*direction
                        true_physical = likelihood.ac_predict(cherry, r, x, states['p'], states['q'],
                            states['source_p'], states['source_q'], states['root_v'])
                        data = _means_observations(true_physical, joint_meter)
                        base, projected, projection_step = _initial_points(star, center, states, geometry, direction, epsilon)
                        row.update(true_shared_r=r, true_shared_x=x,
                            mean_observations=_observations_record(data),
                            truth_local_means_physical=_physical_record(true_physical),
                            projection_step=projection_step)
                        fitted_candidates = []
                        for initial_name, initial, options in (
                            ('baseline_two_starts', base, record['config']['default_competing_fit_options']),
                            ('projected_additional_start', projected, record['config']['projected_competing_fit_options'])):
                            fit_started, fit_before = perf_counter(), ac_calls[0]
                            attempted = dict(initial_name=initial_name, declared_multistart=options['multistart'],
                                             success=False, initial=initial)
                            row['fits'].append(attempted)
                            try:
                                fitted = fit_candidate(star, data, joint_meter, initial=initial,
                                    seed=seed+1000+100*curve_index+20*direction_index+2*scale_index,
                                    **options)
                                actual_sse, physical = _independent_sse(star, fitted, data, joint_meter)
                                replay_ok = abs(actual_sse-fitted.loss) <= 1e-8*max(1., abs(fitted.loss))
                                attempt_valid = bool(fitted.success and replay_ok)
                                delta_beta = np.r_[fitted.r-base['r'], fitted.x-base['x']]
                                delta_states = np.c_[fitted.p-base['p'], fitted.q-base['q'],
                                    fitted.source_p-base['source_p'], fitted.source_q-base['source_q']]
                                attempted.update(fit=_point_record(fitted), success=attempt_valid,
                                    independent_full_sse=actual_sse, independent_replay_passed=bool(replay_ok),
                                    physical=_physical_record(physical),
                                    shared_compensation_l2_pu=float(np.linalg.norm(delta_beta)),
                                    shared_compensation_over_epsilon=float(np.linalg.norm(delta_beta)/epsilon),
                                    state_compensation_l2_pu=float(np.linalg.norm(delta_states)),
                                    state_compensation_over_epsilon=float(np.linalg.norm(delta_states)/epsilon),
                                    source_compensation_l2_pu=float(np.linalg.norm(delta_states[:, 6:])),
                                    active_bound_count=fitted.diagnostics.get('active_bound_count'))
                                if attempt_valid:
                                    fitted_candidates.append((actual_sse, initial_name, fitted))
                            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                                attempted.update(error=type(error).__name__+':'+str(error))
                            attempted.update(ac_calls=ac_calls[0]-fit_before, seconds=perf_counter()-fit_started)
                        if fitted_candidates:
                            achieved, initial_name, fitted = min(fitted_candidates, key=lambda item: item[0])
                            predicted = row['predicted_full_sse']
                            ratio = achieved/predicted if predicted > 1e-14 else None
                            row.update(success=True, achieved_local_full_sse=achieved,
                                best_initial_name=initial_name, ratio_local_upper_to_predicted=ratio,
                                all_fit_attempts_success=all(f['success'] for f in row['fits']),
                                relative_prediction_error=None if ratio is None else abs(ratio-1.),
                                smallest_scale_ratio_passed=(.85 <= ratio <= 1.15) if epsilon == .125 and ratio is not None else None)
                        else:
                            row.update(error='all_competitor_fit_initializations_failed')
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                        row.update(error=type(error).__name__+':'+str(error))
                    row.update(ac_calls=ac_calls[0]-before, seconds=perf_counter()-row_started)
                    record['records'].append(row)
                    if not row['success']:
                        record['failures'].append(dict(kind='required_local_fit', curve=curve,
                            direction=direction_name, epsilon=epsilon, error=row.get('error')))
            print(json.dumps(dict(curve_completed=curve, rows=len(record['records']),
                                  seconds=perf_counter()-curve_started)))
            record.update(projections=projections, total_ac_calls=ac_calls[0], seconds=perf_counter()-began)
            _write(output, record)
        source_end = _hashes()
        small = [row for row in record['records'] if row['epsilon'] == .125]
        record.update(source_sha256_end=source_end, source_bytes_unchanged=source_start == source_end,
            status='complete', required_rows=24, completed_rows=len(record['records']),
            successful_rows=sum(row['success'] for row in record['records']),
            fit_attempt_count=sum(len(row['fits']) for row in record['records']),
            fit_attempt_failure_count=sum(not fit['success'] for row in record['records'] for fit in row['fits']),
            independent_projection_checks_passed=all(g['projection_comparison_passed'] for g in projections),
            smallest_scale_ratio_checks=dict(total=len(small), passed=sum(row.get('smallest_scale_ratio_passed') is True for row in small),
                all_passed=all(row.get('smallest_scale_ratio_passed') is True for row in small)),
            total_ac_calls=ac_calls[0], seconds=perf_counter()-began)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        record.update(status='initialization_failure', error=type(error).__name__+':'+str(error),
                      total_ac_calls=ac_calls[0], seconds=perf_counter()-began)
        record['failures'].append(dict(kind='initialization', error=record['error']))
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
    _write(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'evidence/structured_information/ac_local_scale_audit.json.gz')
    parser.add_argument('--seed', type=int, default=970001)
    args = parser.parse_args()
    result = audit(args.output, seed=args.seed)
    print(json.dumps({key: result.get(key) for key in ('status', 'completed_rows', 'successful_rows',
        'fit_attempt_count', 'fit_attempt_failure_count', 'smallest_scale_ratio_checks', 'total_ac_calls', 'seconds')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())