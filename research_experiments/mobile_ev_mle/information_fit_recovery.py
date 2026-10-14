"""Separate numerical recovery of frozen information-study fit failures.

Original trials, gate and 19 science sources are immutable. This module uses
saved observations, saved joint meters and the exact saved support ordering.
Every least-squares start and returned/failed finite state is saved and replayed
by an independent branch-current sweep and Gaussian objective calculation.
Successful recovery is diagnostic and never reclassifies the frozen trial.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import gzip
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
from . import likelihood
from .information_replay import _obs, _meter, audit_source
from .gradient_price_study import fit_record, serial

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'evidence/structured_information/v3_frozen'
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q', 'source_p', 'source_q')


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(serial(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_manifest(frozen):
    names = ['freeze.json', 'science_source_snapshot.json.gz']
    for phase in ('calibration', 'evaluation'):
        names.extend(str(p.relative_to(frozen)) for p in (frozen / phase).glob('trial_*.json.gz'))
        names.extend(str(p.relative_to(frozen)) for p in (frozen / phase).glob('*.json'))
        names.extend(str(p.relative_to(frozen)) for p in (frozen / phase / 'tables').glob('*') if p.is_file())
    return {name: sha(frozen / name) for name in sorted(set(names))}


def exact_candidate(row, n):
    supports = tuple(tuple(map(int, support)) for support in row['supports'])
    candidate = likelihood.RadialCandidate.from_supports(row['name'], n, supports)
    if candidate.supports != supports:
        raise ValueError('saved edge parameter ordering differs from reconstructed canonical tree')
    return candidate


def independent_replay(candidate, row, observations, meter):
    """Different AC sweep: backward branch summation followed by forward drops."""
    r, x = (np.asarray(row[key], float) for key in ('r', 'x'))
    p, q, sp, sq = (np.asarray(row[key], float) for key in ('p', 'q', 'source_p', 'source_q'))
    z = r + 1j*x
    root = np.asarray(observations.root_v)
    load = np.zeros((observations.t, candidate.m), complex)
    load[:, 0] = sp + 1j*sq
    load[:, candidate.terminal_edges] += p + 1j*q
    voltage = np.broadcast_to(root[:, None], load.shape).astype(complex).copy()
    for iteration in range(1, 501):
        nodal = np.conj(load/voltage)
        branch = nodal.copy()
        for edge in range(candidate.m-1, -1, -1):
            parent = candidate.parents[edge]
            if parent >= 0:
                branch[:, parent] += branch[:, edge]
        new = np.empty_like(voltage)
        for edge, parent in enumerate(candidate.parents):
            parent_voltage = root if parent < 0 else new[:, parent]
            new[:, edge] = parent_voltage-z[edge]*branch[:, edge]
        if np.max(abs(new-voltage)) < 2e-13:
            voltage = new
            break
        voltage = new
    else:
        raise RuntimeError('independent branch sweep did not converge')
    nodal = np.conj(load/voltage)
    branch = nodal.copy()
    for edge in range(candidate.m-1, -1, -1):
        parent = candidate.parents[edge]
        if parent >= 0:
            branch[:, parent] += branch[:, edge]
    master = root*np.conj(branch[:, 0])
    equation = []
    for edge, parent in enumerate(candidate.parents):
        parent_voltage = root if parent < 0 else voltage[:, parent]
        equation.append(voltage[:, edge]-parent_voltage+z[edge]*branch[:, edge])
    equation_residual = float(np.max(abs(np.asarray(equation))))
    balance = master-load.sum(axis=1)-(abs(branch)**2*z).sum(axis=1)
    balance_residual = float(np.max(abs(balance)))
    physical = dict(p=p, q=q, v=abs(voltage[:, candidate.terminal_edges]), master_p=master.real,
                    master_q=master.imag, source_p=sp, source_q=sq)
    channel_sse = {}
    normalizer = 0.
    prediction_errors = {}
    for channel in CHANNELS:
        observed = getattr(observations, channel)
        if observed is None:
            continue
        sigma = np.broadcast_to(getattr(meter, channel+'_sigma'), observed.shape)
        mean = physical[channel]
        if channel in ('p', 'q', 'v'):
            bias = np.asarray(getattr(meter, 'bias_'+channel))
            if not np.allclose(row['bias_'+channel], bias, atol=1e-12, rtol=0):
                raise ValueError('saved bias differs from actual joint meter')
            mean = mean+bias
        channel_sse[channel] = float(np.sum(((mean-observed)/sigma)**2))
        normalizer += float(np.sum(np.log(2*np.pi*sigma**2)))
        if channel in row.get('predicted', {}):
            prediction_errors[channel] = float(np.max(abs(mean-np.asarray(row['predicted'][channel]))))
    cost = sum(channel_sse.values())
    loglik = -.5*(cost+normalizer)
    domain = bool(np.all((r >= 1e-6-1e-10)&(r <= .1+1e-10)) and
                  np.all((x >= 1e-6-1e-10)&(x <= .1+1e-10)) and
                  np.all(abs(p) <= 1+1e-10) and np.all(abs(q) <= 1+1e-10) and
                  np.all(abs(sp) <= .5+1e-10) and np.all(abs(sq) <= .5+1e-10))
    voltage_ok = bool(abs(voltage).min() >= .85-1e-10 and abs(voltage).max() <= 1.15+1e-10)
    errors = []
    if equation_residual > 2e-10: errors.append('AC_equation')
    if balance_residual > 2e-10: errors.append('AC_power_balance')
    if not domain: errors.append('parameter_domain')
    if not voltage_ok: errors.append('voltage_domain')
    if 'loss' in row and not np.isclose(cost, row['loss'], atol=1e-6, rtol=5e-8): errors.append('SSE')
    if 'loglik' in row and not np.isclose(loglik, row['loglik'], atol=1e-6, rtol=5e-8): errors.append('loglik')
    if prediction_errors and max(prediction_errors.values()) > 2e-10: errors.append('predictions')
    return dict(success=not errors, errors=errors, full_SSE=cost, half_SSE=.5*cost, loglik=loglik,
                Gaussian_normalizer=normalizer, channel_SSE=channel_sse,
                AC_equation_residual=equation_residual, AC_balance_residual=balance_residual,
                voltage_min=float(abs(voltage).min()), voltage_max=float(abs(voltage).max()),
                parameter_domain=domain, voltage_domain=voltage_ok, prediction_max_abs_errors=prediction_errors,
                iterations=iteration, implementation='backward_branch_currents_forward_voltage_drops')


def vector_row(candidate, vector, observations, meter, **metadata):
    m, n, t = candidate.m, candidate.n, observations.t
    vector = np.asarray(vector)
    states = vector[2*m:].reshape(t, 2*n+2)
    return dict(name=candidate.name, supports=candidate.supports, r=vector[:m], x=vector[m:2*m],
                p=states[:, :n], q=states[:, n:2*n], source_p=states[:, 2*n], source_q=states[:, 2*n+1],
                **{key: np.broadcast_to(getattr(meter, key), (n,)) for key in ('bias_p', 'bias_q', 'bias_v')},
                **metadata)


def active_coordinates(candidate, solution):
    m, n = candidate.m, candidate.n
    rows = []
    for index in np.flatnonzero(solution.active_mask):
        if index < 2*m:
            kind = 'r' if index < m else 'x'
            rows.append(dict(index=int(index), variable=kind, support=candidate.supports[index % m],
                             value=float(solution.x[index]), side='lower' if solution.active_mask[index] < 0 else 'upper'))
        else:
            time, offset = divmod(int(index-2*m), 2*n+2)
            kind = 'p' if offset < n else 'q' if offset < 2*n else 'source_p' if offset == 2*n else 'source_q'
            rows.append(dict(index=int(index), variable=kind, time=time,
                             terminal=offset % n if offset < 2*n else None,
                             value=float(solution.x[index]), side='lower' if solution.active_mask[index] < 0 else 'upper'))
    return rows


def run_fit(candidate, observations, meter, *, initial, mode, seed, max_nfev, multistart=4):
    """Instrument frozen solver without editing its bytes or stopping rules."""
    calls = dict(attempted=0, successful=0, failed=0)
    starts = []
    original_ac, original_solver = likelihood.ac_predict, likelihood.least_squares
    began = perf_counter()
    def counted(*args, **kwargs):
        calls['attempted'] += 1
        try:
            answer = original_ac(*args, **kwargs)
        except Exception:
            calls['failed'] += 1
            raise
        calls['successful'] += 1
        return answer
    def captured(fun, x0, *args, **kwargs):
        index = len(starts)
        record = dict(index=index, initial=vector_row(candidate, x0, observations, meter))
        starts.append(record)
        run_start = perf_counter()
        before = dict(calls)
        latest = {}
        def tracked(point):
            residual = fun(point)
            latest.update(vector=np.asarray(point).copy(), loss=float(residual@residual))
            return residual
        try:
            residual0 = tracked(x0)
            record['initial']['loss'] = float(residual0@residual0)
            answer = original_solver(tracked, x0, *args, **kwargs)
            record['returned'] = vector_row(candidate, answer.x, observations, meter,
                loss=float(answer.fun@answer.fun), solver_success=bool(answer.success), status=int(answer.status),
                message=str(answer.message), nfev=int(answer.nfev), njev=int(answer.njev),
                optimality=float(answer.optimality), active_coordinates=active_coordinates(candidate, answer))
            return answer
        except Exception as error:
            record['exception'] = type(error).__name__+':'+str(error)
            if latest:
                record['last_finite_state'] = vector_row(candidate, latest['vector'], observations, meter,
                    loss=latest['loss'], solver_success=False)
            raise
        finally:
            record['seconds'] = perf_counter()-run_start
            record['AC_calls'] = {key: calls[key]-before[key] for key in calls}
    likelihood.ac_predict, likelihood.least_squares = counted, captured
    fitted = None
    error = None
    try:
        fitted = likelihood.fit_candidate(candidate, observations, meter, initial=initial, seed=seed,
            multistart=multistart, max_nfev=max_nfev, tol=1e-10, linear_solver='exact')
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        error = type(exc).__name__+':'+str(exc)
    finally:
        likelihood.ac_predict, likelihood.least_squares = original_ac, original_solver
    solver_seconds = perf_counter()-began
    replay_begin = perf_counter()
    replays = []
    for start in starts:
        for key in ('initial', 'returned', 'last_finite_state'):
            if key in start:
                audit = independent_replay(candidate, start[key], observations, meter)
                start[key]['independent_replay'] = audit
                replays.append(audit)
    best_row = None if fitted is None else fit_record(candidate, fitted)
    best_replay = None if best_row is None else independent_replay(candidate, best_row, observations, meter)
    if best_replay is not None: replays.append(best_replay)
    return dict(candidate=candidate.name, mode=mode, seed=seed,
        options=dict(multistart=multistart, max_nfev=max_nfev, tol=1e-10, linear_solver='exact'),
        attempted_AC_calls=calls, solver_seconds=solver_seconds, replay_seconds=perf_counter()-replay_begin,
        error=error, starts=starts, returned_best_fit=best_row, returned_best_fit_replay=best_replay,
        all_replays_passed=all(row['success'] for row in replays), replay_count=len(replays),
        global_loss_lower_bound=None, global_MLE_certified=False,
        all_initial_and_returned_finite_restart_states_retained=True)


def run(output=None):
    output = Path(output or FROZEN/'failure_diagnostic')
    if (output/'summary.json').exists():
        raise ValueError('completed diagnostic exists; use a new output directory')
    frozen = read_json(FROZEN/'freeze.json')
    before = immutable_manifest(FROZEN)
    source_before = audit_source(frozen['source_sha256'], snapshot=FROZEN/'science_source_snapshot.json.gz')
    if not source_before['success'] or source_before['checked_count'] != 19:
        raise ValueError('frozen science19 byte check failed')
    trial_path = FROZEN/'evaluation/trial_950052.json.gz'
    trial = read_json(trial_path)
    record = next(row for row in trial['records'] if row['method'] == 'constant_fixed')
    history = _obs(trial['history_observations'])
    future = _obs(record['observations'])
    joint = likelihood.concat_observations(history, future)
    meter = _meter(trial['measurement_models']['joint'])
    failures = [name for name in trial['candidate_names'] if not next(row for row in record['fits'] if row['name'] == name)['success']]
    if failures != ['clade_12', 'star', 'clade_02'] or record['status'] != 'joint_fit_failure':
        raise ValueError('target frozen failure differs')
    gate = read_json(FROZEN/'calibration/thresholds.json')['relative']['threshold']
    if gate != 2.9617106149485366: raise ValueError('frozen threshold differs')
    protocol = dict(schema='frozen_information_fit_recovery_v1', history_id='950052', method='constant_fixed',
        diagnostic_only=True, frozen_result_reclassification=False, target_candidates=failures,
        primary_modes=['warm_history', 'cold_meters'], primary_options=dict(multistart=4, max_nfev=800,
            tol=1e-10, linear_solver='exact'), fallback_options=dict(multistart=4, max_nfev=1500),
        fallback_trigger='candidate has no successful primary warm/cold restart',
        fallback_mode='continue_best_finite_primary_state', seed_policy='saved policy seed+200+candidate index',
        original_immutable_SHA256=before, frozen_threshold=gate, source_before=source_before,
        solver_source='unaltered frozen likelihood.fit_candidate', actual_saved_joint_meter=trial['measurement_models']['joint'],
        original_trial_SHA256=sha(trial_path), diagnostics_source_SHA256=sha(Path(__file__)),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    write_json(output/'protocol.json', protocol)
    write_json(output/'input_states.json.gz', dict(history=trial['history_observations'], future=record['observations'],
        joint_meter=trial['measurement_models']['joint'], original_method_status=record['status'],
        original_trial_complete=trial['complete'], original_fits=record['fits'], historical_fits=trial['historical_fits']))
    originals = []
    for row in record['fits']:
        candidate = exact_candidate(row, joint.n)
        originals.append(dict(candidate=row['name'], original_success=row['success'], original_status=row['status'],
            original_loss=row['loss'], replay=independent_replay(candidate, row, joint, meter)))
    write_json(output/'original_fit_replay.json', originals)
    if not all(row['replay']['success'] for row in originals): raise ValueError('original saved fits replay failed')
    histories = {row['name']: row for row in trial['historical_fits']}
    all_runs = []
    for name in failures:
        original = next(row for row in record['fits'] if row['name'] == name)
        candidate = exact_candidate(original, joint.n)
        seed = int(trial['seeds']['policy'])+200+trial['candidate_names'].index(name)
        runs = []
        for mode, initial in (('warm_history', histories[name]), ('cold_meters', None)):
            result = run_fit(candidate, joint, meter, initial=initial, mode=mode, seed=seed, max_nfev=800)
            write_json(output/(name+'_'+mode+'.json.gz'), result)
            runs.append(result); all_runs.append(result)
            print(json.dumps(dict(candidate=name, mode=mode, success=bool(result['returned_best_fit'] and result['returned_best_fit']['success']),
                loss=None if result['returned_best_fit'] is None else result['returned_best_fit']['loss'],
                AC_calls=result['attempted_AC_calls'], seconds=result['solver_seconds']), sort_keys=True), flush=True)
        success = any(start.get('returned', {}).get('solver_success') for item in runs for start in item['starts'])
        if not success:
            finite = [item['returned_best_fit'] for item in runs if item['returned_best_fit'] is not None]
            initial = min(finite, key=lambda item: item['loss']) if finite else None
            result = run_fit(candidate, joint, meter, initial=initial, mode='fallback_continue_best_finite', seed=seed,
                max_nfev=1500)
            write_json(output/(name+'_fallback.json.gz'), result)
            runs.append(result); all_runs.append(result)
    recovered = []
    for name in failures:
        candidate_runs = [row for row in all_runs if row['candidate'] == name]
        finite = [row['returned_best_fit'] for row in candidate_runs if row['returned_best_fit'] is not None]
        accepted = [row for row in finite if row['success']]
        best = min(accepted or finite, key=lambda row: row['loss']) if finite else None
        all_returns = [start['returned'] for item in candidate_runs for start in item['starts'] if 'returned' in start]
        recovered.append(dict(candidate=name, recovered=bool(accepted), best_attained_successful_loss=None if not accepted else min(row['loss'] for row in accepted),
            minimum_all_attained_finite_loss=min((row['loss'] for row in all_returns), default=None),
            best_fit=best, successful_restart_count=sum(row['solver_success'] for row in all_returns),
            failed_restart_count=sum(not row['solver_success'] for row in all_returns),
            restart_count=len(all_returns), actual_AC_calls=sum(row['attempted_AC_calls']['attempted'] for row in candidate_runs)))
    true_original = next(row for row in originals if row['candidate'] == trial['true_name'])
    true_loss = true_original['replay']['full_SSE']
    wrong_losses = [row['best_attained_successful_loss'] for row in recovered if row['recovered']]
    after = immutable_manifest(FROZEN)
    source_after = audit_source(frozen['source_sha256'], snapshot=FROZEN/'science_source_snapshot.json.gz')
    summary = dict(schema='frozen_information_fit_recovery_v1', original_trial_complete=trial['complete'],
        original_method_status=record['status'], original_failed_candidate_count=len(failures),
        recovered_count=sum(row['recovered'] for row in recovered), candidate_results=recovered,
        original_true_candidate=trial['true_name'], original_true_fit_independent_loss=true_loss,
        diagnostic_signed_margin_full_SSE=None if len(wrong_losses) != 3 else min(wrong_losses)-true_loss,
        diagnostic_only=True, original_threshold=gate, frozen_result_reclassification=False,
        original_immutable_records_unchanged=before == after, original_immutable_file_count=len(before),
        science19_unchanged=source_before == source_after, source_after=source_after,
        total_fit_calls=len(all_runs), total_restart_calls=sum(len(row['starts']) for row in all_runs),
        total_solver_nfev=sum(start['returned']['nfev'] for row in all_runs for start in row['starts'] if 'returned' in start),
        total_attempted_AC_calls=sum(row['attempted_AC_calls']['attempted'] for row in all_runs),
        total_successful_AC_calls=sum(row['attempted_AC_calls']['successful'] for row in all_runs),
        total_failed_AC_calls=sum(row['attempted_AC_calls']['failed'] for row in all_runs),
        total_solver_seconds=sum(row['solver_seconds'] for row in all_runs),
        total_replay_seconds=sum(row['replay_seconds'] for row in all_runs),
        total_new_state_replays=sum(row['replay_count'] for row in all_runs), original_state_replays=len(originals),
        all_replays_passed=all(row['all_replays_passed'] for row in all_runs),
        no_global_MLE_or_loss_lower_bound=True, remaining_uncertainty='attained local loss only; other local minima and AC equilibria unenumerated; frozen calibrated policy remains max200')
    write_json(output/'summary.json', summary)
    if not summary['original_immutable_records_unchanged'] or not summary['science19_unchanged'] or not summary['all_replays_passed']:
        raise RuntimeError('diagnostic integrity/replay failed; failure record saved')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('candidate_results', 'source_after')}, sort_keys=True), flush=True)


if __name__ == '__main__': main()
