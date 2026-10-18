"""Replay saved three/seven-day evidence without optimization or new samples."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from ..local_rx_identifiability.audit import (
    _audit_fit as _old_audit_fit, _candidate, _close, _issue, _read, _truth_leaks, _write,
)
from .diagnostics import diagnose, derivative_blocks


SOURCE_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES = ('star', 'pair01', 'pair02', 'pair12')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(spec):
    return f"{spec['network']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def _expected(pilot):
    if pilot:
        return [dict(network='paper15', seed=32991, N=288, boundary=b) for b in ('U', 'R')]
    return [dict(network=net, seed=seed, N=n, boundary=b)
            for net in ('paper15', 'paper11') for seed in range(33001, 33005)
            for n in (288, 672) for b in ('U', 'R', 'K')]


def _inputs(inference, spec, issues, path):
    n, mode = spec['N'], spec['boundary']
    known, meter = inference.get('root_known'), inference.get('root_meter')
    obs = inference.get('observed', {})
    if set(obs) != {'p', 'q', 'v'} or any(np.shape(obs.get(k)) != (n, 3) for k in ('p', 'q', 'v')):
        _issue(issues, 'observation_channels_or_dimensions', path, {k:np.shape(v) for k, v in obs.items()})
    if mode == 'U' and (known is not None or meter is not None):
        _issue(issues, 'unmeasured_boundary_trajectory_exposed', path, 'U received a boundary input')
    if mode == 'R':
        if known is not None or meter is None or len(meter) != 2:
            _issue(issues, 'noisy_boundary_input', path, 'R requires a meter and no fixed boundary')
        elif np.shape(meter[0]) != (n,) or not np.isfinite(np.asarray(meter[0], float)).all():
            _issue(issues, 'noisy_boundary_dimensions', path, np.shape(meter[0]))
        elif np.any(np.asarray(meter[1]) != .0005):
            _issue(issues, 'noisy_boundary_sigma', path, meter[1])
    if mode == 'K' and (known is None or np.shape(known) != (n,) or meter is not None):
        _issue(issues, 'ideal_boundary_input', path, 'K requires its explicitly ideal boundary only')
    if inference.get('truth_used') is not False:
        _issue(issues, 'truth_flag', path, inference.get('truth_used'))
    if inference.get('interval_minutes') != 15 or n not in (288, 672):
        _issue(issues, 'duration', path, 'expected 288 or 672 ordinary 15-minute records')
    if 'ideal' not in inference.get('information_rule', '').lower():
        _issue(issues, 'boundary_information_declaration', path, inference.get('information_rule'))
    for leak in _truth_leaks(inference):
        _issue(issues, 'truth_field_leak', path, leak)


def _traces(fitted, issues, path):
    attempts = fitted.get('attempts', [])
    eligible = []
    for index, attempt in enumerate(attempts):
        trace = np.asarray(attempt.get('trace', []), float)
        if trace.size and (not np.isfinite(trace).all() or np.any(np.diff(trace) >= 0)):
            _issue(issues, 'attempt_trace_not_decreasing', f'{path}.attempt{index}', trace.tolist())
        if 'loss' in attempt:
            if not trace.size or not _close(float(trace[-1]), float(attempt['loss'])):
                _issue(issues, 'attempt_trace_final_loss', f'{path}.attempt{index}', attempt.get('loss'))
            if int(attempt.get('accepted_steps', -1)) != len(trace)-1:
                _issue(issues, 'attempt_trace_step_count', f'{path}.attempt{index}', attempt.get('accepted_steps'))
            eligible.append(attempt)
    if 'loss' in fitted:
        trace = np.asarray(fitted.get('trace', []), float)
        if not trace.size or not _close(float(trace[-1]), float(fitted['loss'])) or np.any(np.diff(trace) >= 0):
            _issue(issues, 'chosen_trace', path, fitted.get('trace'))
        if not eligible:
            _issue(issues, 'chosen_attempt_missing', path, 'state present without an attained attempt')
        else:
            best = min(eligible, key=lambda a:(not a.get('physical', False), float(a['loss'])))
            if not _close(float(best['loss']), float(fitted['loss'])):
                _issue(issues, 'chosen_loss_not_best_legal_attempt', path, (best['loss'], fitted['loss']))


def audit_fit(candidate, fitted, observed, sigma, domain, issues, path, *,
              root_known=None, root_meter=None, saved_diagnostics=None):
    """Independent physical solver, raw SSE, derivative and trace replay."""
    _traces(fitted, issues, path)
    raw = copy.deepcopy(fitted)
    root_sse = 0.
    if root_meter is not None and all(k in fitted for k in ('root_v', 'loss')):
        measured, std = (np.asarray(k, float) for k in root_meter)
        root_sse = float(np.sum(((np.asarray(fitted['root_v'])-measured)/std)**2))
        raw['loss'] = float(raw['loss'])-root_sse
        # The old replay assumes any unfixed root is unmetered; disable only
        # that obsolete rank check here, and audit the complete rank below.
        raw.pop('information', None)
    local_issues = []
    row = _old_audit_fit(candidate, raw, observed, sigma, domain, local_issues, path,
                         root_known=root_known)
    for item in local_issues:
        if not fitted.get('qualified') and item['code'] == 'physical_domain_violation':
            item['severity'] = 'warning'
        issues.append(item)
    if not fitted.get('qualified'):
        _issue(issues, 'scientific_fit_not_qualified', path, fitted.get('status'), 'warning')
    if not row.get('replayed'):
        return row
    row.update(root_meter_sse=root_sse, complete_raw_sse=row['raw_sse']+root_sse)
    if not _close(row['complete_raw_sse'], float(fitted['loss'])):
        _issue(issues, 'complete_loss_mismatch', path, row['complete_raw_sse'])
    if 'J' in fitted and not _close(2.*float(fitted['J']), float(fitted['loss'])):
        _issue(issues, 'J_not_half_sse', path, fitted['J'])
    try:
        value = derivative_blocks(candidate, observed, sigma, fitted,
                                  root_known=root_known, root_meter=root_meter)
        a, c, residual = (value[k] for k in ('shared', 'state', 'residual'))
        count, n, m = len(observed['p']), candidate.n, candidate.m
        expected_rows = 3*n+(root_meter is not None)
        if residual.shape != (count, expected_rows):
            _issue(issues, 'residual_shape', path, residual.shape)
        if 'residual' in fitted and (np.shape(fitted['residual']) != residual.shape or
                not np.allclose(fitted['residual'], residual, atol=1e-8, rtol=1e-9)):
            _issue(issues, 'saved_residual_mismatch', path, np.shape(fitted['residual']))
        for key, exact in (('theta', value['theta']), ('local', value['local'])):
            if key not in fitted or np.shape(fitted[key]) != exact.shape or not np.allclose(fitted[key], exact, atol=1e-9, rtol=1e-9):
                _issue(issues, 'saved_coordinate_mismatch', path, key)
        low = np.r_[np.full(m, np.log(domain['z_min'])), np.full(m, np.arctan(domain['ratio_min']))]
        high = np.r_[np.full(m, np.log(domain['z_max'])), np.full(m, np.arctan(domain['ratio_max']))]
        sig = {k:np.broadcast_to(np.asarray(sigma[k]), (count, n)) for k in ('p', 'q')}
        cap = np.asarray(domain['power_caps'])
        ll = np.c_[(-cap-observed['p'])/sig['p'], (-cap-observed['q'])/sig['q']]
        lu = np.c_[(cap-observed['p'])/sig['p'], (cap-observed['q'])/sig['q']]
        if root_known is None:
            ll = np.c_[ll, np.full(count, (domain['root_min']-1.)/.01)]
            lu = np.c_[lu, np.full(count, (domain['root_max']-1.)/.01)]
        ga, gc = np.einsum('tri,tr->i', a, residual), np.einsum('tri,tr->ti', c, residual)
        pg = max(float(np.max(abs(np.clip(-ga, low-value['theta'], high-value['theta'])))),
                 float(np.max(abs(np.clip(-gc, ll-value['local'], lu-value['local'])))))
        row['recomputed_projected_gradient'] = pg
        if not _close(pg, float(fitted.get('projected_gradient', np.nan)), atol=1e-6, rtol=1e-6):
            _issue(issues, 'projected_gradient_mismatch', path, pg)
        if fitted.get('qualified') and pg > 1e-4+1e-8:
            _issue(issues, 'qualified_without_stationarity', path, pg)
        recomputed = diagnose(candidate, observed, sigma, domain, fitted,
                              root_known=root_known, root_meter=root_meter)
        info = recomputed['profile_information']
        row['profile_rank'] = info['rank']
        if int(fitted.get('information', {}).get('rank', -1)) != info['rank']:
            _issue(issues, 'fit_information_rank_mismatch', path, fitted.get('information'))
        if root_known is None and root_meter is None and info['rank'] > 2*m-2:
            _issue(issues, 'unmeasured_common_trunk_rank', path, info['rank'])
        for check in recomputed['finite_direction_checks']:
            if check['residual_relative_error'] > 5e-6 and check['residual_max_abs_error'] > 1e-6:
                _issue(issues, 'finite_derivative_check', path, check)
            if 'J_gradient_analytic' in check and not _close(
                    check['J_gradient_analytic'], check['J_gradient_finite'], atol=1e-4, rtol=1e-5):
                _issue(issues, 'finite_objective_gradient_check', path, check)
        row['max_finite_derivative_relative_error'] = max(d['residual_relative_error'] for d in recomputed['finite_direction_checks'])
        identity = recomputed['local_star_identity']
        if identity.get('supported') and max(identity['branch_max_abs'], identity['contrast_max_abs']) > 2e-10:
            _issue(issues, 'local_physical_identity', path, identity)
        if saved_diagnostics is not None:
            for key in ('loss_recomputed', 'loss_reported', 'loss_absolute_difference'):
                if not _close(float(saved_diagnostics.get(key, np.nan)), float(recomputed[key])):
                    _issue(issues, 'saved_diagnostic_loss', path, key)
            if saved_diagnostics.get('profile_information', {}).get('rank') != info['rank']:
                _issue(issues, 'saved_diagnostic_rank', path, saved_diagnostics.get('profile_information'))
            saved_fd = saved_diagnostics.get('finite_direction_checks', [])
            if len(saved_fd) != len(recomputed['finite_direction_checks']):
                _issue(issues, 'saved_finite_check_count', path, len(saved_fd))
            else:
                for old, new in zip(saved_fd, recomputed['finite_direction_checks']):
                    keys = ('residual_relative_error', 'residual_max_abs_error')
                    if 'J_gradient_analytic' in new:
                        keys += ('J_gradient_analytic', 'J_gradient_finite')
                    for key in keys:
                        if not _close(float(old.get(key, np.nan)), float(new[key]), atol=1e-7, rtol=1e-5):
                            _issue(issues, 'saved_finite_check_mismatch', path, (key, old.get(key), new[key]))
    except Exception as exc:
        _issue(issues, 'point_diagnostic_replay_failed', path, repr(exc))
    return row


def audit_directory(directory, *, destination=None):
    started, directory = perf_counter(), Path(directory)
    issues, rows, paired = [], [], {}
    config = _read(directory/'configuration.json')
    tasks, pilot = config['tasks'], bool(config.get('pilot'))
    expected = _expected(pilot)
    representative = config.get('representative_replay') is True
    if representative:
        valid_subset = (len(tasks) == 2 and all(s in _expected(False)+_expected(True) for s in tasks)
                        and {s['boundary'] for s in tasks} == {'U', 'R'}
                        and len({(s['network'], s['seed'], s['N']) for s in tasks}) == 1)
        if not valid_subset:
            _issue(issues, 'representative_slot_matrix', directory, tasks)
    elif tasks != expected:
        _issue(issues, 'configuration_slot_matrix', directory, tasks)
    families = {(s['network'], s['seed']) for s in tasks}
    if config.get('independent_random_families') != len(families):
        _issue(issues, 'independent_family_denominator', directory, config.get('independent_random_families'))
    if config.get('protocol_sha256') != _sha(Path(__file__).with_name('protocol.md')):
        _issue(issues, 'protocol_hash_mismatch', directory, config.get('protocol_sha256'))
    manifest = config.get('source_manifest', {})
    if not manifest:
        _issue(issues, 'source_manifest_missing', directory, '')
    for name, digest in manifest.items():
        path = (SOURCE_ROOT/name).resolve()
        if not path.is_relative_to(SOURCE_ROOT.resolve()) or not path.is_file() or _sha(path) != digest:
            _issue(issues, 'source_hash_mismatch', directory, name)
    completed = qualified = state_count = 0
    for spec in tasks:
        key = _key(spec); row = dict(key=key, spec=spec, fits={}); rows.append(row)
        try:
            result = _read(directory/'results'/f'{key}.json.gz')
            if result.get('spec') != spec:
                _issue(issues, 'result_spec', key, result.get('spec'))
            if result.get('status') != 'completed':
                row.update(status=result.get('status'), failure_preserved=True)
                _issue(issues, 'scientific_slot_failed', key, result.get('error'), 'warning'); continue
            completed += 1
            path = directory/'inference'/f'{key}.json.gz'
            inference = _read(path)
            if result.get('inference_sha256') != _sha(path):
                _issue(issues, 'inference_hash_mismatch', key, result.get('inference_sha256'))
            if inference.get('spec') != spec:
                _issue(issues, 'inference_spec', key, inference.get('spec'))
            _inputs(inference, spec, issues, key)
            observed = {k:np.asarray(v, float) for k, v in inference['observed'].items()}
            pair = paired.setdefault((spec['network'], spec['seed']), [])
            pair.append((spec, inference))
            truth = _read(directory/'truth'/f'{key}.json.gz')
            if result.get('evaluation', {}).get('truth_read_after_inference_seal') is not True:
                _issue(issues, 'inference_seal_declaration', key, result.get('evaluation'))
            if spec['boundary'] == 'K' and not np.array_equal(inference['root_known'], np.asarray(truth['root_v'])[:spec['N']]):
                _issue(issues, 'ideal_boundary_not_declared_truth', key, '')
            if spec['boundary'] == 'R' and np.array_equal(inference['root_meter'][0], np.asarray(truth['root_v'])[:spec['N']]):
                _issue(issues, 'noisy_boundary_replaced_by_truth', key, '')
            if set(inference.get('fits', {})) != set(CANDIDATES):
                _issue(issues, 'candidate_denominator', key, list(inference.get('fits', {})))
            for name in CANDIDATES:
                fitted = inference.get('fits', {}).get(name)
                if fitted is None:
                    continue
                qualified += int(bool(fitted.get('qualified')))
                row['fits'][name] = audit_fit(_candidate(name), fitted, observed,
                    inference['sigma'], inference['domain'], issues, f'{key}.{name}',
                    root_known=inference.get('root_known'), root_meter=inference.get('root_meter'),
                    saved_diagnostics=inference.get('diagnostics', {}).get(name))
                state_count += int(row['fits'][name].get('replayed', False))
            if result.get('qualified') != sum(bool(f.get('qualified')) for f in inference['fits'].values()):
                _issue(issues, 'result_qualified_denominator', key, result.get('qualified'))
        except Exception as exc:
            _issue(issues, 'slot_audit_exception', key, repr(exc)); row['audit_error'] = repr(exc)
    pair_checks = 0
    for family, records in paired.items():
        for index, (a, ia) in enumerate(records):
            for b, ib in records[index+1:]:
                stop = min(a['N'], b['N'])
                for channel in ('p', 'q', 'v'):
                    if not np.array_equal(np.asarray(ia['observed'][channel])[:stop], np.asarray(ib['observed'][channel])[:stop]):
                        _issue(issues, 'paired_observation_mismatch', family, (a, b, channel))
                if ia['sigma'] != ib['sigma'] or ia['domain'] != ib['domain']:
                    _issue(issues, 'paired_meter_or_domain_mismatch', family, (a, b))
                if a['boundary'] == b['boundary'] == 'R' and not np.array_equal(
                        np.asarray(ia['root_meter'][0])[:stop], np.asarray(ib['root_meter'][0])[:stop]):
                    _issue(issues, 'paired_root_readings_mismatch', family, (a, b))
                pair_checks += 1
    extra = set(p.name[:-8] for p in (directory/'results').glob('*.json.gz'))-{_key(s) for s in tasks}
    if extra:
        _issue(issues, 'unconfigured_results', directory, sorted(extra))
    summary = dict(schema='local_rx_three_day_audit_v1', fitting_performed=False,
        representative_replay=representative,
        coverage='only the declared two-record U/R restoration subset' if representative else 'full configured study',
        configured_conditions=len(tasks), independent_random_families=len(families),
        pairing='sample length and boundary modes share histories; candidate fits are not independent histories',
        configured_candidate_slots=len(tasks)*4, completed_conditions=completed,
        qualified_candidate_fits=qualified, replayed_states=state_count, paired_comparisons=pair_checks,
        errors=sum(i['severity'] == 'error' for i in issues), warnings=sum(i['severity'] == 'warning' for i in issues),
        audit_pass=not any(i['severity'] == 'error' for i in issues), issues=issues, conditions=rows,
        audit_source_sha256=_sha(__file__), elapsed_seconds=perf_counter()-started)
    _write(destination or directory/'audit.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--destination'); args = parser.parse_args()
    result = audit_directory(args.directory, destination=args.destination)
    print({k:result[k] for k in ('audit_pass', 'configured_conditions', 'replayed_states', 'errors', 'warnings')})
    raise SystemExit(0 if result['audit_pass'] else 1)


if __name__ == '__main__':
    main()
