"""Independent saved-state AC replay for systematic EV batches.

No parameter fitting, action selection or simulator generation is called.
Missing states and recorded failures remain in the original case denominator.
The same reader accepts raw or compact records that retain meters and states.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_original_review as physical
from .likelihood import RadialCandidate

HERE = Path(__file__).resolve().parent
METHODS = ('none', 'constant', 'highlow', 'square_fixed', 'square_mobile',
           'nearest_fixed', 'gradac_fixed', 'scalar_fixed', 'scalar_hist_fixed', 'scalar_mobile')
MODES = ('fixed_total', 'fixed_holdout', 'joint_total')
STATE_FIELDS = ('r', 'x', 'source_p', 'source_q')


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, message, *, atol=2e-7, rtol=2e-8):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), message)


def _array(value, shape):
    result = np.broadcast_to(np.asarray(value, float), shape)
    require(np.isfinite(result).all(), 'nonfinite meter/state array')
    return result


def _state(row):
    require(isinstance(row, dict) and all(k in row for k in STATE_FIELDS), 'missing fitted state fields')
    return {key: np.asarray(row[key], float) for key in STATE_FIELDS}


def _domain(state, cfg):
    r, x = state['r'], state['x']
    require(r.shape == x.shape and r.ndim == 1 and np.isfinite(r).all()
            and np.isfinite(x).all(), 'line-state shape/nonfinite')
    ratio = cfg['ratio_bounds']
    lo_r, hi_r = cfg['training']['r_bounds']
    lo_x, hi_x = cfg['training']['x_bounds']
    line = bool(np.all(r >= np.asarray(lo_r) - 1e-10) and np.all(r <= np.asarray(hi_r) + 1e-10)
                and np.all(x >= np.asarray(lo_x) - 1e-10) and np.all(x <= np.asarray(hi_x) + 1e-10)
                and np.all(x >= ratio[0] * r - 1e-10) and np.all(x <= ratio[1] * r + 1e-10))
    source = True
    for key in ('source_p', 'source_q'):
        lo, hi = cfg['source'][key + '_bounds']
        source = source and bool(np.isfinite(state[key]).all()
            and np.all(state[key] >= lo - 1e-10) and np.all(state[key] <= hi + 1e-10))
    return line, source


def replay_state(candidate, state, blocks, cfg):
    """Direct calibrated P/Q inputs and an independent branch-current sweep."""
    state = _state(state)
    n = candidate.n
    total_times = sum(len(obs['p']) for _, obs, _ in blocks)
    require(state['r'].shape == (candidate.m,) and state['source_p'].shape == (total_times,)
            and state['source_q'].shape == (total_times,), 'source/time/line state dimensions')
    line, source = _domain(state, cfg)
    output, total, normalizer, start = {}, 0., 0., 0
    maximum_equation = maximum_balance = 0.
    physical_voltage = True
    voltage_bounds = cfg['training'].get('voltage_bounds', [.85, 1.15])
    for label, obs, meter in blocks:
        T = len(obs['p'])
        shape = (T, n)
        p = _array(obs['p'], shape) - _array(meter['bias_p'], shape)
        q = _array(obs['q'], shape) - _array(meter['bias_q'], shape)
        result = physical.forward_physics(candidate, state['r'], state['x'], p, q,
            state['source_p'][start:start + T], state['source_q'][start:start + T], obs['root_v'])
        pieces, channels = [], {}
        for channel in ('v', 'master_p', 'master_q'):
            shape = (T, n) if channel == 'v' else (T,)
            sigma = _array(meter[channel + '_sigma'], shape)
            require(np.all(sigma > 0), 'positive output sigma required')
            bias = _array(meter['bias_v'], shape) if channel == 'v' else 0.
            residual = (result[channel] + bias - _array(obs[channel], shape)) / sigma
            pieces.append(residual.reshape(T, -1))
            channels[channel] = float(np.sum(residual ** 2))
            normalizer += float(np.sum(np.log(2 * np.pi * sigma ** 2)))
        residual = np.hstack(pieces)
        loss = float(np.sum(residual ** 2))
        output[label] = dict(loss=loss, residual=residual, channel_loss=channels, time_count=T)
        total += loss
        start += T
        maximum_equation = max(maximum_equation, result['equation_residual'])
        maximum_balance = max(maximum_balance, result['balance_residual'])
        physical_voltage = physical_voltage and (result['voltage_min'] >= voltage_bounds[0]
            and result['voltage_max'] <= voltage_bounds[1])
    return dict(loss=total, normalizer=normalizer, loglik=-.5 * (total + normalizer),
        blocks=output, line_domain_feasible=line, source_box_feasible=source,
        physical_voltage_feasible=bool(physical_voltage),
        max_AC_equation=maximum_equation, max_power_balance=maximum_balance)


def _compare_score(saved, actual, label, metrics, *, selected=True):
    close(saved['loss'], actual['loss'], label + ': total SSE')
    if 'normalizer' in saved:
        close(saved['normalizer'], actual['normalizer'], label + ': output normalizer', atol=1e-9, rtol=0.)
    if saved.get('loglik') is not None:
        close(saved['loglik'], actual['loglik'], label + ': log likelihood')
    for name, row in saved.get('blocks', {}).items():
        require(name in actual['blocks'], label + ': unexpected score block')
        check = actual['blocks'][name]
        close(row['loss'], check['loss'], label + '/' + name + ': SSE')
        close(row['residual'], check['residual'], label + '/' + name + ': whitened residual')
        metrics['max_whitened_residual_error'] = max(metrics['max_whitened_residual_error'],
            float(np.max(abs(np.asarray(row['residual']) - check['residual']))))
        metrics['max_block_SSE_error'] = max(metrics['max_block_SSE_error'], abs(row['loss'] - check['loss']))
        if 'channel_loss' in row:
            for channel, value in row['channel_loss'].items():
                close(value, check['channel_loss'][channel], label + '/' + name + '/' + channel + ': SSE')
    for key in ('line_domain_feasible', 'source_box_feasible', 'physical_voltage_feasible'):
        if key in saved:
            require(bool(saved[key]) == actual[key], label + ': ' + key + ' flag')
        if selected:
            require(actual[key], label + ': selected fitted state outside ' + key)
    metrics['max_AC_equation'] = max(metrics['max_AC_equation'], actual['max_AC_equation'])
    metrics['max_power_balance'] = max(metrics['max_power_balance'], actual['max_power_balance'])
    metrics['saved_states_replayed'] += 1
    metrics['AC_block_replays'] += len(actual['blocks'])


def review_trial(record):
    """Preserve incomplete and inconsistent cases as separate outcomes."""
    began = perf_counter()
    errors, missing = [], []
    metrics = dict(saved_states_replayed=0, AC_block_replays=0,
        max_whitened_residual_error=0., max_block_SSE_error=0., max_AC_equation=0., max_power_balance=0.,
        method_mode_checks=0, PQ_once_checks=0, recorded_planning_failures=0,
        recorded_inference_failures=0, recorded_operation_failures=0, planning_dependency_checks=0)
    reason = record.get('failure', record.get('status', 'state absent'))

    def attempt(label, work):
        try:
            return work()
        except (ValueError, KeyError, TypeError, IndexError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as exc:
            errors.append(dict(check=label, reason=type(exc).__name__ + ': ' + str(exc)))
            return None

    def absent(label, why=None):
        missing.append(dict(check=label, reason=why or reason))

    methods = {row.get('name'): row for row in record.get('methods', [])}
    if (len(record.get('methods', [])) != len(METHODS) or len(methods) != len(METHODS)
            or set(methods) != set(METHODS)):
        errors.append(dict(check='method denominator', reason='exactly ten distinct method slots required'))
    for method in methods.values():
        metrics['recorded_planning_failures'] += bool(method.get('planning_failure'))
        metrics['recorded_operation_failures'] += not method.get('operation', {}).get('success', False)
        metrics['recorded_inference_failures'] += sum(not row.get('complete', False)
            for row in method.get('inference', {}).values())
    if record.get('selection', {}).get('methods'):
        attempt('planning standalone dependency costs', lambda: _planning(record, methods, metrics))
    required = ('training', 'training_meter', 'validation', 'validation_meter', 'future_meter', 'training_result')
    if any(key not in record for key in required):
        for key in required:
            if key not in record:
                absent(key)
        return _result(record, metrics, errors, missing, began)
    cfg = record['config']
    require(cfg.get('source_interface') == 1, 'same physical source interface1 required')
    n = len(record['training']['p'][0])
    bank = {row['name']: RadialCandidate.from_supports(row['name'], n,
        [tuple(s) for s in row['supports'] if 1 < len(s) < n]) for row in record['candidates']}
    training = (('training', record['training'], record['training_meter']),)
    history = (*training, ('validation', record['validation'], record['validation_meter']))
    trained = {row.get('candidate', row.get('name')): row for row in record['training_result'].get('fits', [])}
    for name, candidate in bank.items():
        if name not in trained:
            absent('training/' + name, str(record['training_result'].get('failures', {})))
            continue
        row = trained[name]
        attempt('training/' + name, lambda c=candidate, r=row: _compare_score(r,
            replay_state(c, r, training, cfg), 'training/' + c.name, metrics, selected=bool(r.get('success'))))
        prefit = record.get('history_prefits', {}).get(name)
        if prefit is None:
            absent('history_prefits/' + name)
            continue
        fixed_pre, joint_pre = prefit['fixed'], prefit['joint']
        val = fixed_pre.get('validation_fit')
        if val is not None:
            attempt('validation/' + name, lambda c=candidate, v=val: _compare_score(v,
                replay_state(c, v, history[1:], cfg), 'validation/' + c.name, metrics,
                selected=bool(v.get('success'))))
        else:
            absent('validation_fit/' + name)
        if joint_pre.get('state') is not None:
            attempt('joint_pre/' + name, lambda c=candidate, r=joint_pre: _compare_score(r,
                replay_state(c, r['state'], history, cfg), 'joint_pre/' + c.name, metrics))
        else:
            absent('joint_pre_state/' + name)
        if val is not None:
            state = {key: row[key] for key in ('r', 'x')}
            state.update({key: np.r_[row[key], val[key]] for key in ('source_p', 'source_q')})
            attempt('fixed_pre/' + name, lambda c=candidate, r=fixed_pre, s=state: _compare_score(r,
                replay_state(c, s, history, cfg), 'fixed_pre/' + c.name, metrics))
        _optimizer(candidate, joint_pre, history, cfg, metrics, attempt, 'joint_pre/' + name)
    futures = record.get('future_records', {})
    if not futures:
        absent('future_records', 'no actual action/fit states: ' + str(reason))
    off = methods.get('none', {}).get('action_key')
    off_obs = futures.get(off, {}).get('observations')
    for key, future in futures.items():
        obs = future['observations']
        blocks = (*history, ('future', obs, record['future_meter']))
        power = np.asarray(future['power_kw'], float)
        attempt('action hash/' + key, lambda: require(
            sha256(np.asarray(power, dtype='<f8').tobytes()).hexdigest() == key, 'action bytes/hash mismatch'))
        if off_obs is not None:
            def pq_once():
                close(np.asarray(obs['p']) - off_obs['p'], power / float(cfg['base_kw']),
                      'EV appears once in observed total P', atol=2e-15, rtol=2e-13)
                close(obs['q'], off_obs['q'], 'same Q and paired noise', atol=0., rtol=0.)
                close(obs['root_v'], off_obs['root_v'], 'same fixed root voltage', atol=0., rtol=0.)
                metrics['PQ_once_checks'] += 1
            attempt('PQ_once/' + key, pq_once)
        else:
            absent('none baseline for PQ_once/' + key)
        comparisons = future.get('comparisons', {})
        for name, candidate in bank.items():
            if name not in comparisons:
                absent('future/' + key + '/' + name, str(future.get('failures', {})))
                continue
            row = comparisons[name]
            for side in ('fixed', 'joint'):
                post = row[side]['post']
                attempt(key + '/' + name + '/' + side, lambda c=candidate, p=post, b=blocks, s=side:
                    _compare_score(p, replay_state(c, p['state'], b, cfg), c.name + '/' + s, metrics))
            def identities():
                fixed, joint = row['fixed'], row['joint']
                for parameter in ('r', 'x'):
                    close(fixed['post']['state'][parameter], trained[name][parameter],
                          'fixed theta differs from training', atol=0., rtol=0.)
                require(joint['joint_total'] <= fixed['fixed_total'] + 1e-8 * (1 + abs(fixed['fixed_total'])),
                        'joint retained attainment worse than fixed feasible witness')
                close(fixed['fixed_total'], fixed['post']['loss'], 'fixed total identity')
                close(joint['joint_total'], joint['post']['loss'], 'joint total identity')
                close(fixed['fixed_holdout'], sum(fixed['post']['blocks'][k]['loss'] for k in ('validation', 'future')),
                      'fixed holdout identity')
                close(fixed['future_increment'], fixed['post']['blocks']['future']['loss'], 'fixed future increment identity')
                close(joint['profile_increment'], joint['joint_total'] - joint['pre']['loss'], 'profile increment identity')
                if name in record.get('history_prefits', {}):
                    previous = record['history_prefits'][name]
                    close(fixed['pre']['loss'], previous['fixed']['loss'], 'action-independent fixed history cache')
                    close(joint['pre']['loss'], previous['joint']['loss'], 'action-independent joint history cache')
                    for parameter in STATE_FIELDS:
                        close(joint['pre']['state'][parameter], previous['joint']['state'][parameter],
                              'joint prefit state must not depend on future action', atol=0., rtol=0.)
                require(row['checks']['source_interface'] == 1 and row['checks']['shared_theta_across_all_joint_blocks']
                        and row['checks']['PQ_direct_inputs'] and row['checks']['EV_already_in_total_PQ'],
                        'declared inference invariants differ')
            attempt(key + '/' + name + '/identities', identities)
            _optimizer(candidate, row['joint']['post'], blocks, cfg, metrics, attempt, key + '/' + name + '/joint_post')
        for name, method in methods.items():
            if method.get('action_key') == key:
                attempt('method/' + name, lambda m=method, f=future, k=key: _method(record, m, f, k, metrics))
    for name, method in methods.items():
        if method.get('action_key') not in futures:
            absent('method action states/' + name, method.get('planning_failure') or reason)
    if not errors and not missing and record.get('status') == 'pipeline_failure':
        errors.append(dict(check='pipeline status', reason='failure status with complete states needs inspection'))
    return _result(record, metrics, errors, missing, began)


def _optimizer(candidate, stage, blocks, cfg, metrics, attempt, label):
    optimizer = stage.get('optimizer')
    if optimizer is not None and all(key in optimizer for key in STATE_FIELDS):
        score = stage.get('optimizer_recomputed_score') or optimizer
        attempt(label + '/optimizer', lambda: _compare_score(score,
            replay_state(candidate, optimizer, blocks, cfg), label + '/optimizer', metrics, selected=False))


def _planning(record, methods, metrics):
    selection = record['selection']
    metadata = selection['methods']
    for name, method in methods.items():
        paid = set()
        def pay(node):
            require(node in metadata, 'missing planning dependency ' + node)
            if node in paid:
                return
            paid.add(node)
            for parent in metadata[node].get('dependencies', []):
                pay(parent)
        pay(name)
        own = sum(metadata[node]['planning_actual_seconds'] for node in paid)
        if name not in ('none', 'constant', 'highlow'):
            own += selection['setup_seconds']
        if any(node in ('nearest_fixed', 'gradac_fixed', 'scalar_fixed', 'scalar_hist_fixed', 'scalar_mobile')
               for node in paid):
            own += selection['context_seconds']
        close(method['planning_seconds'], own, name + ': deduplicated planning dependency cost', atol=1e-9, rtol=0.)
        close(metadata[name]['planning_seconds'], own, name + ': metadata planning cost', atol=1e-9, rtol=0.)
        metrics['planning_dependency_checks'] += 1


def _method(record, method, future, key, metrics):
    close(method['power_kw'], future['power_kw'], 'method/action-key power differs', atol=0., rtol=0.)
    require(set(method['inference']) == set(MODES), 'three inference mode denominator')
    costs = record['common_cost']
    for mode in MODES:
        row = method['inference'][mode]
        side = 'joint' if mode == 'joint_total' else 'fixed'
        expected = {name: comp[side][mode] for name, comp in future['comparisons'].items()}
        require(row['losses'] == expected, 'candidate losses differ from retained fit states')
        true = record['true_name']
        margin = min(value for name, value in expected.items() if name != true) - expected[true] if true in expected else None
        require(row['M'] is None if margin is None else row['M'] is not None, 'missing/nonmissing M differs')
        if margin is not None:
            close(row['M'], margin, 'same-loss actual M', atol=1e-9, rtol=1e-11)
        own = sum(comp['cost']['fixed_future_seconds'] +
            (comp['cost']['joint_post_seconds'] if side == 'joint' else 0.) for comp in future['comparisons'].values())
        close(row['seconds'], own, 'inference pays fixed future witness dependencies', atol=1e-9, rtol=0.)
        total = costs['training_seconds'] + method['planning_seconds'] + own + costs['fixed_history_seconds'] + (
            costs['joint_history_seconds'] if side == 'joint' else 0.)
        close(row['total_seconds'], total, 'standalone route cost', atol=1e-9, rtol=0.)
        complete = (len(expected) == len(record['candidates']) and all(
            comp[side]['post']['success'] for comp in future['comparisons'].values()))
        require(row['complete'] == complete, 'inference completeness flag differs from all candidate states')
        metrics['method_mode_checks'] += 1


def _result(record, metrics, errors, missing, began):
    complete = not errors and not missing
    return dict(case_id=record.get('case_id'), seed=record.get('seed'), phase=record.get('phase'),
        index=record.get('index'), raw_status=record.get('status'), raw_failure=record.get('failure'),
        candidate_miss=record.get('candidate_miss'), success=complete, state_review_complete=complete,
        available_checks_passed=not errors, status='failed' if errors else 'incomplete' if missing else 'passed',
        errors=errors, missing=missing, **metrics, seconds=perf_counter() - began)


def source_audit(directory):
    freeze = read(directory / 'freeze.json')
    snapshot = read(directory / 'source_snapshot.json.gz')
    hashes, payloads = freeze['source_sha256'], snapshot['source_base64']
    require(snapshot['source_sha256'] == hashes and set(payloads) == set(hashes), 'snapshot/source coverage')
    for name, expected in hashes.items():
        require(sha256(base64.b64decode(payloads[name], validate=True)).hexdigest() == expected,
                'frozen source payload hash: ' + name)
    return dict(policy_id=freeze['policy_id'], source_sha256=hashes, source_count=len(hashes),
                snapshot_sha256=sha256((directory / 'source_snapshot.json.gz').read_bytes()).hexdigest())


def review_directories(directories, *, expected_count=None):
    began = perf_counter()
    rows, audits, seen = [], [], set()
    for value in directories:
        directory = Path(value)
        audit = None
        try:
            audit = source_audit(directory)
            audits.append(dict(directory=str(directory.resolve()), success=True,
                policy_id=audit['policy_id'], source_count=audit['source_count'], snapshot_sha256=audit['snapshot_sha256']))
        except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
            audits.append(dict(directory=str(directory.resolve()), success=False, reason=type(exc).__name__ + ': ' + str(exc)))
        files = sorted({*directory.glob('case_*.json.gz'), *directory.glob('case_*.json')})
        for path in files:
            try:
                record = read(path)
                identifier = (record.get('phase'), record.get('case_id'), record.get('seed'))
                require(identifier not in seen, 'duplicate phase/case/history record')
                seen.add(identifier)
                if audit is not None:
                    require(record['policy_id'] == audit['policy_id'] and record['source_sha256'] == audit['source_sha256'],
                            'record/freeze identity differs')
                row = review_trial(record)
            except (OSError, EOFError, ValueError, KeyError, TypeError, IndexError, AssertionError,
                    FloatingPointError, np.linalg.LinAlgError) as exc:
                row = dict(success=False, state_review_complete=False, available_checks_passed=False, status='failed',
                    errors=[dict(check='record/schema/replay', reason=type(exc).__name__ + ': ' + str(exc))], missing=[])
            row.update(input_path=str(path.resolve()), input_sha256=sha256(path.read_bytes()).hexdigest())
            rows.append(row)
    tools = [Path(__file__), Path(physical.__file__), HERE / 'likelihood.py', HERE.parent / 'mobile_ev_probing/ac_general_v9.py']
    count_ok = expected_count is None or len(rows) == expected_count
    return dict(schema='systematic_EV_independent_state_review_v1', success=bool(rows and count_ok
        and all(row['success'] for row in rows) and all(a['success'] for a in audits)),
        case_denominator=len(rows), expected_case_count=expected_count, expected_count_matches=count_ok,
        status_counts=dict(Counter(row['status'] for row in rows)),
        phase_counts=dict(Counter(str(row.get('phase')) for row in rows)),
        unique_seed_count=len({row.get('seed') for row in rows if row.get('seed') is not None}),
        saved_states_replayed=sum(row.get('saved_states_replayed', 0) for row in rows),
        AC_block_replays=sum(row.get('AC_block_replays', 0) for row in rows),
        method_mode_checks=sum(row.get('method_mode_checks', 0) for row in rows),
        planning_dependency_checks=sum(row.get('planning_dependency_checks', 0) for row in rows),
        PQ_once_checks=sum(row.get('PQ_once_checks', 0) for row in rows), source_audits=audits,
        independent_tool_sha256={str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in tools},
        cases=rows, seconds=perf_counter() - began,
        scope='Saved-state physical and arithmetic replay only; no new fit, selector or topology probability.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int)
    args = parser.parse_args()
    result = review_directories(args.directory, expected_count=args.expected_count)
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ('success', 'case_denominator', 'status_counts',
        'saved_states_replayed', 'AC_block_replays', 'method_mode_checks', 'PQ_once_checks', 'seconds')}))


if __name__ == '__main__':
    main()
