"""Independent AC, source-QP, price and cost replay for mobile-EV-v2.

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
from . import ev_mobile_cases_v2 as newcases
from . import ev_systematic_cases as oldcases
from . import ev_systematic_scalar_audit as source_qp

HERE = Path(__file__).resolve().parent
METHODS = ('none', 'constant', 'highlow', 'square_fixed', 'square_mobile',
           'scalar_fixed', 'scalar_mobile', 'cap_fixed', 'cap_mobile', 'jointgrad_fixed', 'jointgrad_mobile')
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
        errors.append(dict(check='method denominator', reason='exactly eleven distinct method slots required'))
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
    attempt('operations and planning arithmetic', lambda: audit_planning(record, methods, bank, trained, metrics))
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
            own += selection.get('setup_seconds', 0.)
        if any(node in ('scalar_fixed', 'scalar_mobile', 'cap_fixed', 'cap_mobile', 'jointgrad_fixed', 'jointgrad_mobile')
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
        rivals = [value for name, value in expected.items() if name != true]
        margin = min(rivals) - expected[true] if true in expected and rivals else None
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


def _domains(record):
    cfg=record['config'];case_id=cfg['case_id'];T=cfg['horizon'];n=len(record['training']['p'][0])
    maker=newcases if case_id in newcases.CASES else oldcases
    return {key:maker.service_domain(case_id,T,n,mobile=key=='mobile') for key in ('fixed','mobile')}


def _operation(domain,audit):
    """Reconstruct a saved route and replay it without solving a MILP."""
    ix=domain.indices;x=np.zeros(domain.model.n_variables)
    u=np.asarray(audit['u_kw'],float);T,n=ix.u.shape
    require(u.shape==(T,n),'saved operation power dimensions')
    x[ix.u]=u;x[ix.charge]=np.maximum(u,0.);x[ix.discharge]=np.maximum(-u,0.)
    x[ix.energy]=np.asarray(audit['energy_kwh'],float)
    x[ix.mode]=(np.max(u,axis=1)>1e-10).astype(int)
    for saved in audit['route']:
        matches=[j for j,arc in enumerate(ix.arcs) if vars(arc)==saved]
        require(len(matches)==1,'saved operation arc not in public domain')
        x[ix.arc[matches[0]]]+=1
    replay=domain.check(x)
    require(replay['feasible'],'saved operation route violates public domain: '+str(replay['issues']))
    for name in ('u_kw','energy_kwh','soc','grid_energy_kwh','throughput_kwh',
                 'travel_energy_kwh','conversion_loss_kwh','driver_cost','moves'):
        close(audit[name],replay[name],'operation replay '+name,atol=2e-7,rtol=2e-9)
    return replay


def _output(physical_result):
    return np.column_stack((physical_result['v'],physical_result['master_p'],physical_result['master_q']))


def _sigma(meter,T,n):
    return np.column_stack((_array(meter['v_sigma'],(T,n)),
        _array(meter['master_p_sigma'],(T,)),_array(meter['master_q_sigma'],(T,))))


def _reference(record,bank,trained,a,u):
    """Predict from training and forecast only; actual future records are absent."""
    forecast=record['forecast'];T,n=np.asarray(u).shape;meter=record['future_meter']
    p=np.asarray(forecast['p'])+np.asarray(u)/float(record['config']['base_kw'])
    q=np.asarray(forecast['q']);root=np.asarray(forecast.get('root',forecast.get('root_v')))
    sources=forecast['source_by_candidate'][a]
    value=physical.forward_physics(bank[a],trained[a]['r'],trained[a]['x'],p,q,
        sources['source_p'],sources['source_q'],root)
    obs=dict(p=p+_array(meter['bias_p'],(T,n)),q=q+_array(meter['bias_q'],(T,n)),
        v=value['v']+_array(meter['bias_v'],(T,n)),master_p=value['master_p'],
        master_q=value['master_q'],root_v=root)
    return value,obs,p,q,root


def _fixed_parameter_prices(candidate,state,p,q,root,meter,base_kw,*,step_kw=.01):
    """Independent central physical differences; all times perturb in parallel.

    AC is time-separable, so perturbing column j at every time returns every
    corresponding time-local derivative with just two branch-sweep calls.
    Parameters and source are held at the saved state; no fit is performed.
    """
    T,n=p.shape;sigma=_sigma(meter,T,n);J=np.empty((T,n+2,n))
    sp=np.asarray(state['source_p'])[-T:];sq=np.asarray(state['source_q'])[-T:]
    for j in range(n):
        plus=p.copy();minus=p.copy();plus[:,j]+=step_kw/base_kw;minus[:,j]-=step_kw/base_kw
        upper=physical.forward_physics(candidate,state['r'],state['x'],plus,q,sp,sq,root)
        lower=physical.forward_physics(candidate,state['r'],state['x'],minus,q,sp,sq,root)
        J[:,:,j]=(_output(upper)-_output(lower))/(2*step_kw)/sigma
    return J


def _source_profile(record,profile,bank,trained,metrics):
    names=list(bank);T,n=np.asarray(profile['power_kw']).shape
    expected={(a,b) for a in names for b in names if a!=b}
    require(len(profile['pairs'])==len(expected) and {(p['a'],p['b']) for p in profile['pairs']}==expected,
        'all source-profile directed competitors required')
    require(all(profile.get(k) is False for k in ('uses_truth','uses_validation','uses_future_noise')),
        'source profile uses hidden inputs')
    values=[]
    for pair in profile['pairs']:
        if not pair.get('success'):
            metrics['recorded_failed_predictive_pairs']+=1;continue
        a,b=pair['a'],pair['b'];ref,_,p,q,root=_reference(record,bank,trained,a,profile['power_kw'])
        src=np.asarray(pair['source'],float)
        require(src.shape==(T,2),'source-profile dimensions')
        for j,k in enumerate(('source_p_bounds','source_q_bounds')):
            lo,hi=record['config']['source'][k]
            require(np.all(src[:,j]>=lo-1e-10) and np.all(src[:,j]<=hi+1e-10),'source profile box')
        rival=physical.forward_physics(bank[b],trained[b]['r'],trained[b]['x'],p,q,src[:,0],src[:,1],root)
        close(pair['reference_values'],_output(ref),'predicted source reference mean')
        close(pair['competitor_values'],_output(rival),'fixed-line source competitor mean')
        residual=(_output(rival)-_output(ref))/_sigma(record['future_meter'],T,n)
        close(pair['residual'],residual,'source profile whitened residual')
        value=.5*float(np.sum(residual**2));close(pair['half_SSE'],value,'source profile half-SSE')
        require(pair.get('source_interface')==1 and pair.get('line_parameters_fixed'),
            'source profile physical interface/fixed lines')
        values.append((a,b,value));metrics['source_profile_pair_checks']+=1
    if profile.get('success'):
        require(len(values)==len(expected),'successful source profile lost failed competitors')
    return values


def _joint_profile(record,profile,bank,trained,metrics):
    names=list(bank);K=len(names);cfg=record['config'];u=np.asarray(profile['power_kw'],float)
    T,n=u.shape;H=len(record['training']['p']);base_kw=float(cfg['base_kw'])
    require(all(profile.get(k) is False for k in ('uses_truth','uses_validation','uses_future_noise')),
        'joint predictive profile uses hidden inputs')
    require(profile.get('source_interface')==1 and profile.get('uses_training_history')
        and profile.get('line_parameters_fixed_for_references') and profile.get('one_trained_parameter_vector')
        and profile.get('rival_line_parameters_shared_history_and_prediction'),
        'joint predictive profile declared line/source invariants')
    require(len(profile['references'])==K and {r['a'] for r in profile['references']}==set(names),
        'joint predictive reference denominator')
    prices={};losses={};all_success=True
    for reference in profile['references']:
        a=reference['a'];require(set(reference['fits'])==set(names),'every reference must retain all K fits')
        ref,obs,p,q,root=_reference(record,bank,trained,a,u)
        reference_state={key:trained[a][key] for key in ('r','x')}
        reference_state.update(record['forecast']['source_by_candidate'][a])
        Ja=_fixed_parameter_prices(bank[a],reference_state,p,q,root,record['future_meter'],base_kw)
        for b,row in reference['fits'].items():
            all_success=all_success and bool(row.get('success') and row.get('gradient_reliable'))
            if not (row.get('success') and row.get('gradient_reliable')):
                metrics['recorded_failed_predictive_pairs']+=1
            if row.get('state') is None:
                require(not profile.get('success') and not row.get('success'),'successful predictive fit missing its state')
                continue
            state=_state(row['state'])
            require(state['r'].shape==(bank[b].m,) and state['x'].shape==(bank[b].m,),
                'predictive fit must share one R/X vector across H and future')
            require(state['source_p'].shape==(H+T,) and state['source_q'].shape==(H+T,),
                'predictive source spans training and predicted future only')
            blocks=(('training',record['training'],record['training_meter']),('future',obs,record['future_meter']))
            replay=replay_state(bank[b],state,blocks,cfg)
            _compare_score(row,replay,'joint predictive '+a+'__to__'+b,metrics,selected=bool(row.get('success')))
            if row.get('optimizer') is not None and row.get('optimizer_recomputed_score') is not None:
                _compare_score(row['optimizer_recomputed_score'],replay_state(bank[b],row['optimizer'],blocks,cfg),
                    'joint predictive optimizer '+a+'__to__'+b,metrics,selected=False)
            Jb=_fixed_parameter_prices(bank[b],state,p,q,root,record['future_meter'],base_kw)
            residual=replay['blocks']['future']['residual']
            price=np.einsum('tij,ti->tj',Jb-Ja,residual)
            saved=np.asarray(row['gradient_per_kw'],float)
            close(saved,price,'joint predictive envelope price physical finite difference',atol=2e-5,rtol=2e-5)
            error=float(np.max(abs(saved-price)/(1+abs(price))))
            metrics['max_envelope_price_scaled_error']=max(metrics['max_envelope_price_scaled_error'],error)
            require(row.get('future_reference_derivative_subtracted'),'reference mean derivative missing')
            prices[a,b]=saved;losses[a,b]=replay['loss'];metrics['joint_predictive_fit_checks']+=1
    expected={(a,b) for a in names for b in names if a!=b}
    require(len(profile['pairs'])==len(expected) and {(p['a'],p['b']) for p in profile['pairs']}==expected,
        'joint predictive directed task denominator')
    for pair in profile['pairs']:
        a,b=pair['a'],pair['b']
        if (a,b) not in losses or (a,a) not in losses:
            require(not pair.get('success'),'successful pair lacks required shared-line fits');continue
        half=.5*(losses[a,b]-losses[a,a])
        close(pair['half_gap'],half,'shared-line signed predictive half-gap')
        close(pair['own_loss'],losses[a,a],'reference attained own loss')
        close(pair['rival_loss'],losses[a,b],'rival attained loss')
        close(pair['gradient_per_kw'],prices[a,b]-prices[a,a],'signed half-gap envelope subtraction')
    if profile.get('success'):
        require(all_success and len(losses)==K*K,'complete joint profile missing a required local fit')
        pairs=profile['pairs'];nearest=[];groups=[]
        for a in names:
            rows=[p for p in pairs if p['a']==a];value=min(p['half_gap'] for p in rows)
            active=[p for p in rows if np.isclose(p['half_gap'],value,atol=1e-9,rtol=1e-9)]
            nearest.append(value);groups.append(active)
        score=float(np.mean(nearest));close(profile['score'],score,'joint complete score')
        agg=profile['aggregation'];close(agg['score'],score,'joint aggregation score')
        require(agg.get('success') and len(agg['active_groups'])==K,'joint active reference groups')
        for expected_group,saved_group in zip(groups,agg['active_groups']):
            require({(p['a'],p['b']) for p in expected_group}=={(p['a'],p['b']) for p in saved_group},'joint nearest active competitors')
            lookup={(p['a'],p['b']):p for p in expected_group}
            for saved_pair in saved_group:
                actual_pair=lookup[saved_pair['a'],saved_pair['b']]
                close(saved_pair['half_gap'],actual_pair['half_gap'],'active task half-gap')
                close(saved_pair['gradient_per_kw'],actual_pair['gradient_per_kw'],'active task price copy')
        mean=np.mean([np.mean([p['gradient_per_kw'] for p in group],axis=0) for group in groups],axis=0)
        close(agg['gradient_per_kw'],mean,'joint aggregate active price')
    metrics['joint_profile_checks']+=1


def _profile_calls(raw,joint):
    calls=[row['profile'] for row in raw.get('initial_candidates',[]) if 'profile' in row]
    if joint:
        calls.extend(row['raw_AC_joint_profile'] for start in raw.get('starts',[])
            for row in start.get('iterations',[]) if 'raw_AC_joint_profile' in row)
    else:
        calls.extend(row['raw_AC_profile'] for row in raw.get('grids',[]) if 'raw_AC_profile' in row)
    return calls


def _cache_costs(metadata,metrics):
    """Rebuild the actual-call sequence; cached profiles retain their old time."""
    for joint,names in ((False,('scalar_fixed','scalar_mobile','cap_fixed','cap_mobile')),
                        (True,('jointgrad_fixed','jointgrad_mobile'))):
        cache={}
        for name in names:
            raw=metadata[name].get('raw',{});counters=raw.get('counters')
            if counters is None:continue
            calls=_profile_calls(raw,joint);new=[];hits=0
            for profile in calls:
                key=np.asarray(profile['power_kw'],dtype='<f8').tobytes()
                if key in cache:
                    hits+=1
                    require(profile==cache[key],'cached predictive profile differs from its original paid value')
                else:
                    new.append(profile)
                    if not profile.get('deadline_reached'):cache[key]=profile
            if joint:
                require(counters['profile_evaluations']==len(new) and counters['cache_hits']==hits,'joint cache call counts')
                close(counters['actual_profile_seconds'],sum(p['seconds'] for p in new),'joint incremental profile seconds',atol=1e-9,rtol=0.)
                require(counters['joint_fit_calls']==sum(p['joint_fit_calls'] for p in new),'joint incremental fit calls')
                require(counters['fixed_source_witness_fit_calls']==sum(p['fixed_source_witness_fit_calls'] for p in new),'joint incremental witness calls')
                events=counters['payment_log'];require(len(events)==len(new),'joint payment event denominator')
                for event,profile in zip(events,new):
                    close(event['seconds'],profile['seconds'],'joint payment elapsed time',atol=1e-9,rtol=0.)
                    require(event['joint_fit_calls']==profile['joint_fit_calls'] and event['success']==profile['success'],'joint payment fit/status')
            else:
                require(counters['profile_evaluations']==len(new) and counters['profile_cache_hits']==hits,'source cache call counts')
                close(counters['actual_profile_seconds'],sum(p['seconds'] for p in new),'source incremental profile seconds',atol=1e-9,rtol=0.)
                require(counters['pair_fit_calls']==sum(p['paid_pair_fits'] for p in new),'source incremental fit calls')
                events=counters['cache_payment_log'];require(len(events)==len(calls),'source cache payment event denominator')
                require(sum(e['cache_hit'] for e in events)==hits,'source cache hit event count')
                close(sum(e['actual_seconds'] for e in events),counters['actual_profile_seconds'],'source payment event elapsed time',atol=1e-9,rtol=0.)
            require(counters['actual_profile_seconds']<=metadata[name]['planning_actual_seconds']+1e-6,
                'planning own elapsed cost omits paid profile computation')
            metrics['incremental_cache_cost_checks']+=1


def _gradient_policy(raw,domain,method_power,metrics):
    if not raw.get('success'):return
    initials={p['origin']:p for p in raw['initial_candidates'] if p.get('success') and 'profile' in p}
    require(initials,'successful joint policy needs a complete initial')
    initial=max(p['score'] for p in initials.values());close(raw['initial_best_score'],initial,'joint initial best')
    candidate_scores=[initial]
    for start in raw.get('starts',[]):
        require(start['origin'] in initials,'gradient start lacks initial full score')
        current=initials[start['origin']]['profile'];close(start['initial_score'],current['score'],'gradient start score')
        if str(start.get('failure','')).startswith('zero local price'):
            maximum=max(float(np.max(abs(p['gradient_per_kw']))) for group in current['aggregation']['active_groups'] for p in group)
            require(maximum<1e-12 and not start['iterations'],'unjustified zero-price start stop')
        for row in start.get('iterations',[]):
            close(row['before_score'],current['score'],'gradient current complete score')
            solved=row['raw'];metrics['gradient_MILP_checks']+=1
            groups=current['aggregation']['active_groups']
            if not solved.get('success'):
                require(not row['accepted'],'failed gradient MILP accepted');continue
            x=np.asarray(solved['x'],float);audit=domain.check(x)
            require(audit['feasible'],'gradient proposal operations')
            power=np.asarray(solved['power_kw']);close(power,x[domain.indices.u],'gradient proposal power projection')
            delta=power-np.asarray(current['power_kw']);change=float(np.max(abs(delta)))
            close(row['proposal_max_delta_kw'],change,'gradient proposal change')
            trust=float(solved['trust_kw']);require(change<=trust+1e-7,'gradient trust region')
            directional=float(np.mean([min(float(np.sum(np.asarray(p['gradient_per_kw'])*delta)) for p in group) for group in groups]))
            close(solved['attained_directional_gain'],directional,'gradient attained linear objective',atol=2e-7,rtol=2e-8)
            predicted=solved.get('predicted_directional_gain');bound=solved.get('linear_best_bound')
            if predicted is not None:require(predicted<=directional+2e-6*(1+abs(directional)),'gradient auxiliary objective exceeds actual active min')
            if bound is not None:require(directional<=bound+2e-6*(1+abs(bound)),'gradient attained value exceeds solver upper bound')
            require(solved.get('full_AC_global_gap') is None,'linear MILP bound mislabeled as AC bound')
            if row.get('repeated_AC_score_avoided'):
                require(change<=1e-10 or (bound is not None and bound<=1e-10*(1+abs(current['score']))),'unjustified zero-step/first-order stop')
                require(not row['accepted'] and 'raw_AC_joint_profile' not in row,'stopped step scored or accepted')
            elif 'raw_AC_joint_profile' in row:
                trial=row['raw_AC_joint_profile'];close(trial['power_kw'],power,'gradient AC proposal inputs')
                should=bool(trial.get('success') and trial['score']>current['score']+1e-9+1e-10*abs(current['score']))
                require(row['accepted']==should,'joint AC acceptance rule')
                if should:current=trial;candidate_scores.append(current['score'])
            else:require(not row['accepted'],'accepted gradient step lacks AC predictive score')
            close(row['selected_score'],current['score'],'gradient retained complete score')
    close(raw['selected_score'],max(candidate_scores),'gradient best score across starts')
    close(raw['selected_profile']['score'],raw['selected_score'],'gradient selected profile score')
    close(raw['selected_profile']['power_kw'],method_power,'gradient final action profile')
    require(raw['selected_score']>=initial-1e-10 and raw['inheritance_protected'],'gradient inherited plan protection')
    attempts=[row for start in raw.get('starts',[]) for row in start.get('iterations',[])]
    require(raw.get('gradient_MILP_calls')==len(attempts) and raw.get('gradient_MILP_accepted')==sum(row['accepted'] for row in attempts),
        'gradient MILP attempt/acceptance counters')


def _proxy(record,raw,*,cap,metrics):
    pairs=record['selection']['frozen_pairs'];names=[r['name'] for r in record['candidates']]
    pair_names=[tuple(p['name'].split('__to__')) for p in pairs];n=len(record['training']['p'][0])
    for grid in raw.get('grids',[]):
        table=grid['table']
        if not table.get('success'):
            metrics['recorded_incomplete_proxy_tables']+=1;continue
        metrics['proxy_table_checks']+=1;T=len(table['background'][0]);errors=[]
        for k,pair in enumerate(pairs):
            c,B,G=(np.asarray(pair[v],float) for v in ('c','B','A0'))
            low,high,center=(np.asarray(pair[v],float) for v in ('lower','upper','base_center'))
            error=0.
            for t in range(T):
                rows=np.r_[np.arange(t*n,(t+1)*n),T*n+t,T*n+T+t];cols=np.arange(2*t,2*t+2)
                ct,Gt=c[rows],G[np.ix_(rows,cols)];lo,hi,base=low[cols],high[cols],center[cols]
                bounds=[table['background'][k][t]['primal_dual_gap']]
                for j,site in enumerate(table['tables'][k][t]):
                    b=B[rows,t*n+j];params=site['curve_parameters']
                    for key,value in (('c',ct),('b',b),('G',Gt),('source_lower',lo),('source_upper',hi),('source_center',base)):
                        close(params[key],value,'saved scalar curve '+key)
                    for node in site['nodes']:
                        source_qp.node_check(ct,b,Gt,lo,hi,base,node,params['power_range'])
                        metrics['proxy_endpoint_checks']+=1
                        metrics['recorded_failed_proxy_QPs']+=not node.get('success',False)
                    if j==0:
                        zero=[node for node in site['nodes'] if node['p']==0.]
                        require(len(zero)==1 and table['background'][k][t]==zero[0],
                            'proxy zero-action background differs from its audited zero node')
                    curvature=float(b@b);spacing=float(np.max(np.diff(site['grid']),initial=0.))
                    gap=max(node['primal_dual_gap'] for node in site['nodes']);bound=curvature*spacing**2/8+gap
                    close(site['error_upper'],bound,'proxy scalar chord error')
                    close(site['primal_upper'],[p['primal_upper'] for p in site['nodes']],'proxy endpoint upper values')
                    bounds.append(bound)
                error+=max(bounds)
            errors.append(error)
        close(table['pair_error_upper'],errors,'proxy task chord errors')
        solved=grid.get('raw')
        if solved is None:continue
        metrics['proxy_MILP_checks']+=1;action=solved.get('action_QP')
        qps=action if isinstance(action,list) else action.get('pairs',[]) if isinstance(action,dict) else []
        if not qps:
            require(solved.get('frozen_objective_lower') is None,'invalid raw action has lower certificate')
            metrics['recorded_invalid_proxy_actions']+=1;continue
        require(len(qps)==len(pairs),'proxy action QP task denominator')
        lower=[];upper=[]
        for pair,qp in zip(pairs,qps):
            c,B,G=(np.asarray(pair[v],float) for v in ('c','B','A0'))
            lo,hi,base=(np.asarray(pair[v],float) for v in ('lower','upper','base_center'))
            primal,dual,_=source_qp.qp_check(c,G,lo,hi,base,B@np.asarray(solved['power_kw']).ravel(),qp)
            lower.append(.5*dual);upper.append(.5*primal);metrics['proxy_action_QP_checks']+=1
            metrics['recorded_failed_proxy_QPs']+=not qp.get('success',False)
        history=raw.get('history_losses') or {}
        def aggregate(values):
            return float(np.mean(np.minimum(values,cap))) if cap is not None else source_qp._aggregate(names,pair_names,values,history)
        lb,primal=aggregate(lower),aggregate(upper);close(solved['frozen_objective_lower'],lb,'frozen proxy lower')
        if isinstance(action,dict):
            close(action['dual_lower'],lb,'scalar all-pair dual lower');close(action['primal_upper'],primal,'scalar all-pair feasible upper')
        bound=solved.get('frozen_objective_upper')
        if bound is not None:
            require(lb<=primal+2e-7 and primal<=bound+2e-7+2e-8*abs(bound),'frozen proxy interval direction')
            close(solved['frozen_absolute_gap'],max(0.,bound-lb),'frozen proxy gap')
            if solved.get('solver_proxy_incumbent') is not None:
                require(solved['solver_proxy_incumbent']<=bound+2e-7+2e-8*abs(bound),'proxy incumbent/bound order')
        if cap is not None:
            close(solved['capped_chord_error_upper'],np.mean(np.minimum(cap,errors)),'capped chord error propagation')
        require(solved.get('full_AC_global_gap') is None,'proxy bound mislabeled as AC global bound')


def audit_planning(record,methods,bank,trained,metrics):
    defaults=dict(source_profile_pair_checks=0,joint_profile_checks=0,joint_predictive_fit_checks=0,
        max_envelope_price_scaled_error=0.,recorded_failed_predictive_pairs=0,incremental_cache_cost_checks=0,
        gradient_MILP_checks=0,proxy_table_checks=0,proxy_endpoint_checks=0,proxy_MILP_checks=0,
        proxy_action_QP_checks=0,recorded_invalid_proxy_actions=0,recorded_incomplete_proxy_tables=0,operation_replay_checks=0)
    defaults['recorded_failed_proxy_QPs']=0
    for key,value in defaults.items():metrics.setdefault(key,value)
    selection=record['selection'];require(all(selection.get(k) is False for k in ('uses_truth','uses_validation','uses_future_noise')),
        'planning selection uses hidden inputs')
    n=len(record['training']['p'][0]);domains=_domains(record)
    require(len(bank)==len(record['candidates']),'candidate denominator')
    for row in record['candidates']:
        candidate=bank[row['name']]
        require(tuple(map(tuple,row['supports']))==candidate.supports,'complete ordered edge supports required')
    seen_profiles=set()
    for name,method in methods.items():
        domain=domains['mobile' if name.endswith('mobile') else 'fixed'];metadata=selection['methods'][name]
        if name=='none':close(method['power_kw'],np.zeros_like(method['power_kw']),'none action zero')
        elif method['operation'].get('success'):
            replay=_operation(domain,method['operation']['audit'])
            close(replay['u_kw'],method['power_kw'],'operation/method power')
            close(method['operation']['driver_cost'],replay['driver_cost'],'saved driver cost')
            metrics['operation_replay_checks']+=1
        raw=metadata.get('raw',{})
        for profile in _profile_calls(raw,name.startswith('jointgrad'))+([raw['selected_profile']] if 'selected_profile' in raw else []):
            digest=sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if digest in seen_profiles:continue
            seen_profiles.add(digest)
            if 'references' in profile:_joint_profile(record,profile,bank,trained,metrics)
            else:_source_profile(record,profile,bank,trained,metrics)
        if name.startswith(('scalar','cap')):
            cap=raw.get('cap') if name.startswith('cap') else None
            if name.startswith('cap'):
                parent=selection['methods']['scalar_mobile' if name.endswith('mobile') else 'scalar_fixed'].get('raw',{})
                for grid in raw.get('grids',[]):
                    if grid.get('table_inherited_from_scalar_parent'):
                        candidates=[g['table'] for g in parent.get('grids',[]) if g['grid_points']==grid['grid_points']]
                        require(len(candidates)==1 and candidates[0]==grid['table'],'inherited cap table differs from its paid scalar parent')
            _proxy(record,raw,cap=cap,metrics=metrics)
            if raw.get('success'):
                values=[p['half_SSE'] for p in raw['selected_profile']['pairs']]
                if cap is None:
                    pair_names=[(p['a'],p['b']) for p in raw['selected_profile']['pairs']]
                    score=source_qp._aggregate(list(bank),pair_names,values,raw.get('history_losses') or {})
                else:score=float(np.mean(np.minimum(values,cap)))
                close(raw['selected_score'],score,'source-objective selected score')
                close(raw['selected_profile']['power_kw'],method['power_kw'],'source-objective selected curve')
        if name.startswith('jointgrad'):_gradient_policy(raw,domain,method['power_kw'],metrics)
    _cache_costs(selection['methods'],metrics)


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
    tools = [Path(__file__), Path(physical.__file__), Path(source_qp.__file__),
        HERE/'ev_systematic_review.py',HERE/'ev_mobile_cases_v2.py',HERE/'ev_systematic_cases.py',
        HERE/'ev_mobile_service_domain.py',HERE/'likelihood.py',HERE.parent/'mobile_ev_probing/ac_general_v9.py']
    count_ok = expected_count is None or len(rows) == expected_count
    return dict(schema='mobile_EV_v2_independent_state_review', success=bool(rows and count_ok
        and all(row['success'] for row in rows) and all(a['success'] for a in audits)),
        case_denominator=len(rows), expected_case_count=expected_count, expected_count_matches=count_ok,
        status_counts=dict(Counter(row['status'] for row in rows)),
        phase_counts=dict(Counter(str(row.get('phase')) for row in rows)),
        unique_seed_count=len({row.get('seed') for row in rows if row.get('seed') is not None}),
        saved_states_replayed=sum(row.get('saved_states_replayed', 0) for row in rows),
        AC_block_replays=sum(row.get('AC_block_replays', 0) for row in rows),
        method_mode_checks=sum(row.get('method_mode_checks', 0) for row in rows),
        planning_dependency_checks=sum(row.get('planning_dependency_checks', 0) for row in rows),
        source_profile_pair_checks=sum(row.get('source_profile_pair_checks',0) for row in rows),
        joint_profile_checks=sum(row.get('joint_profile_checks',0) for row in rows),
        joint_predictive_fit_checks=sum(row.get('joint_predictive_fit_checks',0) for row in rows),
        max_envelope_price_scaled_error=max((row.get('max_envelope_price_scaled_error',0.) for row in rows),default=0.),
        incremental_cache_cost_checks=sum(row.get('incremental_cache_cost_checks',0) for row in rows),
        gradient_MILP_checks=sum(row.get('gradient_MILP_checks',0) for row in rows),
        proxy_endpoint_checks=sum(row.get('proxy_endpoint_checks',0) for row in rows),
        proxy_action_QP_checks=sum(row.get('proxy_action_QP_checks',0) for row in rows),
        operation_replay_checks=sum(row.get('operation_replay_checks',0) for row in rows),
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
