"""History-only prediction and explanation-domain sensitivity diagnostics.

This audit is outside the frozen calibrated policy. It evaluates already
planned controls, plus at most two MILP variants per permitted history. It
never accesses evaluator truth, seed, or realized future measurement arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    from .adaptive_history import (build_history_context, build_frozen_pairs,
        evaluate_frozen_pair_actual, _predict_path, _forecast_specs, _future_for_center)
    from .ev_information_milp import FrozenPair, solve_pair, adaptive_search
    from .ev_operations_milp import EVConfig, check_plan
    from .likelihood import Observations
    from .candidates import RadialCandidate
    from .simulation import model, BASE_KW
    from .adaptive_result_audit import _dump, _csv
except ImportError:
    from research_experiments.mobile_ev_mle.adaptive_history import (
        build_history_context, build_frozen_pairs, evaluate_frozen_pair_actual,
        _predict_path, _forecast_specs, _future_for_center)
    from research_experiments.mobile_ev_mle.ev_information_milp import (
        FrozenPair, solve_pair, adaptive_search)
    from research_experiments.mobile_ev_mle.ev_operations_milp import EVConfig, check_plan
    from research_experiments.mobile_ev_mle.likelihood import Observations
    from research_experiments.mobile_ev_mle.candidates import RadialCandidate
    from research_experiments.mobile_ev_mle.simulation import model, BASE_KW
    from adaptive_result_audit import _dump, _csv


SOURCE_RADII = (.02, .05, .15)
LINE_RADII = (.1, .3, .6)
CENTRE_COUNTS = (1, 2)
METHODS = ('none', 'constant_fixed', 'fixed_pulse', 'random_action',
           'legacy_library_proxy', 'fixed_mle_milp', 'refit_milp')
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q', 'root_v')
INPUT_POWER_TOLERANCE_KW = 2e-6


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype='<f8')
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def history_input_hash(history):
    return _hash_arrays(*[getattr(history, key) for key in CHANNELS])


def history_seed(history):
    """A deterministic controller seed derived entirely from present meters."""
    return int(history_input_hash(history)[:8], 16)


def load_history_actions(trial_file):
    """Copy only H channels and already planned EV P/Q/sites from archives."""
    trial_file = Path(trial_file)
    saved = json.loads(trial_file.read_text(encoding='utf-8'))
    actions = {}
    for record in saved['records']:
        method = str(record['method'])
        if method not in METHODS or method in actions:
            raise ValueError('seven unique one-round methods are required by this audit')
        raw = record['action']
        p, q = np.array(raw['p'], float), np.array(raw['q'], float)
        sites = np.array(raw['sites'], int)
        if (p.ndim != 2 or p.shape[1] != 3 or q.shape != p.shape or
                sites.shape != (len(p),) or not np.isfinite(p).all() or
                not np.isfinite(q).all() or np.any(p < -INPUT_POWER_TOLERANCE_KW/BASE_KW)
                or np.any(q != 0)):
            raise ValueError('this first audit requires finite charging-only P and zero Q')
        raw_u=p*BASE_KW;canonical=np.maximum(raw_u,0.)
        correction=canonical-raw_u
        warning=dict(power_tolerance_kw=INPUT_POWER_TOLERANCE_KW,
            power_tolerance_pu=INPUT_POWER_TOLERANCE_KW/BASE_KW,
            raw_minimum_kw=float(raw_u.min()),negative_coordinate_count=int(np.sum(raw_u<0)),
            raw_control_hash=_hash_arrays(raw_u),AC_canonical_control_hash=_hash_arrays(canonical),
            AC_correction_l2_kw=float(np.linalg.norm(correction)),
            AC_correction_linf_kw=float(np.max(abs(correction),initial=0.)),
            proxy_uses_raw_control=True,AC_only_control_canonicalized=bool(np.any(raw_u<0)),
            original_EV_trial_modified=False)
        actions[method] = dict(u_kw=raw_u,raw_u_kw=raw_u.copy(),canonical_AC_u_kw=canonical,
            q_kw=q*BASE_KW,sites=sites,numerical_input_audit=warning)
    if set(actions) != set(METHODS):
        raise ValueError('missing required method; no difficult action may be removed')
    archive = trial_file.with_name(trial_file.stem+'_observations.npz')
    with np.load(archive, allow_pickle=False) as values:
        historical = {key: np.array(values['history_'+key]) for key in CHANNELS}
    history = Observations(**historical)
    if history.n != 3 or len({value['u_kw'].shape for value in actions.values()}) != 1:
        raise ValueError('consistent three-terminal horizon required')
    # saved evaluator fields and future arrays are not returned or accessed.
    return history, actions


def complete_bank():
    return [RadialCandidate.from_supports(name, 3, supports) for name, supports in
        [('star', ()), ('clade_01', ((0,1),)), ('clade_02', ((0,2),)), ('clade_12', ((1,2),))]]


def matrix_hash(pair):
    return _hash_arrays(pair.c, pair.B, pair.A, pair.F, pair.scale)


def box_hash(pair):
    return _hash_arrays(pair.lower, pair.upper)


def restrict_pair(context, raw, broad_pair, *, line_radius, source_radius):
    """Intersect the broad box; leave all residual matrices and H states fixed."""
    if not 0 <= line_radius <= raw['metadata']['line_trust_fraction']:
        raise ValueError('line radius must be nested within the frozen broad domain')
    if not 0 <= source_radius <= raw['metadata']['source_delta_bound']:
        raise ValueError('source radius must be nested within the frozen broad domain')
    md = raw['metadata']; free = np.asarray(md['free_indices'], int)
    lower, upper = broad_pair.lower.copy(), broad_pair.upper.copy()
    for name, index, layout in zip(md['candidate_names'], md['centre_indices'], md['variable_layout']):
        centre = context['centres'][name][index]
        a, b = layout['beta']; active = (free>=a)&(free<b)
        positions = free[active]-a
        beta = np.asarray(centre['beta'])[positions]
        lower[active] = np.maximum(lower[active], -line_radius*beta)
        upper[active] = np.minimum(upper[active], line_radius*beta)
        a, b = layout['future_source']; active = (free>=a)&(free<b)
        lower[active] = np.maximum(lower[active], -source_radius)
        upper[active] = np.minimum(upper[active], source_radius)
    metadata = dict(md, line_trust_fraction=float(line_radius), source_delta_bound=float(source_radius),
                    matrix_frozen_for_domain_sensitivity=True)
    data = dict(raw, lower=lower, upper=upper, metadata=metadata)
    pair = FrozenPair(**data)
    if matrix_hash(pair) != matrix_hash(broad_pair):
        raise RuntimeError('nested domain comparison changed a frozen matrix')
    # Factorization keys include the full C, scale, and free-column partition;
    # sharing this per-history storage preserves the same QP and saves QR work.
    pair._solver_cache['history_qp_qr'] = broad_pair._solver_cache.setdefault('history_qp_qr', {})
    return data, pair


def source_compensation(context, raw, pair, z):
    md = raw['metadata']; full = np.asarray(md['fixed_values'], float).copy()
    full[np.asarray(md['free_indices'], int)] = z
    free = np.asarray(md['free_indices'], int)
    rows = []
    for name, index, layout in zip(md['candidate_names'], md['centre_indices'], md['variable_layout']):
        a, b = layout['future_source']; step = full[a:b]
        ids = np.flatnonzero((free>=a)&(free<b))
        if len(step):
            saturation = np.isclose(z[ids], pair.lower[ids], rtol=0, atol=1e-7) | np.isclose(
                z[ids], pair.upper[ids], rtol=0, atol=1e-7)
            delta = step.reshape(-1,2)
        else:
            saturation, delta = np.zeros(0, bool), np.zeros((0,2))
        a, b = layout['beta']; centre_beta = np.asarray(context['centres'][name][index]['beta'])
        beta_step = full[a:b]
        rows.append(dict(candidate=name, centre_index=index, source_node=1,
            source_step_l2=float(np.linalg.norm(step)), source_step_linf=float(np.max(abs(step), initial=0)),
            source_p_step=delta[:,0].tolist(), source_q_step=delta[:,1].tolist(),
            source_bound_active_count=int(saturation.sum()), source_coordinate_count=len(step),
            beta_step_relative_linf=float(np.max(abs(beta_step)/centre_beta, initial=0))))
    return rows


def solve_controls(context, raw_pairs, pairs, controls, *, rollback_active=True):
    """Keep every required pair; a failed pair invalidates the global minimum."""
    rows = []; cache = {}
    for method, action in controls.items():
        u = np.asarray(action['u_kw'], float)
        key = np.ascontiguousarray(u, dtype='<f8').tobytes()
        if key not in cache:
            solved=[]
            for pair in pairs:
                try:
                    solved.append(solve_pair(pair, u))
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                    solved.append(dict(success=False,error=str(error),q_lower=None,q_upper=None))
            cache[key] = solved
        else:
            solved = cache[key]
        ledger = []
        for raw, pair, answer in zip(raw_pairs, pairs, solved):
            if answer.get('q_lower') is None:
                ledger.append(dict(pair=pair.name,success=False,error=answer.get('error'),
                    q_lower=None,q_upper=None,matrix_hash=matrix_hash(pair),box_hash=box_hash(pair)))
                continue
            if answer['q_lower'] > answer['q_upper']+1e-7*max(1., answer['q_upper']):
                raise RuntimeError('invalid inner QP lower > upper')
            start = raw['metadata']['future_row_start']
            residual = answer['residual']
            ledger.append(dict(pair=pair.name, success=answer['success'],
                q_lower=answer['q_lower'], q_upper=answer['q_upper'], numeric_gap=answer['numeric_gap'],
                affine_history_cost=float(residual[:start]@residual[:start]),
                affine_future_distance_squared=float(residual[start:]@residual[start:]),
                matrix_hash=matrix_hash(pair), box_hash=box_hash(pair),
                acceleration=answer.get('acceleration'),
                source_compensation=source_compensation(context, raw, pair, answer['z'])))
        failure = [item['pair'] for item in ledger if not item['success']]
        result = dict(method=method, status='required_QP_failure' if failure else 'complete',
            required_pair_count=len(pairs), failed_pairs=failure, pairs=ledger,
            q_lower=None if failure else min(answer['q_lower'] for answer in solved),
            q_upper=None if failure else min(answer['q_upper'] for answer in solved),
            controls_hash=_hash_arrays(u), history_removed_from_objective=False)
        if not failure:
            index = min(range(len(solved)), key=lambda i:solved[i]['q_upper'])
            result.update(active_pair=pairs[index].name, active_affine=ledger[index])
            if rollback_active:
                try:
                    if np.any(u < -INPUT_POWER_TOLERANCE_KW):
                        raise ValueError('genuine discharge cannot use the charging-only AC replay')
                    canonical=np.maximum(u,0.)
                    actual = evaluate_frozen_pair_actual(context, raw_pairs[index], canonical, solved[index]['z'])
                    delta=canonical-u
                    actual.update(raw_proxy_control_hash=_hash_arrays(u),
                        AC_control_hash=_hash_arrays(canonical),
                        AC_control_correction_l2_kw=float(np.linalg.norm(delta)),
                        AC_control_correction_linf_kw=float(np.max(abs(delta),initial=0.)),
                        affine_response_correction_norm=float(np.linalg.norm(pairs[index].B@delta.ravel())),
                        raw_control_affine_cost=solved[index]['q_upper'],
                        original_EV_trial_modified=False)
                    result['active_AC_witness'] = {k:v for k,v in actual.items()
                        if k not in ('actual_residual','affine_residual')}
                except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                    result['active_AC_witness'] = dict(status='required_AC_replay_failure',
                        error=str(error), physically_feasible=False, nonlinear_infimum_computed=False)
            result['AC_rollback_scope'] = 'active affine pair witness only; not all nonlinear competitors'
        rows.append(result)
    return rows


def scenario_context(context, specs):
    """Change declared future paths while retaining the exact same H centres."""
    specs = _forecast_specs(specs)
    history, meter, horizon = context['history'], context['model'], context['config']['horizon']
    p = history.p-np.broadcast_to(meter.bias_p, history.p.shape)
    q = history.q-np.broadcast_to(meter.bias_q, history.q.shape)
    scenarios = [dict(spec, p0=_predict_path(p, horizon, spec['baseline']),
        q0=_predict_path(q, horizon, spec['baseline']),
        root_v=np.full(horizon, np.mean(history.root_v))) for spec in specs]
    return {**context, 'scenarios':scenarios,
            'summary':{**context['summary'],'scenario_specs':specs,
                       'forecast_is_coverage_guarantee':False}}


def default_models():
    return {
        'mean': [dict(name='mean')],
        'ols_trend': [dict(name='ols_trend',baseline='ols_trend',source_baseline='ols_trend')],
        'tracking_095': [dict(name='tracking_095',tracking_gain=.95)],
        'tracking_105': [dict(name='tracking_105',tracking_gain=1.05)],
        'source_shift_minus': [dict(name='source_shift_minus',source_shift_p=-.05,source_shift_q=-.02)],
        'source_shift_plus': [dict(name='source_shift_plus',source_shift_p=.05,source_shift_q=.02)],
        'mean_and_ols': [dict(name='mean'),dict(name='ols_trend',baseline='ols_trend',source_baseline='ols_trend')],
    }


def nested_checks(domains):
    """Wide-domain lower > narrow-domain upper detects a true bound conflict."""
    rows=[]
    for small, large in product(domains, domains):
        s, l = small['domain'], large['domain']
        if s==l or not (s['line_radius']<=l['line_radius'] and s['source_radius']<=l['source_radius']
                       and s['centres']<=l['centres']):
            continue
        narrow = {row['method']:row for row in small['controls']}
        wide = {row['method']:row for row in large['controls']}
        for method in narrow:
            if method not in wide:
                rows.append(dict(method=method,small=s,large=l,checked=False,
                    invalid=False,reason='required wide-domain evaluation failed'))
                continue
            a,b=narrow[method],wide[method]
            complete=a['status']=='complete' and b['status']=='complete'
            invalid=complete and b['q_lower']>a['q_upper']+1e-7*max(1.,a['q_upper'],b['q_lower'])
            rows.append(dict(method=method,small=s,large=l,checked=complete,
                wide_lower=b.get('q_lower'),narrow_upper=a.get('q_upper'),invalid=bool(invalid),
                theorem='minimum over an enlarged fixed-matrix explanation domain cannot increase'))
    return rows


def _operations(horizon):
    access=np.ones((horizon,3),bool);access[:2,2]=False
    return EVConfig(n_slots=horizon,n_sites=3,access=access)


def _search(context, raw, pairs, controls, u0):
    config=_operations(context['config']['horizon'])
    result=adaptive_search(config,pairs,initial_us=[u0,np.zeros_like(u0)],
        max_iterations=3,segments=2,max_segments=8,solver_seconds=3.,total_seconds=25.,
        absolute_gap=1e-4,relative_gap=1e-3,seed=history_seed(context['history']))
    record={k:v for k,v in result.items() if k!='plan'}
    record.update(device_config=asdict(config),calibrated_policy_component=False,
                  complete_EV_action_enumeration=False)
    if result.get('plan') is not None:
        plan=result['plan'];record['plan']=plan.as_dict();record['operations']=check_plan(plan,config)
        prediction=[]
        for name,centres in context['centres'].items():
            for index,centre in enumerate(centres):
                for scenario in context['scenarios']:
                    try:
                        future=_future_for_center(context,centre,scenario,plan.u_kw)
                        voltage=abs(future['geometry']['physical']['voltage'])
                        lo,hi=float(voltage.min()),float(voltage.max())
                        prediction.append(dict(candidate=name,centre=index,scenario=scenario['name'],
                            voltage_min=lo,voltage_max=hi,feasible=lo>=.9 and hi<=1.1))
                    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                        prediction.append(dict(candidate=name,centre=index,scenario=scenario['name'],
                            feasible=False,error=str(error)))
        record['nominal_voltage_admission']=dict(feasible=all(row['feasible'] for row in prediction),
                                                checks=prediction,limits=[.9,1.1])
        new=dict(u_kw=plan.u_kw,q_kw=np.zeros_like(plan.u_kw),sites=np.asarray(record['operations']['sites']))
        record['selected_control_audit']=solve_controls(context,raw,pairs,{'audit_MILP':new})[0]
    return record


def audit_history(history, actions, *, run_milp=False):
    began=perf_counter();horizon=len(next(iter(actions.values()))['u_kw'])
    seed=history_seed(history);u0=np.zeros((horizon,history.n));u0[:,0]=3.
    context=build_history_context(history,complete_bank(),model(history.n),horizon=horizon,
        max_centers=2,max_attempts_per_candidate=3,max_nfev=300,multistart=2,
        line_trust_fraction=.6,source_delta_bound=.15,center_loss_slack=12.,seed=seed)
    output=dict(history_hash=history_input_hash(history),history_only_controller_seed=seed,
        historical_context=context['summary'],selector_uses_truth=False,
        selector_uses_future_noise=False,selector_uses_evaluation_seed=False,
        calibrated_policy_component=False,coverage_guarantee=False,domains=[],models=[],milp={},
        input_controls={name:dict(raw_u_kw=np.asarray(action['u_kw']).tolist(),
            canonical_AC_u_kw=np.maximum(np.asarray(action['u_kw']),0.).tolist(),
            numerical_input_audit=action.get('numerical_input_audit')) for name,action in actions.items()})
    if context['status']!='ready':
        output.update(status='required_history_failure',seconds=perf_counter()-began)
        return output
    try:
        broad=build_frozen_pairs(context,u0,line_trust_fraction=.6,source_delta_bound=.15)
    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
        output.update(status='required_proxy_failure',error=str(error),seconds=perf_counter()-began)
        return output
    base=[FrozenPair(**raw) for raw in broad]
    for count,source,line in product(CENTRE_COUNTS,SOURCE_RADII,LINE_RADII):
        raw_pairs=[];pairs=[]
        for raw,pair in zip(broad,base):
            if any(index>=count for index in raw['metadata']['centre_indices']):
                continue
            narrowed,narrow_pair=restrict_pair(context,raw,pair,line_radius=line,source_radius=source)
            raw_pairs.append(narrowed);pairs.append(narrow_pair)
        domain=dict(centres=count,source_radius=source,line_radius=line)
        try:
            controls=solve_controls(context,raw_pairs,pairs,actions)
            domain_status='complete'
            error=None
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exception:
            controls=[];domain_status='required_domain_failure';error=str(exception)
        output['domains'].append(dict(domain=domain,controls=controls,
            status=domain_status,error=error,
            matrices_fixed_from_broad_domain=True,required_pair_count=len(pairs),
            pair_domain_hashes={pair.name:pair.fingerprint() for pair in pairs}))
    output['nested_monotonicity_checks']=nested_checks(output['domains'])
    if any(row['invalid'] for row in output['nested_monotonicity_checks']):
        output['status']='numerical_nested_bound_conflict'
    else:
        output['status']='complete'
    for name,specs in default_models().items():
        changed=scenario_context(context,specs)
        record=dict(model=name,scenario_specs=specs,nested_comparison=False,
                    centre_family_unchanged=True,coverage_guarantee=False)
        try:
            raw=build_frozen_pairs(changed,u0,line_trust_fraction=.3,source_delta_bound=.05)
            pairs=[FrozenPair(**row) for row in raw]
            record.update(status='complete',controls=solve_controls(changed,raw,pairs,actions),
                required_pair_count=len(pairs),pair_domain_hashes={pair.name:pair.fingerprint() for pair in pairs})
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            record.update(status='required_model_failure',error=str(error),controls=[])
        output['models'].append(record)
        if run_milp and name=='mean_and_ols':
            try:
                broad_two=build_frozen_pairs(changed,u0,line_trust_fraction=.6,source_delta_bound=.15)
                output['milp']['mean_and_ols_broad']=_search(changed,broad_two,
                    [FrozenPair(**row) for row in broad_two],actions,u0)
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                output['milp']['mean_and_ols_broad']=dict(status='required_model_failure',error=str(error))
    if run_milp:
        try:
            output['milp']['mean_broad']=_search(context,broad,base,actions,u0)
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            output['milp']['mean_broad']=dict(status='required_model_failure',error=str(error))
    output['required_failure_count']=sum(item.get('status')!='complete' for item in output['domains'])+sum(
        row.get('status')!='complete' for item in output['domains'] for row in item['controls'])+sum(
        item.get('status')!='complete' for item in output['models'])+sum(
        row.get('status')!='complete' for item in output['models'] for row in item['controls'])
    all_controls=[row for item in output['domains']+output['models'] for row in item['controls']]
    output['active_AC_witness_failure_count']=sum(
        not row['active_AC_witness'].get('physically_feasible',False) for row in all_controls
        if 'active_AC_witness' in row)
    if output['status']=='complete' and (output['required_failure_count'] or output['active_AC_witness_failure_count']):
        output['status']='complete_with_required_failures'
    output['seconds']=perf_counter()-began
    return output


def _flatten(result):
    for category,items in [('nested_domain',result['domains']),('forecast_model',result['models'])]:
        for item in items:
            label=item.get('domain',{'model':item['model']} if 'model' in item else {})
            for row in item.get('controls',[]):
                actual=row.get('active_AC_witness',{})
                active=row.get('active_affine',{})
                yield dict(history_hash=result['history_hash'],category=category,label=label,method=row['method'],
                    status=row['status'],q_lower=row.get('q_lower'),q_upper=row.get('q_upper'),
                    active_pair=row.get('active_pair'),affine_history_cost=active.get('affine_history_cost'),
                    affine_future_distance_squared=active.get('affine_future_distance_squared'),
                    actual_history_costs=actual.get('actual_history_costs'),
                    actual_future_distance_squared=actual.get('actual_future_distance_squared'),
                    actual_soft_cost=actual.get('actual_soft_cost'),Taylor_residual_norm=actual.get('joint_taylor_error_norm'),
                    AC_witness_physically_feasible=actual.get('physically_feasible'),
                    source_compensation=active.get('source_compensation'),required_pair_count=row['required_pair_count'],
                    failed_pairs=row['failed_pairs'],nonlinear_infimum_computed=False)


def run_audit(input_folder,out,*,limit=4,milp_histories=2):
    input_folder,out=Path(input_folder),Path(out);out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('history_*.json')) or any(out.glob('failed_input_*.json')) or (out/'progress.json').exists():
        raise ValueError('existing domain audit evidence retained; use a fresh output directory')
    files=sorted(input_folder.glob('trial_[0-9]*.json'))[:limit]
    if not files:raise ValueError('no completed pilot historical archives')
    records=[];csv=[]
    for index,file in enumerate(files):
        started=perf_counter()
        failure_stage='input_loading'
        try:
            history,actions=load_history_actions(file)
            failure_stage='history_and_domain_audit'
            result=audit_history(history,actions,run_milp=index<milp_histories)
            result['input_trial_filename']=file.name
            result['whitelisted_inputs_sha256']=history_input_hash(history)
            result['input_action_hashes']={name:_hash_arrays(value['u_kw'],value['q_kw'],value['sites'])
                                          for name,value in actions.items()}
            result['output_file']=f"history_{result['history_hash'][:12]}.json"
        except Exception as error:
            # Required input/numerical failures preserve the denominator.
            # KeyboardInterrupt/SystemExit remain external interruptions.
            result=dict(history_hash=None,input_trial_filename=file.name,
                output_file=f"failed_input_{file.stem}.json",status='required_input_or_audit_failure',
                failure_stage=failure_stage,error_type=type(error).__name__,error=str(error),seconds=perf_counter()-started,
                domains=[],models=[],milp={},required_failure_count=1,
                independent_history_denominator_preserved=True,
                calibrated_policy_component=False,coverage_guarantee=False,
                original_EV_trial_modified=False)
        _dump(out/result['output_file'],result)
        csv.extend(_flatten(result));records.append(result)
        _dump(out/'progress.json',dict(completed=len(records),requested=len(files),
            last_history_hash=result['history_hash'],selected_by='first sorted completed pilot files'))
        print(json.dumps(dict(history=index+1,count=len(files),status=result['status'],seconds=result['seconds'])),flush=True)
    _csv(out/'domain_controls.csv',csv)
    summary=dict(history_count=len(records),source_radii=SOURCE_RADII,line_radii=LINE_RADII,
        requested_max_histories=limit,selected_file_count=len(files),
        failure_history_count=sum(row['status']!='complete' for row in records),
        independent_history_denominator_includes_input_failures=True,
        centre_counts=CENTRE_COUNTS,selected_by='first sorted completed pilot files, without truth/failure filtering',
        model_specs=default_models(),calibrated_policy_component=False,coverage_guarantee=False,
        status_counts={status:sum(row['status']==status for row in records)
                       for status in sorted(set(row['status'] for row in records))},
        histories=[dict(history_hash=row['history_hash'],seconds=row['seconds'],
            input_trial_filename=row['input_trial_filename'],file=row['output_file'],status=row['status']) for row in records],
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('adaptive_domain_audit.py','adaptive_history.py','ev_information_milp.py')
                       if Path(__file__).with_name(name).exists()})
    _dump(out/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=4)
    parser.add_argument('--milp-histories',type=int,default=2)
    args=parser.parse_args()
    run_audit(args.input,args.out,limit=args.limit,milp_histories=args.milp_histories)


if __name__=='__main__':main()
