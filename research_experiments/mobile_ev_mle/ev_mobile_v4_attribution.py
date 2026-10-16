"""Predeclared, resumable 2^3 offline attribution for frozen v4 EV actions.

Only evaluation histories 0 and 6, their guarded FD/MD plans, and regular/shift
futures are authorized.  True factors generate posthoc noiseless targets; no
selector or training procedure runs.  Every target uses the original v4
evaluate_future(D, target, options, seed), including shared R/X refitting.

There are 64 method/target slots and 256 joint candidate-fit slots.  Identical
targets are cached, including failures, while all slot denominators remain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter
import traceback
from types import SimpleNamespace

import numpy as np

from . import ev_dynamic_study as dynamic
from . import ev_mobile_v4_cases as cases
from . import ev_mobile_v4_study as study
from . import ev_systematic_refit as refit
from . import gradient_price_study as io
from . import likelihood as lk
from .information_context import meter_from_record
from .ev_mobile_v3_score_attribution import factor_parameters, make_target, margin, shapley_change


INDICES = (0, 6)
METHODS = ('guarded__fixed_bidir', 'guarded__mobile_bidir')
REGIMES = ('regular', 'shift')
FACTORS = ('lines', 'background', 'source')
STATES = tuple(product((0, 1), repeat=3))
TARGET_SLOTS = len(INDICES)*len(REGIMES)*len(METHODS)*len(STATES)
CANDIDATE_SLOTS = 4*TARGET_SLOTS


def fingerprint(value):
    return sha256(json.dumps(io.serial(value), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _write(path, value):
    """Atomic leaf-file checkpoint; no existing scientific record is edited."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(io.serial(value), ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.suffix == '.gz': data = gzip.compress(data, mtime=0)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes(data); temporary.replace(path)


def _read(path):
    path = Path(path); data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def _source(path):
    path = Path(path); data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=sha256(data).hexdigest())


def readiness(root, *, current_sources=None):
    """Fail closed until every frozen evaluation history has a final record."""
    root = Path(root); frozen = _read(root/'freeze.json')
    specs = frozen.get('specifications', {}).get('evaluation', [])
    if len(specs) != 8 or {s['index'] for s in specs} != set(range(8)):
        raise ValueError('exactly the eight predeclared v4 evaluation histories required')
    declared = tuple(frozen.get('config', {}).get('posthoc_diagnostic_evaluation_indices', ()))
    if declared != INDICES: raise ValueError('offline indices differ from the predeclared freeze')
    sources = study.source_hashes() if current_sources is None else current_sources
    if sources != frozen.get('source_sha256'):
        raise ValueError('scientific source differs from the frozen evaluation; no attribution fits run')
    manifest = []; selected = {}
    for spec in specs:
        path = root/'evaluation'/(str(spec['index']).zfill(3)+'.json.gz')
        if not path.exists(): raise ValueError('frozen evaluation is not finished: missing '+path.name)
        raw = _read(path)
        if raw.get('spec') != spec or raw.get('policy_id') != frozen['policy_id']:
            raise ValueError('evaluation record identity differs from its freeze')
        if raw.get('status') not in ('complete', 'actual_fit_failure', 'training_failure', 'fatal_failure'):
            raise ValueError('evaluation record has no final status')
        source = dict(_source(path), index=spec['index'], seed=spec['seed'], status=raw['status'])
        manifest.append(source)
        if spec['index'] in INDICES:
            if tuple(spec.get('regimes', ())) != REGIMES:
                raise ValueError('both predeclared future regimes required for attribution')
            selected[spec['index']] = source
        del raw
    return frozen, manifest, selected


def _observation(record):
    return lk.Observations(**{k: None if v is None else np.asarray(v, float) for k, v in record.items()})


def _reference_gap(method, truth):
    nominal = method.get('nominal', {})
    rows = nominal.get('nearest', [])
    selected = [r['half_gap'] for r in rows if r.get('a') == truth]
    if not nominal.get('success') or len(rows) != 4 or len(selected) != 1 or not np.isfinite(selected[0]): return None
    return float(selected[0])


def _actual_halfM(raw, regime, method, names):
    selected = raw.get('methods', {}).get(method, {})
    outcome = raw.get('actual', {}).get(regime, {}).get('actions', {}).get(selected.get('action_key'), {}).get('evaluation', {})
    if selected.get('operation', {}).get('feasible') is not True or not outcome.get('complete'): return None
    rows = outcome.get('comparisons', {})
    if set(rows) != set(names) or any(r.get('joint', {}).get('success') is not True for r in rows.values()): return None
    losses = {k: r['joint'].get('loss') for k, r in rows.items()}
    if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in losses.values()):return None
    result = margin(losses, raw['true_name'], names)
    return None if result is None else .5*result


def prepare(raw, frozen, raw_source):
    """Restore saved D fits; regenerate only the two declared simulator futures."""
    began = perf_counter(); spec = raw['spec']; truth = raw['true_name']
    if spec['index'] not in INDICES: raise ValueError('unapproved attribution history')
    trained = {r.get('candidate', r.get('name')): r for r in raw.get('training_result', {}).get('fits', [])}
    if len(trained) != 4 or truth not in trained or any(r.get('success') is not True for r in trained.values()):
        raise ValueError('all four saved observed-D reference fits required; no retraining allowed')
    predicted_source = raw['forecast']['source_by_candidate'][truth]
    predicted = dict(r=np.asarray(trained[truth]['r']), x=np.asarray(trained[truth]['x']),
        p=np.asarray(raw['forecast']['p']), q=np.asarray(raw['forecast']['q']),
        sp=np.asarray(predicted_source['source_p']), sq=np.asarray(predicted_source['source_q']),
        root=np.asarray(raw['forecast']['root_v']))
    methods = raw['methods']; powers = {}; action_keys = {}
    for method in METHODS:
        if method not in methods or methods[method].get('operation', {}).get('feasible') is not True:
            raise ValueError('predeclared saved action absent or physically invalid: '+method)
        powers[method] = np.asarray(methods[method]['power_kw'], float)
        action_keys[method] = methods[method]['action_key']
    true_by_regime = {}; verification = {}; bank = None
    for regime in REGIMES:
        case = cases.make_case(spec['bank'], spec['seed'], spec['family_index'], regime)
        D = lk.concat_observations(case['train'], case['validation'])
        if fingerprint(io.observations_record(D)) != fingerprint(raw['observed']):
            raise ValueError('regenerated observed D differs from the saved history')
        if any(not np.array_equal(raw['truth'][k], case['truth'][k]) for k in ('r', 'x')):
            raise ValueError('regenerated true line parameters differ')
        bank = tuple(case['bank']); true_candidate = next(c for c in bank if c.name == truth)
        if true_candidate.supports != case['truth']['candidate'].supports:
            raise ValueError('true candidate support differs from the saved label')
        start = case['future_start']; end = start+spec['T']
        true_by_regime[regime] = dict(r=np.asarray(raw['truth']['r']), x=np.asarray(raw['truth']['x']),
            p=case['p'][start:end], q=case['q'][start:end], sp=case['source_p'][start:end],
            sq=case['source_q'][start:end], root=case['root_v'][start:end])
        if not np.array_equal(predicted['root'], true_by_regime[regime]['root']):
            raise ValueError('root differs; three-factor attribution would omit a factor')
        checked = []
        for method in METHODS:
            regenerated = cases.observe_future(case, powers[method])
            saved = raw['actual'][regime]['actions'][action_keys[method]]['future']
            if fingerprint(io.observations_record(regenerated)) != fingerprint(saved):
                raise ValueError('regenerated noisy future differs from the saved fixed action')
            checked.append(method)
        verification[regime] = dict(observed_D_identical=True, true_lines_identical=True,
            saved_noisy_fixed_action_futures_identical=checked, common_root_identical=True)
    names = [c.name for c in bank]
    if set(names) != set(trained): raise ValueError('saved candidate bank differs')
    for candidate in bank:
        if tuple(map(tuple, trained[candidate.name]['supports'])) != candidate.supports:
            raise ValueError('saved edge ordering differs from the candidate bank')
    config = frozen['config']; raw_cfg = raw['config']
    options = refit._options({**raw_cfg['training'], **raw_cfg['source'], 'ratio_bounds':raw_cfg['ratio_bounds'],
        'multistart':config['multistart_predictive'], 'max_nfev':config['max_nfev']})
    nominal = {m:dict(Phi=methods[m]['nominal'].get('score'), true_ref_m=_reference_gap(methods[m], truth)) for m in METHODS}
    md_log = raw.get('families', {}).get('guarded', {}).get('logs', {}).get('mobile_bidir', {})
    direct = md_log.get('direct_parents', [])
    highest = max(direct, key=lambda row: row['score'])['domain'] if direct else None
    return dict(schema='mobile_v4_offline_attribution_history_v1', spec=spec, raw_source=raw_source,
        true_name=truth, candidate_names=names,
        candidate_bank=[dict(name=c.name, n=c.n, supports=c.supports) for c in bank],
        observed=raw['observed'], observed_meter=raw['observed_meter'], future_meter=raw['future_meter'],
        trained=trained, predicted=predicted, true_by_regime=true_by_regime, powers=powers, action_keys=action_keys,
        base_kw=raw_cfg['base_kw'], fit_options=options, fit_seed=spec['seed']+4000,
        nominal=nominal, observed_halfM={r:{m:_actual_halfM(raw,r,m,names) for m in METHODS} for r in REGIMES},
        selected_MD_highest_Phi_direct_parent=highest,
        FD_is_selected_guard_anchor=highest == 'fixed_bidir',
        verification=verification, preparation_seconds=perf_counter()-began,
        observed_D_retrained=False, planning_called=False, source_interface=1)


def restore(meta):
    bank = [lk.RadialCandidate.from_supports(r['name'], r['n'], r['supports']) for r in meta['candidate_bank']]
    fits = {name:SimpleNamespace(**{k:np.asarray(row[k],float) for k in ('r','x','source_p','source_q')})
            for name, row in meta['trained'].items()}
    return dict(bank=bank, candidate=next(c for c in bank if c.name == meta['true_name']),
        D=_observation(meta['observed']), dm=meter_from_record(meta['observed_meter']),
        fm=meter_from_record(meta['future_meter']), fits=fits)


def target_identity(meta, parameters, power):
    """Identity before AC generation, allowing exact repeated-state caching."""
    return fingerprint(dict(candidate=meta['true_name'], parameters=parameters, power_kw=power,
                            future_meter=meta['future_meter'], base_kw=meta['base_kw']))


def fit_identity(meta, target_observations):
    return fingerprint(dict(observed=meta['observed'], observed_meter=meta['observed_meter'],
        target=target_observations, future_meter=meta['future_meter'], trained=meta['trained'],
        options=meta['fit_options'], seed=meta['fit_seed'], candidate_bank=meta['candidate_bank']))


def outcome_values(outcome, truth, names):
    rows = outcome.get('comparisons', {}) or {}; failures = outcome.get('failures', {}) or {}
    seen = (set(rows)|set(failures)) & set(names)
    success = {k:r['joint']['loss'] for k, r in rows.items() if k in names
        and r.get('joint', {}).get('success') is True and isinstance(r['joint'].get('loss'), (int,float))
        and not isinstance(r['joint'].get('loss'),bool)
        and math.isfinite(r['joint']['loss'])}
    complete = outcome.get('complete') is True and set(success) == set(names)
    M = margin(success, truth, names) if complete else None
    return dict(complete=complete, halfM=None if M is None else .5*M, M=M,
                candidate_slots=dict(expected=len(names), attempted=len(seen), success=len(success),
                    failed=len(seen)-len(success), not_run=len(names)-len(seen)))


def evaluate_slot(output, meta, context, regime, bits, method, *, evaluate=None):
    """One atomic target/evaluation unit. Failed results are retained and reused."""
    output = Path(output); index = meta['spec']['index']; state = ''.join(map(str,bits))
    path = output/f'history_{index:03d}'/regime/(method+'_'+state+'.json')
    if path.exists(): return _read(path)
    parameters = factor_parameters(meta['predicted'], meta['true_by_regime'][regime], bits)
    power = np.asarray(meta['powers'][method], float); request = target_identity(meta, parameters, power)
    target_path = output/'targets'/(request+'.json.gz'); target_reused = target_path.exists()
    if target_reused: target = _read(target_path)
    else:
        began = perf_counter()
        with dynamic.count_physics_calls() as counters:
            try:
                target_obs, physical = make_target(context['candidate'], parameters, power, context['fm'],
                    base_kw=meta['base_kw'], ac_predict=lk.ac_predict)
                target = dict(success=True, observations=io.observations_record(target_obs), physical=physical,
                    parameters=parameters, power_kw=power, true_reference=meta['true_name'])
            except Exception as exc:
                target = dict(success=False, failure=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc(),
                              parameters=parameters, power_kw=power)
        target.update(target_request=request, seconds=perf_counter()-began, counters=dict(counters))
        _write(target_path, target)
    slot = dict(index=index, regime=regime, method=method, state=state,
        factors=dict(zip(FACTORS,bits)), target_request=request, target_source=_source(target_path),
        target_reused=target_reused, expected_candidate_slots=4, cache_reused=False)
    if not target.get('success'):
        slot.update(status='target_failure', failure=target.get('failure'), candidate_slots=dict(expected=4,attempted=0,success=0,failed=0,not_run=4))
        _write(path, slot); return slot
    identity = fit_identity(meta, target['observations']); cache_path = output/'fit_cache'/(identity+'.json.gz')
    if cache_path.exists(): fitted = _read(cache_path); slot['cache_reused'] = True
    else:
        began = perf_counter()
        with dynamic.count_physics_calls() as counters:
            try:
                evaluator = study.evaluate_future if evaluate is None else evaluate
                result = evaluator(context['bank'], context['D'], context['dm'], _observation(target['observations']),
                    context['fm'], context['fits'], meta['fit_options'], meta['fit_seed'])
                fitted = dict(outcome=result, fit_identity=identity, fit_seed=meta['fit_seed'],
                    true_name=meta['true_name'], candidate_names=meta['candidate_names'])
            except Exception as exc:
                fitted = dict(outcome=dict(complete=False, comparisons={}, losses={}, failures={}),
                    failure=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc(), fit_identity=identity,
                    true_name=meta['true_name'], candidate_names=meta['candidate_names'])
        fitted.update(seconds=perf_counter()-began, counters=dict(counters))
        _write(cache_path, fitted)
    values = outcome_values(fitted['outcome'], meta['true_name'], meta['candidate_names'])
    slot.update(status='complete' if values['complete'] else 'fit_failure', **values,
        fit_identity=identity, fit_source=_source(cache_path), failure=fitted.get('failure'),
        fit_failures=fitted['outcome'].get('failures', {}))
    _write(path, slot)
    return slot


def event_summary(meta, slots, regime):
    states = {}; delta = {}; values = {m:{} for m in METHODS}
    for bits in STATES:
        key = ''.join(map(str,bits)); state = {}
        for method in METHODS:
            slot = slots.get((regime, method, key), {})
            half = slot.get('halfM') if slot.get('status') == 'complete' else None
            state[method] = dict(status=slot.get('status','not_run'), halfM=half,
                candidate_slots=slot.get('candidate_slots',dict(expected=4,attempted=0,success=0,failed=0,not_run=4)))
            if half is not None: values[method][bits] = half
        if all(state[m]['halfM'] is not None for m in METHODS):
            delta[bits] = state[METHODS[1]]['halfM']-state[METHODS[0]]['halfM']
        states[key] = dict(methods=state, delta_halfM=delta.get(bits))
    nominal = meta.get('nominal', {})
    subtract = lambda key: (None if any(nominal.get(m,{}).get(key) is None for m in METHODS)
                            else nominal[METHODS[1]][key]-nominal[METHODS[0]][key])
    phi, true_ref = subtract('Phi'), subtract('true_ref_m')
    observed = meta.get('observed_halfM', {}).get(regime, {})
    actual = None if any(observed.get(m) is None for m in METHODS) else observed[METHODS[1]]-observed[METHODS[0]]
    result = dict(index=meta['spec']['index'], bank=meta['spec']['bank'], regime=regime,
        success=len(delta)==8, states=states, delta_nominal_Phi=phi, delta_nominal_true_ref_m=true_ref,
        delta_observed_halfM=actual,
        FD_is_selected_guard_anchor=meta.get('FD_is_selected_guard_anchor'),
        selected_MD_highest_Phi_direct_parent=meta.get('selected_MD_highest_Phi_direct_parent'))
    if len(delta) == 8:
        attribution = shapley_change(delta)
        result.update(delta_000=delta[(0,0,0)], delta_111=delta[(1,1,1)],
            Shapley_delta_halfM=attribution,
            Shapley_method_halfM={m:shapley_change(v) for m,v in values.items()},
            metric_aggregation_discrepancy=None if phi is None or true_ref is None else true_ref-phi,
            nominal_vs_same_inference_rule_discrepancy=None if true_ref is None else delta[(0,0,0)]-true_ref,
            observed_minus_111_residual=None if actual is None else actual-delta[(1,1,1)],
            residual_meaning='future conditional P/Q input noise plus voltage/root-PQ metering noise and numerical AC refit differences; not a pure voltage-noise effect')
        terms = [result['metric_aggregation_discrepancy'], result['nominal_vs_same_inference_rule_discrepancy'],
                 *attribution['contributions'].values(), result['observed_minus_111_residual']]
        result['full_gap_decomposition_error'] = (None if phi is None or actual is None or any(v is None for v in terms)
                                                 else sum(terms)-(actual-phi))
    return result


def summarize(output):
    output = Path(output); events=[]; slots=[]; preparation=[]
    for index in INDICES:
        folder = output/f'history_{index:03d}'; path = folder/'history.json.gz'
        if not path.exists():
            failure = _read(folder/'preparation_failure.json') if (folder/'preparation_failure.json').exists() else None
            preparation.append(dict(index=index, success=False, failure=failure,
                seconds=None if failure is None else failure.get('seconds'),
                counters=None if failure is None else failure.get('counters')))
            for regime in REGIMES: events.append(dict(index=index,regime=regime,success=False,failure='history not prepared'))
            continue
        meta = _read(path); preparation.append(dict(index=index,success=True,seconds=meta.get('preparation_seconds'),counters=meta.get('preparation_counters')))
        selected = {}
        for regime in REGIMES:
            for method in METHODS:
                for bits in STATES:
                    key=''.join(map(str,bits)); file=folder/regime/(method+'_'+key+'.json')
                    if file.exists():
                        row=_read(file); slots.append(row); selected[regime,method,key]=row
            events.append(event_summary(meta, selected, regime))
    caches=[_read(p) for p in sorted((output/'fit_cache').glob('*.json.gz'))]
    targets=[_read(p) for p in sorted((output/'targets').glob('*.json.gz'))]
    counts=Counter()
    for row in [*caches,*targets]: counts.update(row.get('counters',{}))
    for row in preparation: counts.update(row.get('counters') or {})
    candidate = {key:sum(s.get('candidate_slots',{}).get(key,0) for s in slots)
                 for key in ('expected','attempted','success','failed','not_run')}
    missing = TARGET_SLOTS-len(slots); candidate['expected'] += 4*missing; candidate['not_run'] += 4*missing
    unique_success = unique_attempted = 0
    for cache in caches:
        # All job records include the same four candidate names through the
        # saved comparison/failure dictionary; count actual work separately.
        value=outcome_values(cache.get('outcome',{}),cache['true_name'],cache['candidate_names'])
        unique_attempted += value['candidate_slots']['attempted']
        unique_success += value['candidate_slots']['success']
    invocations=[_read(p) for p in sorted(output.glob('invocation_*.json'))]
    return dict(schema='mobile_v4_posthoc_factor_attribution_summary_v1',
        independent_history_denominator=2, future_regime_group_denominator=4,
        complete_factor_groups=sum(e['success'] for e in events), target_slot_denominator=TARGET_SLOTS,
        completed_target_slots=len(slots), not_run_target_slots=missing,
        target_slot_status=dict(Counter(s['status'] for s in slots)), candidate_slot_denominator=CANDIDATE_SLOTS,
        candidate_slots=candidate, unique_target_generations=len(targets), unique_target_failures=sum(not t.get('success') for t in targets),
        unique_evaluation_calls=len(caches), unique_attempted_joint_candidate_posts=unique_attempted,
        unique_successful_joint_candidate_posts=unique_success,
        unique_failed_joint_candidate_posts=unique_attempted-unique_success,
        cached_slot_reuses=sum(s.get('cache_reused',False) for s in slots),
        target_generation_seconds=sum(t.get('seconds',0.) for t in targets),
        fit_seconds=sum(c.get('seconds',0.) for c in caches),
        preparation_seconds=sum(p.get('seconds',0.) or 0. for p in preparation),
        invocation_count=len(invocations), recorded_invocation_seconds=sum(v.get('seconds',0.) for v in invocations),
        counters=dict(counts), events=events, preparation=preparation,
        fit_cost_interpretation='256 nominal final-joint candidate slots before caching; each v4 evaluator also runs its fixed-line source fit. Actual solver/AC cost is retained in saved outcomes/counters, not equated to slot count.',
        no_training_recomputed=True, no_selector_called=True, no_evaluation_tuning=True)


def render(summary, path, status):
    text=['# v4预定离线归因', '',
        '仅评价预定历史0、6的既有guarded固定充放电与移动充放电计划；不改变路线、参数参考、预测、评分规则或校准。历史D、标准差和物理参数域保持原样，全部竞争者按原v4流程重拟合D与无噪未来。', '',
        r'$m_S(u)=\frac12[\min_{b\ne a}L_b(D,E_S(u))-L_a(D,E_S(u))]$，$a$是真候选且仅用于本离线分析；$S$控制线路、背景P/Q、同接口源分别采用训练预测(0)或仿真真实值(1)。令$\Delta_S=m_S(u_{MD})-m_S(u_{FD})$。', '',
        '三个因素按六种替换顺序平均分配Δ111−Δ000；相加恒等，不把贡献解释为独立随机变量。另列平均指标、相同未来下推断流程的差异，以及实测减111残余。残余同时含条件P/Q输入、V与总表噪声及数值拟合差异。', '',
        f"状态 {status}；目标槽 {summary['completed_target_slots']}/{TARGET_SLOTS}；成功归因组 {summary['complete_factor_groups']}/4；候选槽成功 {summary['candidate_slots']['success']}/{CANDIDATE_SLOTS}，失败 {summary['candidate_slots']['failed']}，未运行 {summary['candidate_slots']['not_run']}。", '',
        '|历史/场景|Δ000|Δ111|线路|背景|源|实测Δ|实测−111|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for event in summary['events']:
        label=str(event['index'])+'/'+event['regime']
        if not event['success']:
            text.append('|'+label+'|未完成|||||||');continue
        c=event['Shapley_delta_halfM']['contributions'];fmt=lambda v:'—' if v is None else f'{v:.6g}'
        text.append('|'+label+'|'+'|'.join(fmt(v) for v in [event['delta_000'],event['delta_111'],c['lines'],c['background'],c['source'],event['delta_observed_halfM'],event['observed_minus_111_residual']])+'|')
    text += ['', 'MD相对FD的比较是预定空间诊断。若MD继承的最高Phi父是MC，FD不是guard的逐参考约束锚点；JSON单独记录该情况，避免把它误报为guard违规。', '',
        f"去重后实际调用 {summary['unique_evaluation_calls']} 次四候选评价，原目标槽中的缓存复用 {summary['cached_slot_reuses']} 次；拟合 {summary['fit_seconds']:.3f}s，目标生成 {summary['target_generation_seconds']:.3f}s，恢复核验 {summary['preparation_seconds']:.3f}s。"]
    Path(path).write_text('\n'.join(text)+'\n',encoding='utf8')


def run(root, output, *, budget_seconds=600., resume=False):
    began=perf_counter();root,output=Path(root),Path(output)
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('positive finite budget required')
    frozen, manifest, selected = readiness(root)
    if output.exists() and not resume:raise FileExistsError('retain existing attribution; use --resume')
    output.mkdir(parents=True,exist_ok=True)
    config=dict(schema='mobile_v4_predeclared_factor_attribution_v1', policy_id=frozen['policy_id'],
        indices=INDICES,methods=METHODS,regimes=REGIMES,factors=FACTORS,target_slots=TARGET_SLOTS,candidate_joint_slots=CANDIDATE_SLOTS,
        source_sha256=frozen['source_sha256'],evaluation_manifest=manifest,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_math_sha256=sha256(Path(__file__).with_name('ev_mobile_v3_score_attribution.py').read_bytes()).hexdigest(),
        same_D_same_v4_evaluate_future_options_seed=True,source_interface=1,selector_called=False)
    freeze_path=output/'freeze.json'
    if freeze_path.exists():
        if _read(freeze_path)!=io.serial(config):raise ValueError('diagnostic source/config changed across resume')
    else:_write(freeze_path,config)
    status='complete'; stop=False
    for index in INDICES:
        folder=output/f'history_{index:03d}';folder.mkdir(parents=True,exist_ok=True);meta_path=folder/'history.json.gz'
        failure_path=folder/'preparation_failure.json'
        if failure_path.exists():continue  # Preserve the failed denominator.
        if perf_counter()-began>=budget_seconds:status='checkpointed_budget';break
        if meta_path.exists():meta=_read(meta_path)
        else:
            preparation_began=perf_counter();counters={}
            try:
                with dynamic.count_physics_calls() as counters:
                    meta=prepare(_read(selected[index]['path']),frozen,selected[index])
                meta['preparation_counters']=dict(counters);_write(meta_path,meta)
            except Exception as exc:
                _write(failure_path,dict(index=index,failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
                    seconds=perf_counter()-preparation_began,counters=dict(counters)))
                continue
        context=restore(meta)
        for regime in REGIMES:
            for bits in STATES:
                for method in METHODS:
                    if perf_counter()-began>=budget_seconds:status='checkpointed_budget';stop=True;break
                    slot=evaluate_slot(output,meta,context,regime,bits,method)
                    _write(output/'status.json',dict(status='running',index=index,regime=regime,state=slot['state'],method=method,
                        last_slot_status=slot['status'],elapsed_seconds=perf_counter()-began))
                    print(json.dumps(dict(index=index,regime=regime,state=slot['state'],method=method,status=slot['status'],
                        cache_reused=slot.get('cache_reused'),elapsed_seconds=perf_counter()-began)),flush=True)
                if stop:break
            if stop:break
        if stop:break
    if study.source_hashes()!=frozen['source_sha256']:raise RuntimeError('frozen science changed during attribution')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    _write(output/('invocation_'+stamp+'.json'),dict(status=status,seconds=perf_counter()-began,resumed=resume))
    summary=summarize(output)
    if status=='complete' and summary['complete_factor_groups']<4:status='complete_with_failures'
    _write(output/'summary.json',summary);render(summary,output/'summary.md',status)
    _write(output/'status.json',dict(status=status,summary=summary,invocation_seconds=perf_counter()-began))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--budget-seconds',type=float,default=600.)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    result=summarize(args.output) if args.summarize_only else run(args.root,args.output,budget_seconds=args.budget_seconds,resume=args.resume)
    print(json.dumps({k:result[k] for k in ('completed_target_slots','complete_factor_groups','unique_evaluation_calls')},ensure_ascii=False))


if __name__=='__main__':main()
