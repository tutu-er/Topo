"""Read sealed JSON/NPZ only; aggregate all resource-validation outcomes.

No project solver, AC routine, identifier, or model-fitting module is imported.
The known correct topology is recovered from independently verified old result
labels and the old frozen candidate bank, exclusively for after-run evaluation.
Public outputs contain aggregates, not fitted matrices or observation arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


OLD = 'registered_plus1'
NEW = 'registered_q95_36'
POLICIES = ('no_dfs', 'symmetric_guard', 'symmetric_guard_rotation')
NONZERO = 1e-6


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stats(values):
    x = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
        minimum=float(x.min()) if len(x) else None, maximum=float(x.max()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None, total=float(x.sum()) if len(x) else None)


def count(rows, field):
    return dict(Counter(str(row.get(field, 'missing')) for row in rows))


def values(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def load_run(path, phase, variants):
    path = Path(path).resolve()
    p = read(path/'protocol.json')
    complete = read(path/'complete.json') if (path/'complete.json').exists() else None
    task_completion = (read(complete['main']['path'])
        if complete and isinstance(complete.get('main'),dict) and Path(complete['main']['path']).exists() else {})
    entries = [e for e in read(path/'inputs.json') if e['resource_variant'] in variants]
    rows, quotes, forecasts = [], [], {}
    for e in entries:
        score = read(e['score']['path'])
        for draw in range(p['forecasts']):
            directory = path/'fraction0.10'/e['id']/f'forecast{draw:02d}'
            key = (e.get('original_id', e['id']), e['resource_variant'], draw)
            cfg = read(directory/'resource_case.json') if (directory/'resource_case.json').exists() else None
            planner = values(directory/'planner.npz') if (directory/'planner.npz').exists() else None
            plan = read(directory/'plans.json') if (directory/'plans.json').exists() else {}
            quote = dict(plan.get('symmetric_guard', dict(status='missing', feasible=False)),
                _phase=phase, _variant=e['resource_variant'], _forecast=draw)
            quotes.append(quote)
            forecasts[key] = dict(directory=directory, entry=e, score=score, config=cfg, planner=planner)
            stored = read(directory/'results.json') if (directory/'results.json').exists() else []
            indexed = {(r['repeat'], r['strategy']):r for r in stored}
            if len(indexed) != len(stored):
                raise ValueError('Duplicate execution/strategy records: '+str(directory))
            for repeat in range(p['repeats']):
                for strategy in POLICIES:
                    row = dict(indexed.get((repeat, strategy), dict(status='missing')))
                    row.update(_phase=phase, _variant=e['resource_variant'], _original=e.get('original_id', e['id']),
                        _draw=draw, _repeat=repeat, _strategy=strategy, _score=score,
                        _folder=directory/f'execution{repeat:02d}'/strategy, _forecast_context=forecasts[key])
                    rows.append(row)
    return dict(path=path, phase=phase, protocol=p, complete=complete, task_completion=task_completion, entries=entries,
                rows=rows, quotes=quotes, forecasts=forecasts)


def truth_identity(previous):
    """Use old, already evaluated labels; never infer truth from new outcomes."""
    identities = defaultdict(set)
    evidence_counts = Counter()
    for r in previous['rows']:
        if r.get('status') == 'complete' and r.get('native_map_correct'):
            bank = r['_score']['native_bank']
            identities[r['_original']].add(bank[r['native_selected_index']]['topology_id'])
            evidence_counts[r['_original']] += 1
    result = {}
    for e in previous['entries']:
        original = e.get('original_id', e['id'])
        if len(identities[original]) != 1:
            raise ValueError('Old verified labels do not identify one correct topology: '+original)
        result[original] = next(iter(identities[original]))
    return result, dict(evidence_counts)


def fixed_pair(score, true_id):
    bank = score['native_bank']
    correct = [i for i,m in enumerate(bank) if m['topology_id'] == true_id]
    wrong = [i for i,m in enumerate(bank) if m['topology_id'] != true_id]
    if not correct or not wrong:
        return None
    target = min(correct, key=lambda i:(bank[i]['history_sse'],i))
    other = int(score['native_index'])
    if bank[other]['topology_id'] == true_id:
        other = min(wrong, key=lambda i:(bank[i]['history_sse'],i))
    return target, other


def pair_J(path, pair):
    if pair is None or not Path(path).exists():
        return None
    a = values(path)
    for k, candidate in enumerate(a['pairs']):
        if set(map(int,candidate)) == set(pair):
            return float(a['J'][k])
    raise ValueError('Fixed native pair missing from saved evidence')


def action_quantities(action, config):
    """Independent scalar energy/fee arithmetic; no production domain class."""
    clock = (8+np.arange(24)+.5)%24
    p_multiplier = np.ones(24)
    pricing = config['pricing']
    for a,b in pricing['peak_clock_windows']:
        p_multiplier[(clock>=a)&(clock<b)] = pricing['clock_hour_multiplier']['morning_evening_peak']
    for a,b in pricing['night_clock_windows']:
        p_multiplier[(clock>=a)&(clock<b)] = pricing['clock_hour_multiplier']['night']
    totals = dict(active_abs_kwh=0.,reactive_abs_kvarh=0.,active_yuan=0.,reactive_yuan=0.)
    node_actions = defaultdict(lambda:np.zeros(24))
    node_fees = defaultdict(float)
    for col,(resource,unit) in enumerate(zip(config['resources'],config['action_units'],strict=True)):
        x = action[:,col]
        is_p = unit == 'kW'
        suffix = 'kwh' if is_p else 'kvarh'
        multiplier = p_multiplier if is_p else np.asarray(resource.get('hourly_fee_multiplier',np.ones(24)))
        fee = float(np.sum(multiplier*(resource['fee_positive_yuan_per_'+suffix]*np.maximum(x,0.)
            +resource['fee_negative_yuan_per_'+suffix]*np.maximum(-x,0.))))
        totals['active_abs_kwh' if is_p else 'reactive_abs_kvarh'] += float(np.abs(x).sum())
        totals['active_yuan' if is_p else 'reactive_yuan'] += fee
        key = str(resource['node_label'])+'/'+unit
        node_actions[key] += x
        node_fees[key] += fee
    totals['total_yuan'] = totals['active_yuan']+totals['reactive_yuan']
    nodes = {key:dict(nonzero_hours=int(np.count_nonzero(np.abs(a)>NONZERO)),
        nonzero_action=bool(np.max(np.abs(a),initial=0.)>NONZERO),
        peak=float(np.max(np.abs(a),initial=0.)),net_abs_energy=float(np.abs(a).sum()),fee_yuan=node_fees[key])
        for key,a in node_actions.items()}
    return totals,nodes


def replay_rows(run, truths):
    checks = Counter()
    maximum = defaultdict(float)
    pair_identity = {}
    for r in run['rows']:
        if r.get('status') != 'complete':
            continue
        path = r['_folder']/'observations.npz'
        if not path.exists():
            checks['missing_complete_observation_files'] += 1
            continue
        z = values(path);s = r['_score'];true_id = truths[r['_original']]
        H = np.asarray([m['H'] for m in s['native_bank']]);hist = np.asarray([m['history_sse'] for m in s['native_bank']])
        power = np.c_[z['P'],z['Q']]
        new = np.sum((z['Y'][None]-np.einsum('tc,knc->ktn',power,H))**2,axis=(1,2))
        total = hist+new
        scaled = total/(s['cells']+z['Y'].size)/s['reference_mse']
        eligible = np.flatnonzero(scaled<=scaled.min()+s['validation_tie'])
        chosen = int(min(eligible,key=lambda i:(s['native_bank'][i]['internal'],scaled[i],i)))
        correct = s['native_bank'][chosen]['topology_id'] == true_id
        checks['native_selection_checked'] += 1
        checks['native_selection_mismatch'] += chosen != r['native_selected_index']
        checks['native_correctness_mismatch'] += bool(correct) != bool(r['native_map_correct'])
        maximum['native_total_sse_error'] = max(maximum['native_total_sse_error'],float(np.max(np.abs(total-z['native_total_sse']))))
        r['_replayed_correct'] = bool(correct)
        representative_H=np.asarray([m['H'] for m in s['parameters']])
        representative_loss=np.asarray([m['history_sse'] for m in s['parameters']])+np.sum(
            (z['Y'][None]-np.einsum('tc,knc->ktn',power,representative_H))**2,axis=(1,2))
        ll=-(representative_loss-representative_loss.min())/(2*s['sigma2'])
        ll-=np.log(np.exp(ll).sum())
        maximum['representative_logprob_error']=max(maximum['representative_logprob_error'],float(np.max(np.abs(ll-z['representative_log_probability']))))
        true_representatives = [i for i,m in enumerate(s['models']) if m['topology_id']==true_id]
        true_weight = float(np.exp(ll[true_representatives]).sum()) if true_representatives else 0.
        best = int(np.argmax(ll));best_correct = s['models'][best]['topology_id']==true_id
        r['_true_weight'] = true_weight
        r['_true_high'] = true_weight>=.9
        r['_wrong_high'] = bool(not best_correct and np.exp(ll[best])>=.9)
        if r.get('p_true') is not None:
            maximum['true_weight_error'] = max(maximum['true_weight_error'],abs(true_weight-r['p_true']))
        pair = fixed_pair(s,true_id)
        pair_identity[r['_original']] = dict(correct_native_index=pair[0],competitor_native_index=pair[1]) if pair else None
        if pair is not None:
            i,j = pair;ss=s['sigma2'];physical = np.c_[z['true_P'],z['true_Q']]
            clean_y=z['true_root'][:,None]**2-z['true_V']**2
            h=(hist[j]-hist[i])/(2*ss)
            clean_gap=(np.sum((clean_y-physical@H[j].T)**2)-np.sum((clean_y-physical@H[i].T)**2))/(2*ss)
            r['_pair_values']=dict(actual_J=float(np.sum((physical@(H[i]-H[j]).T)**2)/(2*ss)),
                clean_logodds=float(h+clean_gap),observed_logodds=float((total[j]-total[i])/(2*ss)),
                forecast_J=pair_J(r['_forecast_context']['directory']/'exact_requested'/r['_strategy']/'forecast_pair_evidence.npz',pair),
                ideal_J=pair_J(r['_forecast_context']['directory']/'exact_requested'/r['_strategy']/'pair_evidence.npz',pair))
        config=r['_forecast_context']['config']
        requested,nodes=action_quantities(z['requested_action'],config)
        delivered,delivered_nodes=action_quantities(z['delivered_action'],config)
        r['_requested_quantities']=requested;r['_delivered_quantities']=delivered;r['_delivered_nodes']=delivered_nodes
        for key in ('active_abs_kwh','reactive_abs_kvarh','active_yuan','reactive_yuan','total_yuan'):
            for name,actual,stored in [('requested',requested,r.get('requested_quantities',{})),('delivered',delivered,r.get('delivered_quantities',{}))]:
                if key in stored:
                    maximum[name+'_'+key+'_error']=max(maximum[name+'_'+key+'_error'],abs(actual[key]-stored[key]))
        state_ref=r.get('physical_state')
        if state_ref and Path(state_ref['path']).exists():
            with np.load(state_ref['path'],allow_pickle=False) as ac:
                v=ac['voltage_magnitude_pu']
                if 'time_slice' in state_ref:
                    a,b=state_ref['time_slice'];v=v[a:b]
                r['_voltage']=dict(minimum=float(v.min()),maximum=float(v.max()),
                    outside_cells=int(np.count_nonzero((v<.95)|(v>1.05))),cells=int(v.size))
            checks['all_bus_voltage_arrays_checked'] += 1
        else:
            checks['missing_full_voltage_array'] += 1
    by_path={(r['_original'],r['_variant'],r['_draw'],r['_repeat'],r['_strategy']):r for r in run['rows']}
    for r in run['rows']:
        baseline=by_path.get((r['_original'],r['_variant'],r['_draw'],r['_repeat'],'no_dfs'))
        if '_pair_values' not in r or not baseline or '_pair_values' not in baseline:
            continue
        delta={key:r['_pair_values'][key]-baseline['_pair_values'][key]
               for key in r['_pair_values'] if r['_pair_values'][key] is not None and baseline['_pair_values'][key] is not None}
        delta['bias']=delta['clean_logodds']-delta['actual_J']
        delta['meter']=delta['observed_logodds']-delta['clean_logodds']
        r['_evidence_delta']=delta
    return dict(counts=dict(checks),maximum_errors=dict(maximum),fixed_evaluation_pairs=pair_identity,
        scope='Saved-array arithmetic only; full support truth flags and device feasibility are retained from separate execution/verification receipts')


def paired_compare(actual, reference):
    key=lambda r:(r['_original'],r['_draw'],r['_repeat'])
    a,b={key(r):r for r in actual},{key(r):r for r in reference}
    n=gains=losses=0;differences=[]
    for k in sorted(set(a)|set(b)):
        x,y=a.get(k),b.get(k)
        if x is None or y is None or '_replayed_correct' not in x or '_replayed_correct' not in y:
            continue
        n+=1;gains+=x['_replayed_correct'] and not y['_replayed_correct'];losses+=y['_replayed_correct'] and not x['_replayed_correct']
        differences.append(x['_true_weight']-y['_true_weight'])
    return dict(planned_pairs=len(set(a)|set(b)),complete_pairs=n,incomplete_pairs=len(set(a)|set(b))-n,
        actual_correct=sum(r.get('_replayed_correct',False) for r in a.values()),
        reference_correct=sum(r.get('_replayed_correct',False) for r in b.values()),
        added_recoveries=int(gains),lost_recoveries=int(losses),net_change=int(gains-losses),true_weight_change=stats(differences))


def node_summary(records):
    grouped=defaultdict(list)
    for nodes in records:
        for key,value in nodes.items():grouped[key].append(value)
    return {key:dict(actions=len(rows),nonzero_actions=sum(r['nonzero_action'] for r in rows),
        nonzero_hours=sum(r['nonzero_hours'] for r in rows),peak=max(r['peak'] for r in rows),
        net_absolute_energy=stats([r['net_abs_energy'] for r in rows]),fee_yuan=stats([r['fee_yuan'] for r in rows]))
        for key,rows in sorted(grouped.items())}


def summarize(run):
    result={}
    for variant in sorted({r['_variant'] for r in run['rows']}):
        result[variant]={}
        for strategy in POLICIES:
            rows=[r for r in run['rows'] if r['_variant']==variant and r['_strategy']==strategy]
            done=[r for r in rows if '_replayed_correct' in r]
            zero=[r for r in run['rows'] if r['_variant']==variant and r['_strategy']=='no_dfs']
            evidence=[r['_evidence_delta'] for r in done if '_evidence_delta' in r]
            voltages=[r['_voltage'] for r in done if '_voltage' in r]
            plans=[]
            for (_,v,_),ctx in run['forecasts'].items():
                if v==variant and ctx['planner'] is not None and strategy in ctx['planner']:
                    _,nodes=action_quantities(ctx['planner'][strategy],ctx['config']);plans.append(nodes)
            result[variant][strategy]=dict(planned_paths=len(rows),status=count(rows,'status'),
                evaluated_paths=len(done),failed_or_missing_paths=len(rows)-len(done),
                native_correct=sum(r['_replayed_correct'] for r in done),
                full_support_exact=sum(r.get('native_full_support_exact',False) for r in done),
                representative_true_weight=stats([r['_true_weight'] for r in done]),
                true_weight_at_least_90=sum(r['_true_high'] for r in done),wrong_weight_at_least_90=sum(r['_wrong_high'] for r in done),
                zero_fallback_paths=sum(r.get('quote_zero_fallback',False) for r in rows),
                planned_fee_yuan=stats([r.get('planned_fee_yuan') for r in rows]),settled_fee_yuan=stats([r.get('settled_fee_yuan') for r in done]),
                requested={key:stats([r['_requested_quantities'][key] for r in done]) for key in ('active_abs_kwh','reactive_abs_kvarh','active_yuan','reactive_yuan')},
                delivered={key:stats([r['_delivered_quantities'][key] for r in done]) for key in ('active_abs_kwh','reactive_abs_kvarh','active_yuan','reactive_yuan')},
                versus_no_action=paired_compare(rows,zero),
                evidence_gain_true_vs_fixed_competitor={key:dict(stats([r.get(key) for r in evidence]),
                    positive=sum(r.get(key,0)>1e-8 for r in evidence),negative=sum(r.get(key,0)<-1e-8 for r in evidence))
                    for key in ('forecast_J','ideal_J','actual_J','clean_logodds','bias','meter','observed_logodds')},
                all_bus_voltage=dict(checked_paths=len(voltages),minimum=min((v['minimum'] for v in voltages),default=None),
                    maximum=max((v['maximum'] for v in voltages),default=None),outside_cells=sum(v['outside_cells'] for v in voltages),
                    total_cells=sum(v['cells'] for v in voltages),paths_with_excursions=sum(v['outside_cells']>0 for v in voltages)),
                requested_node_actions=node_summary(plans),delivered_node_actions=node_summary([r['_delivered_nodes'] for r in done]),
                root_active_energy_value_change_yuan=stats([r.get('root_energy_value_change_yuan') for r in done]),
                error_types=dict(Counter(str(r.get('error','missing')).splitlines()[0] for r in rows if r.get('status')!='complete')))
        quotes=[q for q in run['quotes'] if q['_variant']==variant]
        initial=[q.get('initial',{}) for q in quotes]
        result[variant]['quotes']=dict(planned_quotes=len(quotes),status=count(quotes,'status'),native_status=count(quotes,'native_status'),
            source=count(quotes,'source'),feasible=sum(q.get('feasible',False) for q in quotes),
            feasible_initials=sum(i.get('feasible',False) for i in initial),
            globally_proved_optimal=sum(q.get('global_optimality_verified',False) for q in quotes),
            zero_fallbacks=sum(q.get('zero_fallback',False) for q in quotes),
            initial_cost_yuan=stats([i.get('cost_yuan') for i in initial]),
            final_cost_yuan=stats([q.get('cost_yuan') for q in quotes]),lower_bound_yuan=stats([q.get('lower_bound_yuan') for q in quotes]),
            relative_gap=stats([q.get('relative_gap') for q in quotes]),
            actual_improvement_yuan=stats([q.get('improvement_from_verified_initial_yuan') for q in quotes]),
            charged_seconds=stats([q.get('charged_total_seconds') for q in quotes]),
            charged_budget_exceeded=sum(q.get('charged_total_budget_exceeded',False) for q in quotes),
            declared_seconds_per_quote=run['protocol']['solver_seconds'],
            seed_seconds=stats([q.get('shared_seed_seconds') for q in quotes]),
            same_fee_rotations=sum(q.get('rotation_same_fee',False) for q in quotes),
            unequal_fee_rotations=sum('rotation_same_fee' in q and not q['rotation_same_fee'] for q in quotes))
    return result


def identical_paths(left,right,left_variant,right_variant):
    lf={(o,d):c for (o,v,d),c in left['forecasts'].items() if v==left_variant}
    rf={(o,d):c for (o,v,d),c in right['forecasts'].items() if v==right_variant}
    checks=Counter();errors=[]
    def same_arrays(a,b,category):
        if a is None or b is None:
            checks[category+'_missing']+=1;return
        if set(a)!=set(b):
            checks[category+'_key_mismatch']+=1
        for key in set(a)&set(b):
            checks[category+'_arrays_checked']+=1
            if not np.array_equal(a[key],b[key],equal_nan=True):
                checks[category+'_array_mismatch']+=1
                errors.append(category+'/'+key)
    def file_arrays(path):
        return values(path) if path.exists() else None
    for key in sorted(set(lf)|set(rf)):
        a,b=lf.get(key),rf.get(key)
        if a is None or b is None:
            checks['forecast_missing']+=1;continue
        same_arrays(None if a['planner'] is None else {'forecast':a['planner']['forecast']},
                    None if b['planner'] is None else {'forecast':b['planner']['forecast']},'forecast')
        checks['candidate_bank_checked']+=1
        if a['score']['parameter_sha256'] != b['score']['parameter_sha256']:
            checks['candidate_bank_mismatch']+=1
        for repeat in range(max(left['protocol']['repeats'],right['protocol']['repeats'])):
            ad=a['directory']/f'execution{repeat:02d}';bd=b['directory']/f'execution{repeat:02d}'
            same_arrays(file_arrays(ad/'paired_randomness.npz'),file_arrays(bd/'paired_randomness.npz'),'randomness')
            same_arrays(file_arrays(ad/'no_dfs'/'observations.npz'),file_arrays(bd/'no_dfs'/'observations.npz'),'zero_observations')
    failed=sum(v for k,v in checks.items() if 'mismatch' in k or 'missing' in k)
    return dict(passed=failed==0 and bool(lf) and bool(rf),counts=dict(checks),
        mismatch_array_names=dict(Counter(errors)),comparison='Exact array equality, including every saved no-action NPZ field, all eta/noise components, forecast and frozen native-bank hash')


def selected(run,variant,strategy):
    return [r for r in run['rows'] if r['_variant']==variant and r['_strategy']==strategy]


def report(paired_run,fresh_run,previous_run,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve completed analysis: '+str(out))
    for path in [previous_run,paired_run]+([fresh_run] if fresh_run is not None else []):
        if not (Path(path)/'complete.json').exists():
            raise ValueError('Run is unfinished; omit --fresh-run for paired-only reporting: '+str(path))
    previous=load_run(previous_run,'previous',[OLD]);paired=load_run(paired_run,'paired',[NEW])
    fresh=load_run(fresh_run,'fresh',[OLD,NEW]) if fresh_run is not None else None
    runs=[previous,paired]+([fresh] if fresh is not None else [])
    truths,old_labels=truth_identity(previous)
    replay={run['phase']:replay_rows(run,truths) for run in runs}
    summaries={run['phase']:summarize(run) for run in runs}
    matching=dict(paired_to_previous=identical_paths(paired,previous,NEW,OLD))
    comparisons=dict(paired_new_vs_old=paired_compare(selected(paired,NEW,'symmetric_guard'),selected(previous,OLD,'symmetric_guard')))
    a=comparisons['paired_new_vs_old']
    completed=all(run['complete'] is not None and run['complete'].get('source_unchanged') for run in runs)
    flags=dict(all_runs_complete_with_locked_sources=completed,
        paired_all_paths_comparable=matching['paired_to_previous']['passed'] and a['incomplete_pairs']==0,
        paired_previous_count_is_26=a['reference_correct']==26,
        paired_updated_at_least_26=a['actual_correct']>=26,
        paired_updated_count_at_least_previous=a['actual_correct']>=a['reference_correct'],
        paired_loses_no_previous_correct_path=a['lost_recoveries']==0 and a['incomplete_pairs']==0)
    if fresh is not None:
        matching['fresh_between_resources']=identical_paths(fresh,fresh,NEW,OLD)
        b=comparisons['fresh_new_vs_old']=paired_compare(selected(fresh,NEW,'symmetric_guard'),selected(fresh,OLD,'symmetric_guard'))
        c=comparisons['fresh_new_vs_zero']=paired_compare(selected(fresh,NEW,'symmetric_guard'),selected(fresh,NEW,'no_dfs'))
        flags.update(fresh_all_paths_comparable=matching['fresh_between_resources']['passed'] and b['incomplete_pairs']==0 and c['incomplete_pairs']==0,
            fresh_updated_count_at_least_old=b['actual_correct']>=b['reference_correct'],
            fresh_updated_count_at_least_zero=c['actual_correct']>=c['reference_correct'],
            fresh_no_losses_vs_old=b['lost_recoveries']==0 and b['incomplete_pairs']==0,
            fresh_no_losses_vs_zero=c['lost_recoveries']==0 and c['incomplete_pairs']==0)
    protocols={run['phase']:dict(protocol=ref(run['path']/'protocol.json'),inputs=ref(run['path']/'inputs.json'),
        complete=ref(run['path']/'complete.json') if run['complete'] else None,
        seed=run['protocol']['seed'],forecasts=run['protocol']['forecasts'],repeats=run['protocol']['repeats'],
        completion_record=run['complete'],
        failed_tasks=len(run['task_completion'].get('failed_tasks',[])),
        failed_task_error_types=dict(Counter(str(task.get('error','unspecified')).splitlines()[0] for task in run['task_completion'].get('failed_tasks',[]))),
        quote_exceptions=run['task_completion'].get('quote_exceptions'),execution_failed_rows=run['task_completion'].get('execution_failed_rows'),
        interpretation=run['protocol'].get('interpretation')) for run in runs}
    answer=dict(created_utc=datetime.now(timezone.utc).isoformat(),analysis_source=ref(__file__),protocols=protocols,
        scope='One known physical day; paired phase reuses old errors, fresh phase uses new conditional error streams. Histories and resource variants are not independent physical days.',
        previous_reference_filter='Only registered_plus1 and the three comparable strategies are aggregated from the previous multi-resource/multi-method run; its overall completion metadata remains referenced.',
        fresh_status='included_complete' if fresh is not None else 'not_included; no fresh-path statistics or conclusions produced',
        truth_source='Old already evaluated native-correct labels joined to the old frozen score; used only after execution',
        old_true_label_evidence_counts=old_labels,groups=summaries,comparisons=comparisons,
        stream_identity=matching,independent_arithmetic=replay,protocol_acceptance=flags,
        node_action_threshold=NONZERO,node_action_units='Peak kW or kvar; hourly sums kWh or kvarh. Node energy is net across same-unit resources; fees remain sum of resource charges.',
        evidence_orientation='Correct native topology versus the fixed historical winner (or strongest wrong history candidate if history winner is correct). Positive evidence gain favors the correct topology.',
        inference_limit='Acceptance flags are the predeclared sample checks, not a statistical recovery guarantee. Independent full device/AC verification is reported separately.')
    out.mkdir(parents=True)
    (out/'REPORT.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Resource validation results','',answer['scope'],'',answer['fresh_status'],'',
        '| Phase | Resource | Strategy | Evaluated/planned | Native correct | Full support exact | True ≥.9 | Wrong ≥.9 | Mean fee/yuan |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def number(value):return 'unavailable' if value is None else f'{value:.6g}'
    for phase,variants in summaries.items():
        for variant,data in variants.items():
            for strategy in POLICIES:
                g=data[strategy]
                lines.append(f"| {phase} | {variant} | {strategy} | {g['evaluated_paths']}/{g['planned_paths']} | {g['native_correct']} | {g['full_support_exact']} | {g['true_weight_at_least_90']} | {g['wrong_weight_at_least_90']} | {number(g['settled_fee_yuan']['mean'])} |")
    lines+=['','| Comparison | Complete pairs | Added recoveries | Lost recoveries | Actual/reference correct |','|---|---:|---:|---:|---:|']
    for name,g in comparisons.items():
        lines.append(f"| {name} | {g['complete_pairs']}/{g['planned_pairs']} | {g['added_recoveries']} | {g['lost_recoveries']} | {g['actual_correct']}/{g['reference_correct']} |")
    lines+=['','Protocol checks:','']+[f'- {key}: {value}' for key,value in flags.items()]
    lines+=['','Every failed/missing execution remains in the planned denominator. JSON includes all status counts, quotation bounds/initialization/timing, node actions, voltage checks, and clean/measured evidence changes. No new optimization, fitting or AC was run.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-run',type=Path,required=True)
    parser.add_argument('--fresh-run',type=Path)
    parser.add_argument('--previous-run',type=Path,default=Path('outputs/flexibility_reaudit_long180_20261114_v1'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=report(args.paired_run,args.fresh_run,args.previous_run,args.out)
    print(json.dumps(dict(output=str(args.out.resolve()),comparisons=result['comparisons'],protocol_acceptance=result['protocol_acceptance']),ensure_ascii=False))
