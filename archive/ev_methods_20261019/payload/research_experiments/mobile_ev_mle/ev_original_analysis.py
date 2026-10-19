"""Per-history analysis and a technical pilot gate from saved evidence only.

This file is outside the scientific source freeze. It never imports or invokes
the selector, simulator, optimizer or AC fitting code. The 12 directed tasks
and paired methods remain measurements of one history, not extra samples.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
METHODS=('none','constant','quality_fixed','quality_mobile','random_mobile',
         'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')
DESIGNS=('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')
CONTRASTS=(('charge_time','charge_fixed','constant'),
    ('fixed_V2G','v2g_fixed','charge_fixed'),('charge_space','charge_mobile','charge_fixed'),
    ('V2G_space','v2g_mobile','v2g_fixed'),('mobile_V2G','v2g_mobile','charge_mobile'),
    ('full_vs_constant','v2g_mobile','constant'),
    ('full_vs_quality','v2g_mobile','quality_mobile'),('full_vs_random','v2g_mobile','random_mobile'))
EVALUATION_SCOPE='independent descriptive AC evaluation; no calibrated rejection claim'


def read_json(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as f:
        return json.load(f)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def number(value):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def stats(values):
    values=[x for x in map(number,values) if x is not None]
    return dict(n=len(values),mean=statistics.fmean(values) if values else None,
        median=statistics.median(values) if values else None,min=min(values) if values else None,
        max=max(values) if values else None,positive_count=sum(v>0 for v in values),
        zero_count=sum(v==0 for v in values),negative_count=sum(v<0 for v in values))


def family_key(supports):
    return tuple(sorted(tuple(sorted(s)) for s in supports if 1<len(s)<3))


def true_candidate(trial):
    wanted=family_key(trial['truth']['family'])
    found=[r['name'] for r in trial['proposal']['bank'] if family_key(r['supports'])==wanted]
    return found[0] if len(found)==1 else None


def signed_margin(fits,true_name,names):
    good={r['name']:number(r.get('loss')) for r in fits if r.get('success')}
    if true_name is None or set(good)!=set(names) or any(v is None for v in good.values()):
        return dict(M=None,half_M=None,correct_min_loss=None,winner_is_true=None,winner=None,nearest_wrong=None)
    nearest=min((n for n in names if n!=true_name),key=lambda n:(good[n],n))
    winner=min(names,key=lambda n:(good[n],n));margin=good[nearest]-good[true_name]
    return dict(M=margin,half_M=.5*margin,correct_min_loss=good[true_name]<=min(good.values())+1e-8,
        winner_is_true=winner==true_name,winner=winner,nearest_wrong=nearest)


def same_curve(left,right):
    if left is None or right is None:return False
    if isinstance(left,(list,tuple)) or isinstance(right,(list,tuple)):
        return (isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)) and len(left)==len(right)
            and all(same_curve(a,b) for a,b in zip(left,right)))
    a,b=number(left),number(right)
    return a is not None and b is not None and abs(a-b)<=1e-7


def prediction_summary(prediction,resolution):
    if not prediction:return dict(mean=None,worst=None,unresolved=None,pairs=0,success=False)
    pairs=prediction.get('pairs',[]);scores=[number(r.get('added')) for r in pairs]
    return dict(mean=number(prediction.get('mean_added_score')),worst=number(prediction.get('worst_added_score')),
        unresolved=sum(x<=resolution for x in scores if x is not None),pairs=len(pairs),
        available_pairs=sum(x is not None for x in scores),success=bool(prediction.get('success')))


def selected_operation(selection,name,row):
    if name=='none':return None
    if name=='constant':return selection.get('constant_operation')
    record=selection.get('methods',{}).get(name,{})
    if name in DESIGNS and not (record.get('guard') or {}).get('accepted'):
        return selection.get('constant_operation')
    op=record.get('operation')
    if op and same_curve(op.get('u_kw'),row.get('power_kw')):return op
    if op and name not in DESIGNS:return op
    return None


def analyze_trial(trial,*,resolution=.01):
    selection=trial['selection'];names=[r['name'] for r in trial['proposal']['bank']]
    true_name=true_candidate(trial);family='star' if not trial['truth']['family'] else 'cherry_'+''.join(map(str,trial['truth']['family'][0]))
    saved={r['method']:r for r in trial['rows']};method_rows=[];pair_rows=[];noiseless_rows=[]
    constant=saved.get('constant',{});predicted=selection.get('predicted_AC',{})
    for name in METHODS:
        row=saved.get(name,{});record=selection.get('methods',{}).get(name,{})
        guard=record.get('guard') or {};raw=record.get('raw') or {}
        pred=predicted.get(name)
        pred_source=name
        if pred is None and same_curve(row.get('power_kw'),constant.get('power_kw')):
            pred=predicted.get('constant');pred_source='constant_same_curve'
        score=prediction_summary(pred,resolution);rawscore=prediction_summary(predicted.get(name+'__raw'),resolution)
        margin=signed_margin(row.get('fits',[]),true_name,names) if row.get('status')=='complete' else signed_margin([],true_name,names)
        mean=row.get('noiseless_AC_refit') or {}
        noiseless=signed_margin(mean.get('fits',[]),true_name,names) if mean.get('all_candidates') else signed_margin([],true_name,names)
        detail={r['candidate']:r for r in mean.get('candidate_distances',[])}
        for candidate in names:
            detail.setdefault(candidate,dict(candidate=candidate,success=False,
                reason='no_saved_noiseless_candidate:'+row.get('status','missing_method'),
                future_channel_SSE=None))
        nearest=detail.get(noiseless['nearest_wrong'],{})
        wrong_details=[d for c,d in detail.items() if c!=true_name]
        mean_added=stats(d.get('added_half_SSE') for d in wrong_details)
        mean_voltage=stats((d.get('future_channel_SSE') or {}).get('v') for d in wrong_details)
        op=selected_operation(selection,name,row)
        rounds=([record['before_recenter']] if record.get('before_recenter') else [])+([raw] if raw else [])
        masters=[it['master'] for round_ in rounds for it in round_.get('iterations',[]) if 'master' in it]
        actual_masters=[m for m in masters if number(m.get('n_variables')) is not None and m['n_variables']>0]
        best=(raw.get('evaluations') or [{}])[raw.get('best_evaluation_index',0)]
        r=dict(seed=trial['seed'],phase=trial['phase'],family=family,domain=trial['domain'],method=name,
            status=row.get('status','missing_method'),candidate_miss=row.get('candidate_miss',true_name is None),
            actual_M=margin['M'],actual_half_M=margin['half_M'],correct_min_loss=margin['correct_min_loss'],
            winner_is_true=margin['winner_is_true'],winner=margin['winner'],true_name=true_name,
            nearest_wrong=margin['nearest_wrong'],R_relative=number(row.get('true_impedance_R_relative')),
            X_relative=number(row.get('true_impedance_X_relative')),
            predicted_mean=score['mean'],predicted_worst=score['worst'],
            predicted_unresolved=score['unresolved'],predicted_pair_count=score['pairs'],
            predicted_success=score['success'],prediction_source=pred_source,
            raw_predicted_mean=rawscore['mean'],raw_predicted_worst=rawscore['worst'],
            raw_predicted_unresolved=rawscore['unresolved'],
            noiseless_joint_M=noiseless['M'],noiseless_joint_half_M=noiseless['half_M'],
            noiseless_nearest_wrong=noiseless['nearest_wrong'],
            noiseless_nearest_wrong_future_V_SSE=number((nearest.get('future_channel_SSE') or {}).get('v')),
            noiseless_wrong_mean_added=mean_added['mean'] if len(wrong_details)==3 and mean_added['n']==3 else None,
            noiseless_wrong_mean_future_V_SSE=mean_voltage['mean'] if len(wrong_details)==3 and mean_voltage['n']==3 else None,
            noiseless_fit_complete=bool(mean.get('all_candidates')),
            grid_net_kwh=number(row.get('grid_net_kwh')),grid_throughput_kwh=number(row.get('grid_throughput_kwh')),
            drive_kwh=0. if name=='none' else number((op or {}).get('travel_energy_kwh')),
            worker_minutes=0. if name=='none' else number((op or {}).get('driver_cost')),
            conversion_loss_kwh=0. if name=='none' else number((op or {}).get('conversion_loss_kwh')),
            moves=0 if name=='none' else (op or {}).get('moves'),
            terminal_energy_kwh=None if not op or not op.get('energy_kwh') else op['energy_kwh'][-1],
            selected_operation_feasible=None if op is None else bool(op.get('feasible')),
            same_battery_service=name!='none',guard_accepted=guard.get('accepted') if name in DESIGNS else None,
            fallback_to_constant=(not bool(guard.get('accepted'))) if name in DESIGNS else None,
            guard_reason=guard.get('reason',record.get('error')),guard_predicted_gain=number(guard.get('gain')),
            raw_status=raw.get('status'),raw_completed=raw.get('completed'),raw_unfinished=raw.get('unfinished'),
            proxy_lower=number(raw.get('objective_lower')),proxy_upper=number(raw.get('objective_upper')),
            proxy_gap=number(raw.get('gap')),raw_best_origin=best.get('origin'),
            main_MILP_calls=len(actual_masters),main_MILP_incumbents=sum(m.get('u') is not None for m in actual_masters),
            QP_calls=sum(rnd.get('qp_calls',0) for rnd in rounds),
            raw_solver_seconds=sum(rnd.get('seconds',0.) for rnd in rounds),recentered=bool(record.get('recentered')),
            history_seconds=trial.get('seconds'),selection_seconds=selection.get('seconds'),
            actual_fit_cache_hit=row.get('cache_hit'),actual_fit_failures=len(row.get('failures',{})))
        method_rows.append(r)
        for stage,prediction in (('selected',pred),('raw',predicted.get(name+'__raw'))):
            if prediction is None:continue
            for pair in prediction.get('pairs',[]):
                added=number(pair.get('added'))
                pair_rows.append(dict(seed=trial['seed'],family=family,method=name,stage=stage,a=pair['a'],b=pair['b'],
                    success=bool(pair.get('success')),added=added,joint=number(pair.get('joint')),
                    self_relative=number(pair.get('self_relative')),weight=number(pair.get('weight')),
                    numerical_unresolved=None if added is None else added<=resolution))
        for candidate,d in detail.items():
            v=d.get('future_channel_SSE') or {}
            noiseless_rows.append(dict(seed=trial['seed'],family=family,method=name,candidate=candidate,
                is_wrong=candidate!=true_name,nearest_wrong=candidate==noiseless['nearest_wrong'],
                success=bool(d.get('success')),reason=d.get('reason'),joint_SSE=number(d.get('attained_joint_SSE')),
                historical_SSE=number(d.get('historical_SSE')),added_half_SSE=number(d.get('added_half_SSE')),
                future_V_SSE=number(v.get('v')),future_P_SSE=number(v.get('p')),future_Q_SSE=number(v.get('q')),
                future_master_P_SSE=number(v.get('master_p')),future_master_Q_SSE=number(v.get('master_q'))))
    byname={r['method']:r for r in method_rows};paired=[]
    fields=('actual_M','actual_half_M','noiseless_joint_M','predicted_mean','predicted_worst',
            'noiseless_nearest_wrong_future_V_SSE','noiseless_wrong_mean_added','noiseless_wrong_mean_future_V_SSE',
            'grid_net_kwh','grid_throughput_kwh','drive_kwh','worker_minutes')
    for label,a,b in CONTRASTS:
        left,right=byname[a],byname[b]
        item=dict(seed=trial['seed'],phase=trial['phase'],family=family,contrast=label,a=a,b=b,
            complete_actual_pair=left['actual_M'] is not None and right['actual_M'] is not None)
        for field in fields:
            lv,rv=number(left[field]),number(right[field]);item['delta_'+field]=None if lv is None or rv is None else lv-rv
        item['delta_correct_min_loss']=(int(left['correct_min_loss'])-int(right['correct_min_loss'])
            if left['correct_min_loss'] is not None and right['correct_min_loss'] is not None else None)
        paired.append(item)
    return dict(seed=trial['seed'],family=family,method_rows=method_rows,paired_rows=paired,
        predicted_pair_rows=pair_rows,noiseless_candidate_rows=noiseless_rows,
        historical_failures=trial.get('historical_failures',{}),selection_failures=selection.get('failures',{}),
        seconds=trial.get('seconds'),selection_seconds=selection.get('seconds'),
        predicted_fit_calls=selection.get('predicted_fit_calls'),predicted_AC_calls=selection.get('predicted_ac_calls'))


def aggregate_method(rows):
    metrics=('actual_M','actual_half_M','predicted_mean','predicted_worst','predicted_unresolved',
        'noiseless_joint_M','noiseless_nearest_wrong_future_V_SSE','noiseless_wrong_mean_added',
        'noiseless_wrong_mean_future_V_SSE','R_relative','X_relative',
        'grid_net_kwh','grid_throughput_kwh','drive_kwh','worker_minutes','proxy_gap','raw_solver_seconds')
    eligible=[r for r in rows if r['correct_min_loss'] is not None]
    return dict(available_histories=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
        complete_actual_histories=len(eligible),correct_min_loss_count=sum(r['correct_min_loss'] for r in eligible),
        exact_winner_correct_count=sum(bool(r['winner_is_true']) for r in eligible),
        candidate_miss_count=sum(bool(r['candidate_miss']) for r in rows),
        accepted_count=sum(r['guard_accepted'] is True for r in rows),
        fallback_count=sum(r['fallback_to_constant'] is True for r in rows),
        proxy_completed_count=sum(r['raw_completed'] is True for r in rows),
        proxy_unfinished_count=sum(r['raw_unfinished'] is True for r in rows),
        main_MILP_calls=sum(r['main_MILP_calls'] for r in rows),
        main_MILP_incumbents=sum(r['main_MILP_incumbents'] for r in rows),
        QP_calls=sum(r['QP_calls'] for r in rows),metrics={key:stats(r[key] for r in rows) for key in metrics})


def _current_hashes(freeze):
    result={};root=HERE.parents[1].resolve()
    for name in freeze['source_sha256']:
        path=((root/name) if name.startswith('rnj_wzzt_core/') else HERE/name).resolve()
        path.relative_to(root)
        result[name]=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def _prediction_complete(record,names):
    expected={(a,b) for a in names for b in names if a!=b}
    if not record or not record.get('success') or record.get('required_fit_states')!=16 or record.get('returned_fit_states')!=16:return False
    if {(r.get('a'),r.get('b')) for r in record.get('pairs',[])}!=expected or len(record.get('pairs',[]))!=12:return False
    if not all(r.get('success') for r in record['pairs']):return False
    records=record.get('records',[])
    return (len(records)==4 and {r.get('a') for r in records}==set(names)
        and all(r.get('success') and not r.get('failures') and len(r.get('fits',[]))==4
            and {f.get('name') for f in r['fits']}==set(names)
            and all(f.get('success') for f in r['fits']) for r in records))


def build_pilot_gate(freeze,trials,review,*,trial_hashes=None,current_hashes=None):
    """Technical readiness only. A performance improvement is never required."""
    conditions=[]
    def check(name,passed,detail=None):conditions.append(dict(name=name,passed=bool(passed),detail=detail))
    planned=freeze.get('seeds',[]);expected=freeze.get('config',{}).get('pilot_seeds',[])
    seen=[t.get('seed') for t in trials];hashes=freeze.get('source_sha256',{})
    check('four_planned_pilot_histories',freeze.get('phase')=='pilot' and len(planned)==4 and len(set(planned))==4 and planned==expected)
    check('all_four_histories_present',len(trials)==4 and len(set(seen))==4 and set(seen)==set(planned),dict(seen=seen,planned=planned))
    current=_current_hashes(freeze) if current_hashes is None else current_hashes
    check('current_scientific_source_matches_freeze',bool(hashes) and current==hashes)
    source=(review or {}).get('source_audit',{})
    source_rows={r.get('name'):r.get('sha256') for r in source.get('rows',[])}
    reviewed_hashes={r.get('name'):r.get('sha256') for r in (review or {}).get('trial_hashes',[])}
    supplied_hashes={r.get('name'):r.get('sha256') for r in (trial_hashes or [])}
    check('independent_review_success',bool(review) and review.get('success') and not review.get('review_errors')
        and review.get('audited_histories')==4 and review.get('available_independent_histories')==4
        and not review.get('missing_histories') and review.get('phase')=='pilot')
    check('independent_review_current_source_identity',source.get('success') and source.get('current_source_matches')
        and source_rows==hashes and all(r.get('current_sha256')==r.get('sha256') for r in source.get('rows',[])))
    check('independent_review_exact_trial_files',bool(supplied_hashes) and reviewed_hashes==supplied_hashes,
        'external review must cover these exact compressed trial bytes')
    for trial in trials:
        prefix=str(trial.get('seed'))+':';selection=trial.get('selection',{});bank=trial.get('proposal',{}).get('bank',[])
        names=[r.get('name') for r in bank];ordered={(a,b) for a in names for b in names if a!=b}
        check(prefix+'freeze_identity',trial.get('source_sha256')==hashes and trial.get('config')==freeze.get('config')
            and trial.get('phase')=='pilot' and trial.get('domain')==freeze.get('domain') and not trial.get('source_changed_during_trial'))
        check(prefix+'four_candidates_twelve_tasks',len(names)==4 and len(set(names))==4
            and selection.get('all_candidates')==4 and selection.get('all_ordered_pairs')==12
            and len(selection.get('context_models',[]))==12
            and {(m.get('a'),m.get('b')) for m in selection.get('context_models',[])}==ordered)
        check(prefix+'history_complete',not trial.get('historical_failures') and len(trial.get('historical_fits',[]))==4
            and {f.get('name') for f in trial.get('historical_fits',[])}==set(names)
            and all(f.get('success') for f in trial.get('historical_fits',[])))
        check(prefix+'selector_information_boundary',selection.get('uses_truth') is False and selection.get('uses_future_noise') is False)
        check(prefix+'constant_physical_plan',selection.get('constant_operation',{}).get('feasible'))
        check(prefix+'constant_prediction_all16',_prediction_complete(selection.get('predicted_AC',{}).get('constant'),names))
        rows={r.get('method'):r for r in trial.get('rows',[])}
        check(prefix+'all_method_denominators',set(rows)==set(METHODS) and len(trial.get('rows',[]))==len(METHODS))
        for name in DESIGNS:
            rec=selection.get('methods',{}).get(name,{});raw=rec.get('raw') or {};lower=number(raw.get('objective_lower'));upper=number(raw.get('objective_upper'))
            check(prefix+name+':final_proxy_interval',lower is not None and upper is not None and upper>=lower,
                dict(lower=lower,upper=upper,status=raw.get('status')))
            required=raw.get('required_pairs',[]);modelnames={m.get('name') for m in (rec.get('recenter_models') or selection.get('context_models',[]))}
            check(prefix+name+':all_twelve_competitors',len(required)==12 and set(required)==modelnames)
            evaluations=raw.get('evaluations') or [];index=raw.get('best_evaluation_index')
            best=evaluations[index] if isinstance(index,int) and 0<=index<len(evaluations) else {}
            check(prefix+name+':best_operation_feasible',best.get('operation_validation',{}).get('feasible')
                and len(best.get('qp_results',[]))==12)
            row=rows.get(name,{});op=selected_operation(selection,name,row)
            check(prefix+name+':executed_operation_feasible',bool(op) and op.get('feasible')
                and row.get('power_kw') is not None and row.get('status') not in ('physical_voltage_failure','physical_or_fit_exception'))
            guard=rec.get('guard') or {}
            if guard.get('accepted'):
                checks=(rec.get('proxy_AC_checks') or [[]])[-1]
                check(prefix+name+':accepted_point_RMS_gate',len(checks)==12
                    and {(r.get('a'),r.get('b')) for r in checks}==ordered
                    and all(r.get('check',{}).get('physical_box_feasible')
                        and r['check'].get('complete_AC_voltage_feasible')
                        and number(r['check'].get('max_channel_RMS')) is not None
                        and r['check']['max_channel_RMS']<=freeze['config']['proxy_channel_RMS_gate'] for r in checks)
                    and guard.get('proxy_same_witness_gate'))
                check(prefix+name+':accepted_prediction_all16',_prediction_complete(selection.get('predicted_AC',{}).get(name+'__raw'),names)
                    and _prediction_complete(selection.get('predicted_AC',{}).get(name),names))
    ready=bool(conditions) and all(c['passed'] for c in conditions)
    return dict(ready_for_independent_evaluation=ready,source_sha256=hashes,pilot_seeds=list(planned),
        evaluation_scope=EVALUATION_SCOPE,conditions=conditions,failed_conditions=[c['name'] for c in conditions if not c['passed']],
        gate_meaning='technical consistency for independent descriptive evaluation; EV gain is not a passing condition',
        performance_improvement_required=False,formal_rejection_risk_calibrated=False,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def summarize_directory(directory,*,review=None,resolution=.01):
    directory=Path(directory);freeze=read_json(directory/'freeze.json');trials=[];files=[];errors=[];seen=set()
    for path in sorted(directory.glob('trial_*.json.gz')):
        trial=read_json(path);seed=trial.get('seed')
        if seed in seen or seed not in freeze['seeds']:raise ValueError('duplicate or unplanned history')
        if trial.get('source_sha256')!=freeze['source_sha256'] or trial.get('config')!=freeze['config']:
            raise ValueError('trial/freeze scientific identity mismatch: '+path.name)
        seen.add(seed);trials.append(trial);files.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    analyses=[]
    for trial in trials:
        try:analyses.append(analyze_trial(trial,resolution=resolution))
        except (KeyError,ValueError,TypeError,IndexError) as error:
            errors.append(dict(seed=trial['seed'],error=type(error).__name__+':'+str(error)))
    rows=[r for a in analyses for r in a['method_rows']];paired=[r for a in analyses for r in a['paired_rows']]
    method={m:aggregate_method([r for r in rows if r['method']==m]) for m in METHODS}
    pairsummary={label:dict(available_histories=sum(r['contrast']==label for r in paired),
        metrics={key:stats(r[key] for r in paired if r['contrast']==label) for key in paired[0] if key.startswith('delta_')})
        for label,_,_ in CONTRASTS} if paired else {}
    families={f:{m:aggregate_method([r for r in rows if r['family']==f and r['method']==m]) for m in METHODS}
        for f in sorted({a['family'] for a in analyses})}
    result=dict(phase=freeze['phase'],domain=freeze['domain'],directory=str(directory.resolve()),
        planned_independent_histories=len(freeze['seeds']),available_independent_histories=len(trials),
        analyzed_independent_histories=len(analyses),missing_histories=[s for s in freeze['seeds'] if s not in seen],
        source_sha256=freeze['source_sha256'],trial_hashes=files,analysis_errors=errors,
        statistical_unit='one independently generated historical dataset; all methods and 12 directions paired within it',
        resolution=resolution,resolution_meaning='added score <= resolution is numerical unresolved; no statistical rejection test',
        formal_calibration_histories=0,formal_rejection_rule=freeze['config'].get('statistical_rejection_rule'),
        metrics=dict(actual_M='minimum wrong-candidate joint SSE minus true-candidate joint SSE',
            actual_half_M='joint log-likelihood difference M/2 for the same measurement model',
            correct_min_loss='true candidate has the minimum attained loss within 1e-8 SSE',
            predicted_mean='uniform mean of 12 H-only complete-AC forecast added scores',
            noiseless_joint_M='offline H + true noiseless future mean, joint signed M; not selector input',
            proxy_gap='declared bilinear proxy upper minus attained original-box QP lower; not AC gap'),
        method_summary=method,paired_summary=pairsummary,family_summary=families,histories=analyses,
        method_rows=rows,paired_rows=paired,
        predicted_pair_rows=[r for a in analyses for r in a['predicted_pair_rows']],
        noiseless_candidate_rows=[r for a in analyses for r in a['noiseless_candidate_rows']],
        independent_review=review,analysis_source=dict(file=Path(__file__).name,
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inside_scientific_freeze=False,
            new_AC_fits=0,new_selector_runs=0))
    return result,freeze,trials


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def fmt(value):return '—' if value is None else f'{value:.4g}' if isinstance(value,(int,float)) else str(value)


def markdown(summary):
    n=summary['available_independent_histories'];out=[f"# EV原域研究：{summary['phase']}描述汇总",'',
        f"已保存 {n}/{summary['planned_independent_histories']} 条独立历史，完成分析 {summary['analyzed_independent_histories']} 条。每条历史是一个统计单位；12个方向和多种方法均为同一历史内的配对结果。",'',
        'M为最邻近错误候选的联合SSE减真候选SSE，M/2为对应对数似然差；正值表示真候选的取得损失更小。预测均值为H-only完整AC的12方向平均附加分数。', '',
        '|方法|实际拟合完整/历史|真候选最小损失|平均M|最小M|平均预测分数|最小预测最坏对|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in summary['method_summary'].items():
        v=r['metrics'];out.append(f"|{name}|{r['complete_actual_histories']}/{n}|{r['correct_min_loss_count']}/{r['complete_actual_histories']}|{fmt(v['actual_M']['mean'])}|{fmt(v['actual_M']['min'])}|{fmt(v['predicted_mean']['mean'])}|{fmt(v['predicted_worst']['min'])}|")
    out+=['','## 逐历史配对差','', '差值为前者减后者。正的ΔM表示该历史的真候选相对错误候选更容易区分；这些差值不附带显著性或风险保证。','',
        '|比较|有效配对历史|平均ΔM|中位ΔM|ΔM正/零/负|平均Δ网侧电量(kWh)|',
        '|---|---:|---:|---:|---:|---:|']
    for label,r in summary['paired_summary'].items():
        v=r['metrics']['delta_actual_M'];out.append(f"|{label}|{v['n']}/{r['available_histories']}|{fmt(v['mean'])}|{fmt(v['median'])}|{v['positive_count']}/{v['zero_count']}/{v['negative_count']}|{fmt(r['metrics']['delta_grid_net_kwh']['mean'])}|")
    out+=['','## 服务与求解','',
        '服务比较固定相同终端电池能量。移动和双向转换可能增加网侧电量、驾驶耗能和吞吐；无EV仅作同观测时长参照。', '',
        '|方法|平均净电量(kWh)|平均吞吐(kWh)|平均驾驶(kWh)|平均工时(min)|接受/历史|回退/历史|代理完成/历史|实际主MILP次数|平均代理gap|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in summary['method_summary'].items():
        v=r['metrics'];out.append(f"|{name}|{fmt(v['grid_net_kwh']['mean'])}|{fmt(v['grid_throughput_kwh']['mean'])}|{fmt(v['drive_kwh']['mean'])}|{fmt(v['worker_minutes']['mean'])}|{r['accepted_count']}/{n}|{r['fallback_count']}/{n}|{r['proxy_completed_count']}/{n}|{r['main_MILP_calls']}|{fmt(v['proxy_gap']['mean'])}|")
    unresolved=summary['resolution'];out+=['',
        f"逐对附加预测分数≤{unresolved:g}记作数值未决，详见 predicted_pairs.csv；这是数值分辨率，未进行统计拒绝检验。弱对保留在12方向中。",'',
        'noiseless_candidates.csv保存每个错误候选在真实无噪声未来计量上的电压SSE及历史/未来成本；它只用于离线评价。per_history.csv保存M/2、逐对未决数、R/X误差、优化终止与资源；paired_differences.csv保留每条历史的时间/V2G/空间差。', '',
        f"独立风险校准历史数为0；这些结果属于声明模拟分布下的描述评价。分析错误 {len(summary['analysis_errors'])} 条，缺失历史 {summary['missing_histories']}。",'']
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--review',type=Path);p.add_argument('--pilot-gate',type=Path);p.add_argument('--resolution',type=float,default=.01)
    args=p.parse_args()
    if not math.isfinite(args.resolution) or args.resolution<0:raise ValueError('finite nonnegative numerical resolution required')
    review=read_json(args.review) if args.review else None
    summary,freeze,trials=summarize_directory(args.directory,review=review,resolution=args.resolution)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,key in [('per_history.csv','method_rows'),('paired_differences.csv','paired_rows'),
            ('predicted_pairs.csv','predicted_pair_rows'),('noiseless_candidates.csv','noiseless_candidate_rows')]:
        write_csv(args.output/name,summary[key])
    if args.pilot_gate:
        gate=build_pilot_gate(freeze,trials,review,trial_hashes=summary['trial_hashes'])
        if summary['analysis_errors']:
            gate['ready_for_independent_evaluation']=False;gate['failed_conditions'].append('analysis_errors')
        write_json(args.pilot_gate,gate);summary['pilot_gate']=gate
    write_json(args.output/'summary.json',summary)
    (args.output/'summary.md').write_text(markdown(summary),encoding='utf-8')
    print(json.dumps(dict(available_histories=summary['available_independent_histories'],
        analyzed_histories=summary['analyzed_independent_histories'],analysis_errors=summary['analysis_errors'],
        ready=None if not args.pilot_gate else summary['pilot_gate']['ready_for_independent_evaluation'])))


if __name__=='__main__':main()
