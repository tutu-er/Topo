"""Read-only compact-policy evidence analysis, outside the science freeze.

Only saved JSON is read. No selector, simulator, AC fit, optimizer or random
action generator is imported. One historical sample is one statistical unit.
Directory summaries stay separate; matched ablations require identical data.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
from . import ev_original_analysis as core

METHODS=(*core.METHODS,'capped_v2g_mobile')
DESIGNS=(*core.DESIGNS,'capped_v2g_mobile')
SUPPORTED_METHODS=(*METHODS,'refined_v2g_mobile')
ALL_DESIGNS=(*DESIGNS,'refined_v2g_mobile')
CONTRASTS=(*core.CONTRASTS,
    ('capped_vs_mean','capped_v2g_mobile','v2g_mobile'),
    ('capped_vs_random','capped_v2g_mobile','random_mobile'),
    ('capped_vs_constant','capped_v2g_mobile','constant'),
    ('refined_vs_cap','refined_v2g_mobile','capped_v2g_mobile'),
    ('refined_vs_random','refined_v2g_mobile','random_mobile'),
    ('refined_vs_constant','refined_v2g_mobile','constant'))
PAIRED_FIELDS=('actual_M','actual_half_M','noiseless_M','noiseless_half_M',
    'predicted_mean','predicted_worst','predicted_cap20_mean','R_relative','X_relative',
    'grid_net_kwh','grid_throughput_kwh','drive_kwh','worker_minutes','moves',
    'proxy_gap','proxy_seconds','new_MILP_proposals','selected_is_new_MILP_proposal')
number=core.number
stats=core.stats


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def _flatten(value):
    if value is None:return None
    if isinstance(value,(list,tuple)):
        answer=[]
        for item in value:
            part=_flatten(item)
            if part is None:return None
            answer.extend(part)
        return answer
    scalar=number(value)
    return None if scalar is None else [scalar]


def same_curve(a,b):
    left,right=_flatten(a),_flatten(b)
    return left is not None and right is not None and core.same_curve(left,right)


def declared_methods(trial):
    """New method absent from v1 is unplanned, not a failed v1 experiment."""
    config=trial.get('config') or {}
    explicit=trial.get('methods') or config.get('methods') or config.get('method_names')
    if explicit is not None:
        if (not isinstance(explicit,(list,tuple)) or len(set(explicit))!=len(explicit)
                or not set(METHODS)<=set(explicit) or not set(explicit)<=set(SUPPORTED_METHODS)):
            raise ValueError('declared method list must contain the legacy methods and supported optional methods')
        return tuple(explicit)
    saved={r['method'] for r in trial.get('rows') or []}
    recorded=set((trial.get('selection') or {}).get('methods') or {})
    return SUPPORTED_METHODS if 'refined_v2g_mobile' in saved|recorded else METHODS


def operations_config(trial):
    """Physical device data plus the actual command derating, when declared."""
    config=trial.get('config') or {};physical=config.get('physical',config)
    saved=trial.get('operations_config')
    device=dict(physical.get('device') or {})
    if saved is not None:
        if not isinstance(saved,dict):raise ValueError('operations_config must be a saved mapping')
        values=saved.get('device',saved)
        if isinstance(values,dict):device.update(values)
    command=(device.get('command_power_kw') if saved is not None else None)
    if command is None and saved is not None:
        command=device.get('p_charge_max_kw',device.get('power_kw'))
    if command is None:command=config.get('command_power_kw',device.get('power_kw'))
    if command is not None:device['command_power_kw']=command
    return device


def identity(trial):
    config=trial.get('config') or {}
    physical=config.get('physical',config)
    truth=trial.get('truth') or {}
    return dict(history_sha256=digest(trial.get('history')),
        truth_sha256=digest(truth),noise_sha256=digest(truth.get('white')),
        measurement_sha256=digest(trial.get('measurement_models')),
        device_sha256=digest(operations_config(trial)),
        data_sha256=digest(dict(history=trial.get('history'),truth=truth,
            measurement_models=trial.get('measurement_models'))))


def prediction_summary(prediction,*,resolution=.01,cap=20.):
    pairs=(prediction or {}).get('pairs') or []
    available=[r for r in pairs if number(r.get('added')) is not None]
    valid=bool(prediction and prediction.get('success') and pairs
        and len(available)==len(pairs) and all(r.get('success') for r in pairs))
    weights=[number(r.get('weight',1/len(pairs))) for r in pairs]
    if valid and (any(w is None or w<0 for w in weights) or abs(sum(weights)-1.)>1e-8):
        raise ValueError('prediction task weights must be normalized and nonnegative')
    values=[number(r['added']) for r in pairs] if valid else []
    return dict(success=valid,pairs=len(pairs),available_pairs=len(available),
        mean=sum(w*v for w,v in zip(weights,values)) if valid else None,
        worst=min(values) if valid else None,
        cap20_mean=sum(w*min(v,cap) for w,v in zip(weights,values)) if valid else None,
        unresolved=sum(v<=resolution for v in values) if valid else None,
        unresolved_among_available=sum(number(r['added'])<=resolution for r in available),
        unresolved_definition='added score <= declared numerical resolution; not a rejection test')


def _losses(fits):
    return {r['name']:number(r.get('loss')) for r in fits or []
            if r.get('success') and number(r.get('loss')) is not None}


def _norm_error(fit,truth,key):
    if not fit or not fit.get('success'):return None
    estimated=fit.get(key);real=truth.get(key)
    if not isinstance(estimated,list) or not isinstance(real,list) or len(estimated)!=len(real):return None
    if any(number(x) is None for x in [*estimated,*real]):return None
    denominator=sum(x*x for x in real)
    return (sum((x-y)**2 for x,y in zip(estimated,real))/denominator)**.5 if denominator>0 else None


def _operation(selection,name,row):
    if name=='none':return None
    operation=(selection.get('constant_operation') if name=='constant'
        else (selection.get('methods',{}).get(name) or {}).get('operation'))
    if operation and same_curve(operation.get('u_kw'),row.get('power_kw')):return operation
    return None


def _rounds(record):
    return ([record['before_recenter']] if record.get('before_recenter') else [])+(
        [record['raw']] if record.get('raw') else [])


def _unique_curves(curves):
    answer=[]
    for curve in curves:
        if _flatten(curve) is not None and not any(same_curve(curve,old) for old in answer):answer.append(curve)
    return answer


def trace_source_lineage(method,method_rows):
    """Trace saved parent-domain origins, separately from new-curve discovery.

    MILP is the declared optimization origin; terminal_MILP_new_curve records
    whether that terminal domain actually dispatched a newly proposed curve.
    Quality-price MILPs have a separate quality terminal.
    """
    bank=method_rows if isinstance(method_rows,dict) else {r['method']:r for r in method_rows}
    path=[method];visited=set();current=method;terminal='unknown';error=None
    terminal_origin=None;terminal_method=None;new_curve=None
    while True:
        if current in visited:error='cycle';break
        visited.add(current)
        row=bank.get(current)
        if row is None:error='missing_parent';break
        if current=='none':terminal='none';terminal_origin=current;terminal_method=current;break
        if current=='constant':terminal='constant';terminal_origin=current;terminal_method=current;break
        if current.startswith('quality'):
            terminal='quality';terminal_origin=current;terminal_method=current;break
        if current.startswith('random'):
            terminal='random';terminal_origin=current;terminal_method=current;break
        origin=row.get('selected_origin')
        if not isinstance(origin,str) or not origin:error='missing_origin';break
        path.append(origin)
        if origin=='MILP':
            terminal='MILP';terminal_origin=origin;terminal_method=current
            new_curve=row.get('selected_is_new_MILP_proposal')
            if isinstance(new_curve,str):new_curve={'True':True,'False':False}.get(new_curve)
            if not isinstance(new_curve,bool):new_curve=None
            break
        if origin=='constant' or origin.startswith('quality') or origin.startswith('random'):
            terminal='constant' if origin=='constant' else 'quality' if origin.startswith('quality') else 'random'
            terminal_origin=origin;terminal_method=current;break
        if origin not in bank:error='missing_parent';break
        current=origin
    if terminal=='random' and method in ALL_DESIGNS:error='random_in_primary_chain'
    from_milp=terminal=='MILP';ancestral=from_milp and terminal_method!=method
    return dict(source_lineage=path,source_lineage_terminal=terminal,
        source_lineage_terminal_origin=terminal_origin,source_lineage_terminal_method=terminal_method,
        source_lineage_from_MILP=from_milp,source_lineage_ancestral_MILP=ancestral,
        source_lineage_terminal_MILP_new_curve=new_curve if from_milp else None,
        source_lineage_ancestral_new_MILP=(bool(new_curve) if new_curve is not None else None) if ancestral else False,
        source_lineage_error=error)


def analyze_trial(trial,*,resolution=.01,label=None):
    methods=declared_methods(trial)
    names=[r['name'] for r in trial['proposal']['bank']]
    if len(names)!=len(set(names)):raise ValueError('duplicate candidates')
    true_name=core.true_candidate(trial)
    family='star' if not trial['truth']['family'] else 'cherry_'+''.join(map(str,trial['truth']['family'][0]))
    selection=trial.get('selection') or {};records=selection.get('methods') or {}
    rows=trial.get('rows') or [];by_method={r['method']:r for r in rows}
    if len(by_method)!=len(rows):raise ValueError('duplicate method rows')
    unknown=set(by_method)-set(methods)
    if unknown:raise ValueError('undeclared methods '+repr(sorted(unknown)))
    common=dict(label=label,seed=trial['seed'],phase=trial.get('phase'),family=family,
        domain=trial.get('domain'),prediction=trial.get('prediction',selection.get('prediction')),
        policy_id=trial.get('policy_id'),**identity(trial))
    output=[];candidate_rows=[];task_rows=[];master_rows=[]
    # Saved row order is execution order. Attribute reused noiseless fits in
    # that order, rather than the report's method display order.
    mean_cache_flags={};mean_fitted_curves=[]
    for row in rows:
        if not row.get('noiseless_AC_refit'):continue
        hit=any(same_curve(row.get('power_kw'),curve) for curve in mean_fitted_curves)
        mean_cache_flags[row['method']]=hit
        if not hit:mean_fitted_curves.append(row.get('power_kw'))
    predictions=selection.get('predicted_AC') or {}
    constant=by_method.get('constant') or {}
    device=operations_config(trial)
    config=trial.get('config') or {};physical=config.get('physical',config)
    rating=(physical.get('device') or {}).get('power_kw')
    command_limit=number(device.get('command_power_kw'));hardware_rating=number(rating)
    tracking=number((config.get('ensemble') or {}).get('tracking_fraction'))
    tracking_limit=command_limit*(1+tracking) if command_limit is not None and tracking is not None else None
    for name in methods:
        row=by_method.get(name) or {};record=records.get(name) or {};raw=record.get('raw') or {}
        fits=row.get('fits') or [];fit_map={r['name']:r for r in fits}
        losses=_losses(fits);noiseless=row.get('noiseless_AC_refit') or {}
        nf=noiseless.get('fits') or [];nl=_losses(nf)
        actual=core.signed_margin(fits,true_name,names) if row.get('status')=='complete' else core.signed_margin([],true_name,names)
        mean=core.signed_margin(nf,true_name,names) if noiseless.get('all_candidates') else core.signed_margin([],true_name,names)
        pred=predictions.get(name);pred_source=name if pred else None
        if pred is None and same_curve(row.get('power_kw'),constant.get('power_kw')):
            pred=predictions.get('constant');pred_source='constant_same_curve' if pred else None
        if pred is None:
            cached=[p for p in selection.get('all_prediction_records') or []
                    if same_curve(row.get('power_kw'),p.get('power_kw'))]
            if len(cached)==1:pred=cached[0];pred_source='saved_H_only_cache_same_curve'
        summary=prediction_summary(pred,resolution=resolution)
        raw_pred=prediction_summary(record.get('raw_prediction'),resolution=resolution)
        op=_operation(selection,name,row)
        rounds=_rounds(record);masters=[]
        for round_index,round_ in enumerate(rounds):
            for it in round_.get('iterations') or []:
                master=it.get('master') or {}
                if number(master.get('n_variables')) is None:continue
                masters.append(master)
                master_rows.append(dict(**common,method=name,round=round_index,iteration=it.get('iteration'),
                    master_form=master.get('master_form'),fallback=master.get('shared_square_fallback'),
                    status=master.get('status'),solver_success=master.get('solver_success'),
                    incumbent=number(master.get('surrogate_incumbent')),upper=number(master.get('objective_upper')),
                    solver_best_bound=number(master.get('solver_best_bound')),
                    bound_consistent=master.get('bound_consistent'),has_action=master.get('u') is not None,
                    variables=master.get('n_variables'),rows=master.get('n_rows'),binary=master.get('n_binary'),
                    operation_binary=master.get('n_operation_binary'),shared_square_binary=master.get('n_shared_square_binary'),
                    residual_binary=master.get('n_residual_binary'),product_binary=master.get('n_product_binary'),
                    seconds=number(master.get('seconds')),mip_gap=number(master.get('solver_mip_gap')),
                    incumbent_is_true_score_lower=master.get('incumbent_is_true_score_lower')))
        initial=record.get('initial_candidates') or []
        initial_curves=[(r.get('operation') or {}).get('u_kw') for r in initial]
        initial_keys=sorted({tuple(round(v,8) for v in flat) for curve in initial_curves
                             if (flat:=_flatten(curve)) is not None})
        proposals=[m['u'] for m in masters if m.get('u') is not None]
        unique=_unique_curves(proposals)
        new=[u for u in unique if not any(same_curve(u,v) for v in initial_curves)]
        evaluations=raw.get('evaluations') or [];best_index=raw.get('best_evaluation_index')
        best=evaluations[best_index] if isinstance(best_index,int) and 0<=best_index<len(evaluations) else {}
        origin=record.get('selected_origin');initial_score=number(record.get('initial_best_score'));final_score=number(record.get('selected_score'))
        selected_is_new=(origin=='MILP' and any(same_curve(row.get('power_kw'),u) for u in new)) if name in ALL_DESIGNS else None
        cost_units=(op or {}).get('cost_units')
        worker=0. if name=='none' else number((op or {}).get('driver_cost'))
        ledger=dict(grid_net_kwh=number(row.get('grid_net_kwh')),
            grid_throughput_kwh=number(row.get('grid_throughput_kwh')),
            drive_kwh=0. if name=='none' else number((op or {}).get('travel_energy_kwh')),
            worker_cost=worker,worker_cost_units=cost_units,
            worker_minutes=worker if name=='none' or cost_units=='worker_minutes' else None,
            moves=0 if name=='none' else (op or {}).get('moves'),
            conversion_loss_kwh=0. if name=='none' else number((op or {}).get('conversion_loss_kwh')),
            terminal_energy_kwh=number((op.get('energy_kwh') or [None])[-1]) if op else None)
        mean_cache_hit=mean_cache_flags.get(name,False)
        actual_new=not bool(row.get('cache_hit'))
        r=dict(**common,method=name,status=row.get('status','missing_method'),
            candidate_count=len(names),candidate_miss=bool(true_name is None or row.get('candidate_miss')),true_name=true_name,
            true_candidate_joint_SSE=losses.get(true_name),minimum_joint_SSE=min(losses.values()) if losses else None,
            actual_M=actual['M'],actual_half_M=actual['half_M'],correct_min_loss=actual['correct_min_loss'],
            exact_winner_is_true=actual['winner_is_true'],winner=actual['winner'],nearest_wrong=actual['nearest_wrong'],
            complete_noisy_candidates=len(losses),noisy_fit_failures=len(row.get('failures') or {}),
            noiseless_M=mean['M'],noiseless_half_M=mean['half_M'],noiseless_complete=bool(noiseless.get('all_candidates')),
            noiseless_true_candidate_SSE=nl.get(true_name),noiseless_minimum_SSE=min(nl.values()) if nl else None,
            noiseless_complete_candidates=len(nl),
            R_relative=number(row.get('true_impedance_R_relative')),
            X_relative=number(row.get('true_impedance_X_relative')),
            predicted_mean=summary['mean'],predicted_worst=summary['worst'],predicted_cap20_mean=summary['cap20_mean'],
            predicted_complete=summary['success'],predicted_tasks=summary['pairs'],predicted_available_tasks=summary['available_pairs'],
            predicted_unresolved=summary['unresolved'],prediction_source=pred_source,
            predicted_unresolved_among_available=summary['unresolved_among_available'],
            raw_predicted_mean=raw_pred['mean'],raw_predicted_worst=raw_pred['worst'],
            cap=record.get('cap'),selected_task_score=final_score,
            selected_origin=origin,raw_best_origin=best.get('origin'),initial_best_origin=record.get('initial_best_origin'),
            initial_best_score=initial_score,selected_is_new_MILP_proposal=selected_is_new,
            initial_action_count=len(initial_keys),initial_actions_sha256=digest(initial_keys),
            initial_origins=[r.get('origin') for r in initial],
            configured_exchange_budget={key:(config.get('exchange') or {}).get(key) for key in
                ('total_seconds','solver_seconds','max_iterations','control_segments','max_segments')},
            inherited_or_initial_retained=(origin!='MILP') if name in ALL_DESIGNS and origin is not None else None,
            retained_constant=(origin=='constant') if name in ALL_DESIGNS and origin is not None else None,
            inheritance_protected=record.get('inheritance_protected'),
            inheritance_check=(final_score>=initial_score-1e-10) if final_score is not None and initial_score is not None else None,
            random_used_by_primary=bool(selection.get('random_used_by_primary') or record.get('random_used')
                or any(str(r.get('origin','')).startswith('random') for r in initial)),
            operation_feasible=None if op is None else bool(op.get('feasible')),
            operation_ledger_matches_executed_curve=op is not None if name!='none' else None,
            raw_proxy_gate=record.get('raw_proxy_gate'),selection_failures=len(record.get('failures') or []),
            proxy_status=raw.get('status'),proxy_completed=raw.get('completed'),proxy_unfinished=raw.get('unfinished'),
            proxy_lower=number(raw.get('objective_lower')),proxy_upper=number(raw.get('objective_upper')),proxy_gap=number(raw.get('gap')),
            QP_calls=sum(v.get('qp_calls',0) for v in rounds),main_MILP_calls=len(masters),
            main_MILP_incumbents=len(proposals),unique_MILP_proposals=len(unique),new_MILP_proposals=len(new),
            max_master_variables=max((m['n_variables'] for m in masters),default=None),
            max_master_binary=max((m['n_binary'] for m in masters),default=None),
            master_seconds=sum(m.get('seconds',0.) for m in masters),
            proxy_seconds=sum(v.get('seconds',0.) for v in rounds),recentered=bool(record.get('recentered')),
            history_seconds=number(trial.get('seconds')),selection_seconds=number(selection.get('seconds')),
            actual_fit_cache_hit=row.get('cache_hit'),
            hardware_rated_power_kw=hardware_rating,command_power_limit_kw=command_limit,
            forecast_tracking_fraction=tracking,forecast_tracking_power_limit_kw=tracking_limit,
            forecast_tracking_within_hardware=(tracking_limit<=hardware_rating+1e-10)
                if tracking_limit is not None and hardware_rating is not None else None,
            commanded_max_abs_power_kw=max((abs(v) for v in _flatten(row.get('power_kw')) or []),default=None),
            commanded_curve_within_limit=(max(abs(v) for v in _flatten(row['power_kw']))<=command_limit+1e-7)
                if command_limit is not None and _flatten(row.get('power_kw')) else None,
            operations_config_sha256=digest(device),
            actual_fit_nominal_calls=(row.get('cost') or {}).get('fit_calls'),
            actual_fit_calls=(row.get('cost') or {}).get('fit_calls',0) if actual_new else 0,
            actual_fit_AC_calls=(row.get('cost') or {}).get('ac_calls',0) if actual_new else 0,
            future_forward_AC_calls=0 if row.get('status') in (None,'operation_or_design_failure') else 1,
            noiseless_fit_cache_by_saved_curve=mean_cache_hit if noiseless else None,
            noiseless_fit_calls=(noiseless.get('cost') or {}).get('fit_calls',0) if not mean_cache_hit else 0,
            noiseless_fit_AC_calls=(noiseless.get('cost') or {}).get('ac_calls',0) if not mean_cache_hit else 0,**ledger)
        for key in ('r','x'):
            field=key.upper()+'_relative'
            if r[field] is None:r[field]=_norm_error(fit_map.get(true_name),trial['truth'],key)
        output.append(r)
        for stage,prediction in (('selected',pred),('raw',record.get('raw_prediction'))):
            for pair in (prediction or {}).get('pairs') or []:
                added=number(pair.get('added'))
                task_rows.append(dict(**common,method=name,stage=stage,a=pair.get('a'),b=pair.get('b'),
                    name=pair.get('name'),ensemble_id=pair.get('ensemble_id'),weight=number(pair.get('weight')),
                    success=bool(pair.get('success')),added=added,joint=number(pair.get('joint')),
                    self_relative=number(pair.get('self_relative')),numerical_unresolved=added<=resolution if added is not None else None))
        nfit_map={f['name']:f for f in nf};details={v['candidate']:v for v in noiseless.get('candidate_distances') or []}
        for candidate in names:
            for stage,fit in (('noisy',fit_map.get(candidate)),('noiseless',nfit_map.get(candidate))):
                detail=details.get(candidate) or {};channels=detail.get('future_channel_SSE') or {}
                candidate_rows.append(dict(**common,method=name,stage=stage,candidate=candidate,
                    is_true=candidate==true_name,success=bool(fit and fit.get('success')),
                    joint_SSE=number((fit or {}).get('loss')),r=(fit or {}).get('r'),x=(fit or {}).get('x'),
                    noiseless_added_half_SSE=number(detail.get('added_half_SSE')) if stage=='noiseless' else None,
                    historical_SSE=number(detail.get('historical_SSE')) if stage=='noiseless' else None,
                    future_V_SSE=number(channels.get('v')) if stage=='noiseless' else None,
                    future_P_SSE=number(channels.get('p')) if stage=='noiseless' else None,
                    future_Q_SSE=number(channels.get('q')) if stage=='noiseless' else None,
                    future_master_P_SSE=number(channels.get('master_p')) if stage=='noiseless' else None,
                    future_master_Q_SSE=number(channels.get('master_q')) if stage=='noiseless' else None,
                    missing_reason=None if fit else 'required_candidate_fit_missing:'+row.get('status','missing_method')))
    byname={r['method']:r for r in output}
    for row in output:row.update(trace_source_lineage(row['method'],byname))
    paired=[]
    for contrast,left,right in CONTRASTS:
        if left not in byname or right not in byname:continue
        a,b=byname[left],byname[right];item=dict(**common,contrast=contrast,left=left,right=right)
        for field in PAIRED_FIELDS:
            va,vb=number(a.get(field)),number(b.get(field))
            # Booleans are counts only when explicitly requested.
            if field=='selected_is_new_MILP_proposal':
                va=None if a.get(field) is None else int(a[field]);vb=None if b.get(field) is None else int(b[field])
            item['delta_'+field]=None if va is None or vb is None else va-vb
        item['complete_actual_pair']=a['actual_M'] is not None and b['actual_M'] is not None
        item['same_initial_action_set']=(a['initial_actions_sha256']==b['initial_actions_sha256'])
        item['same_configured_exchange_budget']=(a['configured_exchange_budget']==b['configured_exchange_budget'])
        paired.append(item)
    return dict(identity=common,declared_methods=methods,method_rows=output,paired_rows=paired,candidate_rows=candidate_rows,
        task_rows=task_rows,master_rows=master_rows,selection_failures=selection.get('failures') or {})


def aggregate_method(rows,*,planned_histories=None):
    complete=[r for r in rows if r['actual_M'] is not None]
    fields=(*PAIRED_FIELDS,'true_candidate_joint_SSE','minimum_joint_SSE','predicted_unresolved',
        'master_seconds','max_master_variables','max_master_binary','main_MILP_calls','QP_calls')
    return dict(history_denominator=len(rows),planned_histories=planned_histories,
        actual_complete=len(complete),actual_failed_or_missing=len(rows)-len(complete),
        status_counts=dict(Counter(r['status'] for r in rows)),
        true_candidate_min_SSE_count=sum(r['correct_min_loss'] is True for r in rows),
        candidate_miss_count=sum(r['candidate_miss'] for r in rows),
        noiseless_complete_count=sum(r['noiseless_M'] is not None for r in rows),
        selected_origin_counts=dict(Counter(str(r['selected_origin']) for r in rows)),
        inheritance_checked=sum(r['inheritance_check'] is not None for r in rows),
        inheritance_failed=sum(r['inheritance_check'] is False for r in rows),
        new_MILP_selected_count=sum(r['selected_is_new_MILP_proposal'] is True for r in rows),
        from_MILP_count=sum(r.get('source_lineage_from_MILP') is True for r in rows),
        ancestral_MILP_count=sum(r.get('source_lineage_ancestral_MILP') is True for r in rows),
        ancestral_new_MILP_count=sum(r.get('source_lineage_ancestral_new_MILP') is True for r in rows),
        source_lineage_error_counts=dict(Counter(r['source_lineage_error'] for r in rows if r.get('source_lineage_error'))),
        proxy_status_counts=dict(Counter(str(r['proxy_status']) for r in rows)),
        proxy_completed=sum(r['proxy_completed'] is True for r in rows),
        proxy_unfinished=sum(r['proxy_unfinished'] is True for r in rows),
        random_used_by_primary_count=sum(r['random_used_by_primary'] for r in rows if r['method'] in ALL_DESIGNS),
        total_main_MILP_calls=sum(r['main_MILP_calls'] for r in rows),
        total_new_MILP_proposals=sum(r['new_MILP_proposals'] for r in rows),
        metrics={field:stats(r.get(field) for r in rows) for field in fields})


def summarize_directory(directory,*,label=None,resolution=.01):
    directory=Path(directory);freeze=core.read_json(directory/'freeze.json')
    label=directory.name if label is None else label
    results=[];files=[];seen=set()
    for path in sorted(directory.glob('trial_*.json.gz')):
        trial=core.read_json(path);seed=trial.get('seed')
        if seed in seen or seed not in freeze['seeds']:raise ValueError('duplicate or unplanned historical sample')
        seen.add(seed)
        if trial.get('phase')!=freeze.get('phase'):raise ValueError('development/evaluation/calibration phases cannot be mixed')
        if trial.get('source_sha256')!=freeze.get('source_sha256') or trial.get('config')!=freeze.get('config'):
            raise ValueError('trial source/config differs from freeze')
        if freeze.get('policy_id') is not None and trial.get('policy_id')!=freeze.get('policy_id'):
            raise ValueError('trial policy identity differs from freeze')
        results.append(analyze_trial(trial,resolution=resolution,label=label))
        files.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed))
    tables={key:[row for result in results for row in result[key]] for key in
        ('method_rows','paired_rows','candidate_rows','task_rows','master_rows')}
    declared=declared_methods(freeze) if not results else results[0]['declared_methods']
    if any(result['declared_methods']!=declared for result in results):raise ValueError('different planned method sets within one directory')
    grouped=defaultdict(list)
    for row in tables['method_rows']:grouped[row['method']].append(row)
    contrast_groups=defaultdict(list)
    for row in tables['paired_rows']:contrast_groups[row['contrast']].append(row)
    summary=dict(label=label,directory=str(directory.resolve()),phase=freeze.get('phase'),
        prediction=freeze.get('prediction'),master_form=freeze.get('master_form'),
        policy_id=freeze.get('policy_id'),source_sha256=freeze.get('source_sha256'),
        planned_histories=len(freeze['seeds']),available_histories=len(results),
        declared_methods=list(declared),
        missing_seeds=sorted(set(freeze['seeds'])-seen),seeds=sorted(seen),
        independent_unit='one historical sample; directed tasks and method outcomes are paired measurements',
        stratification='report descriptive paired mean/median/range/counts; no iid topology-risk claim',
        formal_risk_calibration=False,numerical_unresolved_resolution=resolution,
        methods={name:aggregate_method(grouped[name],planned_histories=len(freeze['seeds'])) for name in declared},
        contrasts={name:dict(history_denominator=len(rows),complete_actual_pairs=sum(r['complete_actual_pair'] for r in rows),
            same_initial_action_set_count=sum(r['same_initial_action_set'] for r in rows),
            same_configured_exchange_budget_count=sum(r['same_configured_exchange_budget'] for r in rows),
            metrics={field:stats(r.get('delta_'+field) for r in rows) for field in PAIRED_FIELDS})
            for name,rows in contrast_groups.items()},trial_hashes=files,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return dict(summary=summary,identities=[r['identity'] for r in results],**tables)


def compare_directories(analyses):
    """Same-seed ablations, never pooling different phases or data realizations."""
    rows=[];excluded=[]
    for i,left in enumerate(analyses):
        for right in analyses[i+1:]:
            la,lb=left['summary']['label'],right['summary']['label']
            li={r['seed']:r for r in left['identities']};ri={r['seed']:r for r in right['identities']}
            lm={(r['seed'],r['method']):r for r in left['method_rows']}
            rm={(r['seed'],r['method']):r for r in right['method_rows']}
            for seed in sorted(set(li)|set(ri)):
                if seed not in li or seed not in ri:
                    excluded.append(dict(left_label=la,right_label=lb,seed=seed,reason='history_missing_in_one_directory'));continue
                a,b=li[seed],ri[seed]
                checks={key:a.get(key)==b.get(key) for key in ('phase','domain','history_sha256','truth_sha256',
                    'noise_sha256','measurement_sha256','device_sha256','data_sha256')}
                if not all(checks.values()):
                    excluded.append(dict(left_label=la,right_label=lb,seed=seed,reason='phase_or_data_identity_mismatch',checks=checks));continue
                for method in left['summary'].get('declared_methods',METHODS):
                    if (seed,method) not in rm:continue
                    x,y=lm[(seed,method)],rm[(seed,method)]
                    row=dict(left_label=la,right_label=lb,seed=seed,phase=a['phase'],family=a['family'],method=method,
                        data_sha256=a['data_sha256'],left_status=x['status'],right_status=y['status'],
                        same_scientific_source=left['summary']['source_sha256']==right['summary']['source_sha256'])
                    for field in PAIRED_FIELDS:
                        xv,yv=number(x.get(field)),number(y.get(field))
                        row['right_minus_left_'+field]=None if xv is None or yv is None else yv-xv
                    rows.append(row)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['left_label'],row['right_label'],row['phase'],row['method'])].append(row)
    summary=[]
    for (left,right,phase,method),items in grouped.items():
        summary.append(dict(left_label=left,right_label=right,phase=phase,method=method,
            matched_history_denominator=len(items),complete_actual_pairs=sum(r['right_minus_left_actual_M'] is not None for r in items),
            left_status_counts=dict(Counter(r['left_status'] for r in items)),
            right_status_counts=dict(Counter(r['right_status'] for r in items)),
            metrics={field:stats(r['right_minus_left_'+field] for r in items) for field in PAIRED_FIELDS}))
    return dict(rows=rows,excluded=excluded,summary=summary,
        direction='right directory minus left directory; one row per shared historical sample and method',
        pooled_history_count=None)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value for key,value in row.items()})


def markdown(analysis):
    summary=analysis['summary'];lines=[f"# {summary['label']} 逐历史汇总",'',
        f"阶段：{summary['phase']}；已保存 {summary['available_histories']}/{summary['planned_histories']} 个独立历史。缺失历史：{summary['missing_seeds']}。",'',
        'M 为最小错误候选 SSE 减真实候选 SSE；正值表示真实候选取得较小误差。完整 AC 拟合为局部取得值。任务方向和预测情景不增加独立历史数。','',
        '| 方法 | 完整/历史分母 | M均值 | 无噪M均值 | 预测均值 | 最终来源 | 本层新MILP | 来自MILP（含继承） | 继承MILP | 代理完成 |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.4g}'
    for name,record in summary['methods'].items():
        metrics=record['metrics'];origins=', '.join(f'{k}:{v}' for k,v in record['selected_origin_counts'].items())
        lines.append(f"| {name} | {record['actual_complete']}/{record['history_denominator']} | {fmt(metrics['actual_M']['mean'])} | {fmt(metrics['noiseless_M']['mean'])} | {fmt(metrics['predicted_mean']['mean'])} | {origins} | {record['new_MILP_selected_count']} | {record['from_MILP_count']} | {record['ancestral_MILP_count']} | {record['proxy_completed']} |")
    lines+=['','未完成、优化失败、候选缺失和未返回未来拟合均保留方法分母；对应数值留空。预测分数 ≤ 数值阈值只标为数值未决。',
        'MILP提案数、最终来源与原域QP间隙分开报告；来源写作MILP仍需检查最终曲线是否确为新增提案。随机价格仅作对照。',
        '来源链沿父域方案追溯；继承父域MILP与本层新发现的MILP曲线分开计数。质量价格优化单独标为quality来源。',
        '跨目录仅对相同阶段、真实参数、历史、未来背景/噪声、计量和设备哈希一致的历史做配对，开发与评价各自汇总。','']
    if 'refined_vs_cap' in summary['contrasts']:
        audit=summary['contrasts']['refined_vs_cap'];n=audit['history_denominator']
        lines+= [f"refined 与 cap 公平性核对：相同初始曲线集合 {audit['same_initial_action_set_count']}/{n}；相同配置预算 {audit['same_configured_exchange_budget_count']}/{n}。",'']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',action='append',required=True,help='label=trial_directory')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resolution',type=float,default=.01)
    args=parser.parse_args();analyses=[];labels=set();args.output.mkdir(parents=True,exist_ok=True)
    if not 0<=args.resolution<1e100:raise ValueError('finite nonnegative resolution required')
    for item in args.input:
        label,separator,path=item.partition('=')
        if not separator:label=Path(item).name;path=item
        if not label or Path(label).name!=label or label in labels:raise ValueError('unique plain labels required')
        labels.add(label);analysis=summarize_directory(path,label=label,resolution=args.resolution);analyses.append(analysis)
        folder=args.output/label;folder.mkdir(parents=True,exist_ok=True)
        for key in ('method_rows','paired_rows','candidate_rows','task_rows','master_rows'):write_csv(folder/(key+'.csv'),analysis[key])
        core.write_json(folder/'summary.json',analysis['summary'])
        (folder/'summary.md').write_text(markdown(analysis),encoding='utf-8')
    comparison=compare_directories(analyses);write_csv(args.output/'matched_directory_contrasts.csv',comparison['rows'])
    core.write_json(args.output/'directory_match_audit.json',{key:value for key,value in comparison.items() if key!='rows'})
    print(json.dumps({a['summary']['label']:dict(available=a['summary']['available_histories'],planned=a['summary']['planned_histories']) for a in analyses}))


if __name__=='__main__':main()
