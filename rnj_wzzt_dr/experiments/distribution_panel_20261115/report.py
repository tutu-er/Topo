"""Complete-denominator, paired, date-cluster summaries of an audited panel.

All path repetitions and four networks move together in a date bootstrap.
Data-valid identification/preparation/execution failures count as unrecovered;
missing raw-data windows remain in the qualification ledger, not invented scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from .audit import POLICIES, MODES, base, identity, entries_from


def checked(item):
    actual=base.ref(item['path'])
    if actual['sha256'] != item['sha256']: raise ValueError('Changed audit input: '+item['path'])
    return Path(item['path'])


def lines(item):
    with checked(item).open(encoding='utf8') as stream:
        return [json.loads(row) for row in stream if row.strip()]


def item_key(row): return row['id'],int(row['forecast']),int(row['repeat']),row['strategy']


def forecast_key(row): return row['id'],int(row['forecast'])


def interruption_panel(rows, quotes):
    """An interrupted forecast affects the comparison of every policy, together."""
    known={forecast_key(r) for r in rows}; affected=defaultdict(list)
    for quote in quotes:
        recovery=quote.get('recovery')
        if not recovery: continue
        if not isinstance(recovery,dict): raise ValueError('Recovery metadata must be a dictionary')
        key=forecast_key(quote)
        if key not in known: raise ValueError('Recovered quote outside the execution panel: '+str(key))
        affected[key].append(quote['policy'])
    marked=[dict(r,interruption_affected=forecast_key(r) in affected) for r in rows]
    forecasts=[dict(id=key[0],forecast=key[1],interruption_affected=key in affected,
        recovered_modes=sorted(set(affected.get(key,[]))),strategies=list(POLICIES)) for key in sorted(known)]
    return marked,forecasts


def historical_recovery_summary(quotes):
    """Lost-point prices are historical evidence, never current execution costs."""
    rows=[]
    for quote in quotes:
        recovery=quote.get('recovery')
        if not recovery: continue
        original=base.read(checked(recovery['original_quote'])) if recovery.get('original_quote') else {}
        rows.append(dict(id=quote['id'],forecast=int(quote['forecast']),policy=quote['policy'],
            kind=recovery.get('kind'),original_action_unavailable=recovery.get('original_action_unavailable'),
            original_quote=recovery.get('original_quote'),interrupted_trace=recovery.get('interrupted_trace'),
            historical_reported_cost_yuan=original.get('cost_yuan'),
            historical_reported_lower_bound_yuan=original.get('lower_bound_yuan'),
            historical_quote_wall_seconds=original.get('quote_wall_seconds'),
            prior_measured_wall_lower_bound=recovery.get('prior_measured_wall_lower_bound'),
            prior_budget_charge_seconds=recovery.get('prior_budget_charge_seconds'),
            remaining_native_budget_seconds=recovery.get('remaining_native_budget_seconds'),
            recovery_verification_seconds=recovery.get('recovery_verification_seconds'),
            additional_quote_wall_seconds=recovery.get('additional_quote_wall_seconds'),
            current_point_source=quote.get('source'),current_point_cost_yuan=quote.get('cost_yuan'),
            current_point_lower_bound_yuan=quote.get('lower_bound_yuan'),
            current_point_relative_gap=quote.get('relative_gap'),
            current_executed_activation_cost_yuan=quote.get('executed_cost',{}).get('activation_yuan'),
            timing_scope=recovery.get('timing_scope'),bound_sources=quote.get('bound_sources'),recovery=recovery))
    fields=('historical_reported_cost_yuan','historical_reported_lower_bound_yuan','historical_quote_wall_seconds',
        'prior_measured_wall_lower_bound','prior_budget_charge_seconds','remaining_native_budget_seconds',
        'recovery_verification_seconds','additional_quote_wall_seconds')
    return dict(n=len(rows),kind_counts=dict(Counter(str(r['kind']) for r in rows)),rows=rows,
        distributions={field:distribution([r[field] for r in rows]) for field in fields},
        scope='Historical lost-point prices and bounds are not current feasible upper bounds. Current cost, bound and gap use the audited returned point; executed cost separately includes any fallback.',
        timing_scope='Prior budget charge is conservative accounting. Measured lower bounds, extra solve time and verification time remain separate; they do not reconstruct an exact interrupted wall time.')


def complete_panel(records, ledger, preparation_failures, protocol):
    """Construct the predeclared data-valid denominator, with no imputation of truth."""
    if len({item_key(r) for r in records}) != len(records): raise ValueError('Duplicate audited execution identities')
    lookup={item_key(r):r for r in records}; rows=[]; qualification=[]
    prepare={r['id']:r for r in preparation_failures}
    for attempt in ledger:
        task=attempt.get('task',attempt)
        tag=identity(task); status=attempt['status']
        data_valid=status not in ('data_incomplete','data_invalid')
        generated=status=='complete'; prepared=generated and tag['id'] not in prepare
        reason='data_incomplete' if not data_valid else ('candidate_or_background_failed' if not generated else
            'resource_or_forecast_preparation_failed' if not prepared else None)
        qualification.append(dict(**tag,data_valid=data_valid,history_prepared=generated,
            resources_and_forecast_prepared=prepared,source_status=status,source_stage=attempt.get('stage'),
            preparation_failure=prepare.get(tag['id']),failure_category=reason))
        if not data_valid: continue
        for forecast in range(protocol['forecasts']):
            for repeat in range(protocol['repeats']):
                for policy in POLICIES:
                    key=(tag['id'],forecast,repeat,policy)
                    if prepared:
                        if key not in lookup: raise ValueError('Missing audited data-valid path '+str(key))
                        row=lookup.pop(key)
                        if row['audit_status'] not in ('passed','recorded_failure'): raise ValueError('Unverified result cannot enter performance report')
                    else:
                        if key in lookup: raise ValueError('Failed preparation unexpectedly has an audited result')
                        row=dict(**tag,forecast=forecast,repeat=repeat,strategy=policy,
                            original_status=reason,audit_status='recorded_failure',
                            evaluation={'native_correct':False})
                    rows.append(row)
    if lookup: raise ValueError('Audited rows outside predeclared ledger: '+str(list(lookup)[:3]))
    return rows,qualification


def recovered(row):
    return bool(row['audit_status']=='passed' and row.get('evaluation',{}).get('native_correct',False))


def mean_or_none(values):
    values=[float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def distribution(values):
    values=[float(v) for v in values if v is not None]
    return dict(n=len(values),minimum=min(values) if values else None,median=float(np.median(values)) if values else None,
        maximum=max(values) if values else None,mean=mean_or_none(values))


def solver_summary(quotes, mode):
    selected=[q for q in quotes if q['policy']==mode]
    feasible=[q for q in selected if q['new_gate_feasible']]
    zero=[q for q in feasible if q.get('zero_action_sufficient',False) and q.get('cost_yuan')==0.]
    nonzero=[q for q in feasible if q.get('nonzero_executed',False)]
    native_nonzero=[q for q in nonzero if q.get('optimizer_invoked')]
    def stats(group):
        return dict(n=len(group),gap=distribution([q['relative_gap'] for q in group]),
            activation_upper_bound_yuan=distribution([q['cost_yuan'] for q in group]),
            activation_lower_bound_yuan=distribution([q['lower_bound_yuan'] for q in group]),
            charged_seconds=distribution([q['charged_total_seconds'] for q in group]),
            rows=[{k:q.get(k) for k in ('id','forecast','source','native_status','optimizer_invoked',
                'initial','cost_yuan','lower_bound_yuan','relative_gap','quote_wall_seconds','charged_total_seconds',
                'recovery','bound_sources')} for q in group])
    return dict(n=len(selected),new_gate_feasible=len(feasible),
        zero_action_sufficient=sum(q.get('zero_action_sufficient',False) for q in selected),
        nonzero_executed=sum(q.get('nonzero_executed',False) for q in selected),
        anchor_fallbacks=sum(q['anchor_fallback'] for q in selected),zero_fallbacks=sum(q['zero_fallback'] for q in selected),
        native_status_counts=dict(Counter(str(q['native_status']) for q in selected)),
        status_counts=dict(Counter(str(q['status']) for q in selected)),
        gap_mean_feasible=mean_or_none([q['relative_gap'] for q in feasible]),
        gap_mean_feasible_scope='Includes zero-cost certificates; do not use this mixture to assess nonzero QCQP convergence.',
        zero_cost_certificates=dict(n=len(zero),scope='Verified zero action and nonnegative activation prices give exact objective zero; no nonconvex search optimality is inferred.'),
        nonzero_feasible_quotes=stats(nonzero),nonzero_feasible_optimizer_invoked=stats(native_nonzero),
        recovered_after_interruption=sum(bool(q.get('recovery')) for q in selected),
        cost_scope='Current audited returned-point costs and bounds only; lost historical incumbents are listed separately.',
        charged_seconds_mean=mean_or_none([q['charged_total_seconds'] for q in selected]))


def summarize(rows):
    completed=[r for r in rows if r['audit_status']=='passed'];n=len(rows)
    metrics=dict(n=n,completed=len(completed),recorded_failures=n-len(completed),
        forecasts=len({forecast_key(r) for r in rows}),
        interruption_affected_forecasts=len({forecast_key(r) for r in rows if r.get('interruption_affected',False)}),
        interruption_affected_rows=sum(bool(r.get('interruption_affected',False)) for r in rows),
        recovered=sum(recovered(r) for r in rows),
        full_support_recovered=sum(bool(r.get('evaluation',{}).get('native_full_support_exact')) for r in completed),
        wrong_high=sum(bool(r['evaluation']['wrong_high']) for r in completed),
        true_high=sum(bool(r['evaluation']['true_high']) for r in completed),
        voltage_screen_failed=sum(not r['voltage_screen_passed'] for r in completed),
        correct_and_voltage_pass=sum(bool(r['recovered_and_voltage_screen_passed']) for r in completed),
        status_counts=dict(Counter(r['original_status'] for r in rows)),
        recovery_rate=None if not n else sum(recovered(r) for r in rows)/n,
        thermal_rating_status='unavailable_not_verified')
    for field in ('planned_cost','settled_cost'):
        metrics[field+'_mean_completed']={key:mean_or_none([r[field][key] for r in completed]) for key in (
            'total_yuan','activation_yuan','availability_yuan','active_activation_yuan','reactive_activation_yuan',
            'reserved_P_kw','reserved_Q_kvar')}
    metrics['cost_denominator']=len(completed)
    metrics['failure_cost_status']='Unavailable failure costs are not imputed as zero.'
    for which in ('requested','delivered'):
        for unit,suffix in (('kW','P_kwh'),('kvar','Q_kvarh')):
            metrics[which+'_'+suffix+'_mean_completed']=mean_or_none([
                sum(d['absolute_energy'] for d in r[which] if d['unit']==unit) for r in completed])
    metrics['electrical_mean_completed']={field:mean_or_none([r['electrical'][field] for r in completed]) for field in (
        'active_loss_kwh','reactive_loss_kvarh','root_signed_kwh','root_import_kwh','root_export_kwh','maximum_branch_current_A')}
    metrics['voltage_range_all_buses']=None if not completed else [
        min(r['electrical']['all_bus_voltage']['minimum'] for r in completed),
        max(r['electrical']['all_bus_voltage']['maximum'] for r in completed)]
    diagnostics=[r['evaluation']['historical_mean_diagnostic'] for r in completed if 'historical_mean_diagnostic' in r['evaluation']]
    metrics['mean_model_error']=dict(n=len(diagnostics),**{field:mean_or_none([d[field] for d in diagnostics]) for field in (
        'K_nominal_forecast_mu','K_actual_input_mu','actual_power_minus_nominal_forecast','clean_minus_actual_K',
        'observed_minus_clean','clean_minus_nominal_K')})
    metrics['squared_voltage_mean_error']=dict(n=len(diagnostics),**{field:mean_or_none([d.get(field) for d in diagnostics]) for field in (
        'clean_Y_minus_K_mean','clean_Y_minus_K_rmse','clean_Y_minus_frozen_H_mean','clean_Y_minus_frozen_H_rmse')})
    absolute=[r['evaluation']['evidence'] for r in completed if 'evidence' in r['evaluation']]
    metrics['directed_true_family_absolute']=dict(n=len(absolute),**{field:mean_or_none([d[field] for d in absolute])
        for field in ('h','J','bias','meter','clean_logodds','observed_logodds')})
    metrics['directed_pair_scope']=('Offline only: the historically best frozen point with correct internal structure, '
        'against the wrong historical winner or otherwise the historically best wrong structure. '
        'The pair remains fixed across strategies and repeats; missing true candidates yield no such diagnostic.')
    increments=[r['evaluation']['evidence_increment'] for r in completed if 'evidence_increment' in r['evaluation']]
    metrics['directed_true_family_increment']=dict(n=len(increments),**{field:mean_or_none([d[field] for d in increments])
        for field in ('J','bias','meter','clean_logodds','observed_logodds')})
    metrics['increment_ratio_status']='No realization ratio is reported; zero predicted gain gives an undefined 0/0 ratio.'
    return metrics


def paired_comparison(rows, policy, baseline):
    selected={item_key(r)[:3]:r for r in rows if r['strategy']==policy}
    reference={item_key(r)[:3]:r for r in rows if r['strategy']==baseline}
    if set(selected)!=set(reference): raise ValueError('Unequal paired denominators')
    gains=losses=0;by_input=defaultdict(list);labels={}
    for key in sorted(selected):
        a,b=selected[key],reference[key]; va,vb=recovered(a),recovered(b)
        gains+=int(va and not vb);losses+=int(vb and not va)
        by_input[key[0]].append(int(va)-int(vb));labels[key[0]]=a
    by_date=defaultdict(list)
    for input_id,values in by_input.items():
        a=labels[input_id];by_date[a['date_cluster']].append(dict(id=input_id,case=a['case'],season=a['season'],difference=float(np.mean(values)),n=len(values)))
    dates=[dict(date_cluster=day,season=group[0]['season'],difference=float(np.mean([r['difference'] for r in group])),
        networks=len(group),inputs=group) for day,group in sorted(by_date.items())]
    return dict(policy=policy,baseline=baseline,n=len(selected),gained=gains,lost=losses,
        pooled_path_difference=None if not selected else (gains-losses)/len(selected),
        date_equal_weight_difference=mean_or_none([d['difference'] for d in dates]),
        dates=dates,complete_four_network_date_difference=mean_or_none([d['difference'] for d in dates if d['networks']==4]))


def cluster_interval(dates, *, seed=2026111503, draws=5000):
    """Stratified resampling of entire dates; never resample individual paths."""
    groups=defaultdict(list)
    for row in dates: groups[row['season']].append(float(row['difference']))
    if not dates: return dict(interval=None,reason='No data-valid dates',date_clusters=0)
    if min(map(len,groups.values())) < 2:
        return dict(interval=None,reason='At least one represented season has fewer than two dates; a stratified interval would be degenerate.',
            date_clusters=len(dates),season_counts={k:len(v) for k,v in groups.items()})
    rng=np.random.default_rng(seed); means=np.zeros(draws)
    for season in sorted(groups):
        values=np.asarray(groups[season]);means+=values[rng.integers(len(values),size=(draws,len(values)))].sum(1)
    means/=len(dates)
    return dict(interval=np.quantile(means,[.025,.975]).tolist(),seed=seed,draws=draws,date_clusters=len(dates),
        season_counts={k:len(v) for k,v in groups.items()},method='Equal-date mean; seasons fixed; whole dates sampled with replacement inside season.')


def method_comparisons(rows, *, primary=False):
    comparisons={}
    for policy in MODES:
        comparisons[policy]={}
        for baseline in ('no_dfs','anchor',policy+'_rotation'):
            comparison=paired_comparison(rows,policy,baseline)
            comparison['date_cluster_95_interval']=cluster_interval(comparison['dates'])
            comparison['inferential_role']=('predeclared_primary' if primary and (policy,baseline)==('mgf_fixed','no_dfs')
                else 'descriptive_comparison' if primary else 'administrative_interruption_sensitivity')
            comparisons[policy][baseline]=comparison
    return comparisons


def interruption_subset(rows, quotes, *, affected):
    selected=[r for r in rows if r['interruption_affected']==affected]
    keys={forecast_key(r) for r in selected}
    return dict(forecasts=len(keys),execution_rows=len(selected),
        groups={p:summarize([r for r in selected if r['strategy']==p]) for p in POLICIES},
        paired_vs_no_dfs={p:paired_comparison(selected,p,'no_dfs') for p in POLICIES},
        comparisons=method_comparisons(selected),
        solver={mode:solver_summary([q for q in quotes if forecast_key(q) in keys],mode) for mode in MODES},
        selection_scope='One common forecast subset for all eight policies, based only on interruption metadata; no outcome-dependent or method-specific exclusions.')


def run(audit_path, out):
    audit_path=Path(audit_path).resolve();out=Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    audit=base.read(audit_path)
    if audit['status']!='passed': raise ValueError('A passed independent audit is required; no performance summary on inconsistent records')
    protocol=base.read(checked(audit['source_protocol']))
    ledger=base.read(checked(protocol['qualification_ledger']))
    failures=base.read(checked(protocol['preparation_failures']))
    if protocol.get('smoke'):
        ids={e['id'] for e in entries_from(base.read(checked(protocol['inputs'])))}|{r['id'] for r in failures}
        ledger=[r for r in ledger if r.get('task',r)['id'] in ids]
    rows,qualification=complete_panel(lines(audit['records']),ledger,failures,protocol)
    quotes=lines(audit['quotes']);rows,forecast_flags=interruption_panel(rows,quotes)
    recovery_history=historical_recovery_summary(quotes)
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'report.py')
    flags_path=out/'execution_interruption_flags.jsonl'
    with flags_path.open('w',encoding='utf8') as stream:
        for row in rows:
            flag={key:row[key] for key in ('id','forecast','repeat','strategy','interruption_affected')}
            stream.write(json.dumps(flag,ensure_ascii=False,allow_nan=False)+'\n')
    comparisons=method_comparisons(rows,primary=True)
    coverage={r['id']:r for r in audit['input_coverage']}
    def split_group(field,flag): return [r for r in rows if r['id'] in coverage and bool(coverage[r['id']][field])==flag]
    groups={policy:summarize([r for r in rows if r['strategy']==policy]) for policy in POLICIES}
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),audit=base.ref(audit_path),
        source_protocol=audit['source_protocol'],phase=protocol.get('phase'),smoke=protocol.get('smoke',False),
        qualification=qualification,qualification_counts=dict(planned=len(qualification),data_valid=sum(r['data_valid'] for r in qualification),
            data_incomplete=sum(not r['data_valid'] for r in qualification),history_prepared=sum(r['history_prepared'] for r in qualification),
            resources_prepared=sum(r['resources_and_forecast_prepared'] for r in qualification)),
        input_coverage_and_offered_resources=list(coverage.values()),
        coverage_counts=dict(assessed=len(coverage),truth_in_native_bank=sum(r['truth_in_native_bank'] for r in coverage.values()),
            truth_in_target_set=sum(r['truth_in_target_set'] for r in coverage.values()),history_correct=sum(r['history_correct'] for r in coverage.values())),
        full_pipeline_execution_rows=len(rows),groups=groups,comparisons=comparisons,
        paired_vs_no_dfs={p:paired_comparison(rows,p,'no_dfs') for p in POLICIES},
        interruption=dict(forecasts=len(forecast_flags),affected_forecasts=sum(f['interruption_affected'] for f in forecast_flags),
            unaffected_forecasts=sum(not f['interruption_affected'] for f in forecast_flags),forecast_records=forecast_flags,
            execution_flags=base.ref(flags_path),historical_recovery=recovery_history,
            unaffected=interruption_subset(rows,quotes,affected=False),affected=interruption_subset(rows,quotes,affected=True),
            scope='Primary groups retain every predeclared data-valid forecast, including interrupted forecasts. The common uninterrupted subset is an additional sensitivity analysis, not a replacement primary result.'),
        by_date={day:{p:summarize([r for r in rows if r['date_cluster']==day and r['strategy']==p]) for p in POLICIES}
            for day in sorted({r['date_cluster'] for r in rows})},
        by_network={case:{p:summarize([r for r in rows if r['case']==case and r['strategy']==p]) for p in POLICIES}
            for case in sorted({r['case'] for r in rows})},
        coverage_strata={field:{str(flag):{p:summarize([r for r in split_group(field,flag) if r['strategy']==p]) for p in POLICIES}
            for flag in (False,True)} for field in ('history_correct','truth_in_native_bank','truth_in_target_set')},
        solver={mode:solver_summary(quotes,mode) for mode in MODES},
        failed_or_unresolved_quotes=[q for q in quotes if not q['new_gate_feasible'] or q['native_status']==9 or q['status']=='exception'],
        zero_gate_disagreements=[dict(id=input_id,forecast=forecast,
            sufficient={q['policy']:q.get('zero_action_sufficient') for q in quotes if (q['id'],q['forecast'])==(input_id,forecast)},
            executed_activation_cost={q['policy']:q['executed_cost']['activation_yuan'] for q in quotes if (q['id'],q['forecast'])==(input_id,forecast)})
            for input_id,forecast in sorted({(q['id'],q['forecast']) for q in quotes})
            if len({q.get('zero_action_sufficient') for q in quotes if (q['id'],q['forecast'])==(input_id,forecast)})>1],
        rotation_unmatched_fees=audit['rotation_unmatched_fees'],
        statistical_scope='Independent units are physical date clusters, with four networks and all forecast/execution repetitions kept together. No AC recovery claim follows from a declared-model gate.',
        failure_cost_scope='Data-valid pipeline failures are unrecovered in every strategy denominator; unavailable costs remain unknown.',
        thermal_rating_status='unavailable_not_verified')
    base.write(out/'report.json',result)
    text=['# 多日期独立核验汇总','',f"阶段：{result['phase']}；接线测试：{result['smoke']}。",'',
        f"计划输入 {len(qualification)}；数据有效 {result['qualification_counts']['data_valid']}；历史准备完成 {result['qualification_counts']['history_prepared']}；资源准备完成 {result['qualification_counts']['resources_prepared']}。",'',
        '|策略|完整内部恢复/全流程路径|新增/丢失（相对无采购）|错误高权重|费用均值，元（完整执行）|',
        '|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        g=groups[policy];c=paired_comparison(rows,policy,'no_dfs')
        cost=g['settled_cost_mean_completed']['total_yuan']
        text.append(f"|{policy}|{g['recovered']}/{g['n']}|{c['gained']}/{c['lost']}|{g['wrong_high']}|{'未取得' if cost is None else f'{cost:.6f}'}|")
    interruption=result['interruption']
    text+=['',f"主表保留全部 {interruption['forecasts']} 个数据有效预测，其中 {interruption['affected_forecasts']} 个受中断影响。任一方法发生中断恢复时，该预测的八种策略统一标记。",'',
        f"以下为共同的 {interruption['unaffected_forecasts']} 个未受中断预测子集，仅作中断敏感性比较：",'',
        '|策略|完整内部恢复/全流程路径|新增/丢失（相对无采购）|错误高权重|费用均值，元（完整执行）|',
        '|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        g=interruption['unaffected']['groups'][policy];c=interruption['unaffected']['paired_vs_no_dfs'][policy]
        cost=g['settled_cost_mean_completed']['total_yuan']
        text.append(f"|{policy}|{g['recovered']}/{g['n']}|{c['gained']}/{c['lost']}|{g['wrong_high']}|{'未取得' if cost is None else f'{cost:.6f}'}|")
    if recovery_history['n']:
        text+=['',f"{recovery_history['n']} 张报价含中断恢复记录。当前费用与上下界按恢复后实际返回点核验；动作丢失的旧报价费用仅列历史记录。旧预算计费、已知实测耗时下界、新增求解时间和恢复核验时间分别保存。"]
    text+=['','失败保留在恢复分母中；失败路径未知费用没有填零。按日期的同权差和季节分层区间见 report.json；原生求解超时、无解未决和回退分别保存。',
        '电气检查覆盖隐藏母线、根及所有终端；热额定缺失，因此热安全保持未核验。声明高斯模型的成功下界与真实AC恢复分开。','']
    (out/'REPORT.md').write_text('\n'.join(text),encoding='utf8')
    base.write(out/'complete.json',dict(status='complete',report=base.ref(out/'report.json'),text=base.ref(out/'REPORT.md')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();answer=run(args.audit,args.out)
    print(dict(qualification=answer['qualification_counts'],rows=answer['full_pipeline_execution_rows']))
