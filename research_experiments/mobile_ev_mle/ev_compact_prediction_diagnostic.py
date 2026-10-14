"""Offline forecast/aggregation failure diagnosis from saved trials only.

True topology selects diagnostic rows after actions were frozen. It is never
sent back to a selector. This module performs no fits or optimization and
does not modify a policy, a raw trial, or any held-out evaluation rule.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from . import ev_compact_analysis as a

METHODS=('charge_mobile','v2g_mobile','refined_v2g_mobile','capped_v2g_mobile')
number=a.number


def _prediction(trial,method,row):
    selection=trial.get('selection') or {};direct=(selection.get('predicted_AC') or {}).get(method)
    if direct is not None:return direct
    matches=[p for p in selection.get('all_prediction_records') or [] if a.same_curve(row.get('power_kw'),p.get('power_kw'))]
    return matches[0] if len(matches)==1 else None


def expand_tasks(prediction,contexts,*,center_only=False):
    """Split exact-dedup context weights using saved aliases, not ID spelling."""
    lookup={c['id']:c for c in contexts};pairs=(prediction or {}).get('pairs') or [];out=[]
    for i,pair in enumerate(pairs):
        context=lookup.get(pair.get('ensemble_id')) or {}
        aliases=context.get('alias_records') or [dict(axis=context.get('axis',0 if center_only else None),
            scenario_id=context.get('scenario_id','center' if center_only else None),weight=context.get('weight',1.))]
        total=sum(float(alias['weight']) for alias in aliases)
        if total<=0:raise ValueError('saved context alias weights must have positive total')
        weight=number(pair.get('weight',1/len(pairs)))
        if weight is None or weight<0:raise ValueError('finite nonnegative saved task weight required')
        for alias in aliases:
            out.append(dict(name=pair.get('name',str(i)),a=pair['a'],b=pair['b'],
                ensemble_id=pair.get('ensemble_id'),axis=alias.get('axis'),scenario=alias.get('scenario_id'),
                explanation_id=(context.get('explanation_ids') or {}).get(pair['a']),
                flat_center_fallback=pair['a'] in alias.get('missing_flat_fallback_candidates',[]),
                weight=weight*float(alias['weight'])/total,success=bool(pair.get('success')),
                added=number(pair.get('added')),self_relative=number(pair.get('self_relative')),
                joint=number(pair.get('joint'))))
    return out


def score_group(rows,*,cap=20.):
    weight=sum(r['weight'] for r in rows);good=bool(rows and weight>0 and all(r['success'] and r['added'] is not None for r in rows))
    own=good and all(r['self_relative'] is not None for r in rows)
    return dict(alias_rows=len(rows),unique_task_count=len({r['name'] for r in rows}),
        task_weight=weight,complete=good,failed_or_missing=sum(not r['success'] or r['added'] is None for r in rows),
        mean_added=sum(r['weight']*r['added'] for r in rows)/weight if good else None,
        mean_cap_added=sum(r['weight']*min(r['added'],cap) for r in rows)/weight if good else None,
        global_mean_contribution=sum(r['weight']*r['added'] for r in rows) if good else None,
        global_cap_contribution=sum(r['weight']*min(r['added'],cap) for r in rows) if good else None,
        mean_self_relative=sum(r['weight']*r['self_relative'] for r in rows)/weight if own else None,
        worst_task_added=min(r['added'] for r in rows) if good else None,
        worst_task_self_relative=min(r['self_relative'] for r in rows) if own else None,
        flat_center_fallback_alias_rows=sum(r['flat_center_fallback'] for r in rows))


def _minimum(values):
    finite=[v for v in values if number(v) is not None]
    return min(finite) if finite and len(finite)==len(values) else None


def _rmse(left,right,scale=1.):
    if left is None or right is None:return None
    try:
        x,y=np.asarray(left,float),np.asarray(right,float);s=np.asarray(scale,float)
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(s).all() or np.any(s<=0):return None
        return float(np.sqrt(np.mean(((x-y)/s)**2)))
    except (ValueError,TypeError,ZeroDivisionError):return None


def _raw_proxy(record,trial,executed_power):
    raw=record.get('raw') or {};ev=raw.get('evaluations') or [];index=raw.get('best_evaluation_index')
    best=ev[index] if isinstance(index,int) and 0<=index<len(ev) else {}
    checks=(record.get('proxy_AC_checks') or [[]])[-1];weights=raw.get('weights') or {}
    hcost={f['name']:f['loss'] for f in trial.get('historical_fits') or [] if f.get('success')}
    rawpred=record.get('raw_prediction') or {};competitor={p.get('name',p['a']+'__to__'+p['b']):p['b'] for p in rawpred.get('pairs') or []}
    exact=[];rms=[]
    for row in checks:
        check=row.get('check') or {};name=row.get('name');b=competitor.get(name)
        value=number(check.get('complete_AC_cost'));w=number(weights.get(name))
        if value is not None and b in hcost and w is not None:exact.append(w*.5*(value-hcost[b]))
        value=number(check.get('max_channel_RMS'))
        if value is not None:rms.append(value)
    complete=bool(checks and len(exact)==len(checks) and len(checks)==len(raw.get('required_pairs') or []))
    interval=best.get('mean_score_interval') or [None,None]
    pred=a.prediction_summary(rawpred)
    return dict(raw_same_curve_as_executed=a.same_curve(raw.get('u'),executed_power),
        raw_proxy_status=raw.get('status'),raw_proxy_gate=record.get('raw_proxy_gate'),
        raw_proxy_mean_lower=number(interval[0]),raw_proxy_mean_upper=number(interval[1]),
        raw_same_witness_AC_mean=sum(exact) if complete else None,
        raw_full_AC_refit_mean=pred['mean'],raw_max_channel_RMS=max(rms) if rms else None,
        raw_proxy_checks=len(checks),raw_required_tasks=len(raw.get('required_pairs') or []))


def analyze_trial(trial,*,label):
    base=a.analyze_trial(trial,label=label);rows={r['method']:r for r in trial.get('rows') or []}
    physical=trial.get('config',{}).get('physical',trial.get('config') or {});base_kw=physical.get('base_kw',20.)
    cap=(trial.get('config') or {}).get('task_cap',20.);context=(trial.get('selection') or {}).get('context') or {}
    contexts=context.get('context_records') or [];true=a.core.true_candidate(trial)
    common=dict(label=label,seed=trial['seed'],phase=trial.get('phase'),family=base['identity']['family'],true_reference=true,
        truth_used='offline diagnostics only; never supplied to selector',policy_id=trial.get('policy_id'))
    summaries=[];groups=[];forecasts=[]
    h=len(trial['history'].get('root_v') or trial['history'].get('p') or [])
    explanation={r['id']:r for bank in (context.get('history_explanations') or {}).values() for r in bank}
    method_map={r['method']:r for r in base['method_rows']}
    for method in METHODS:
        if method not in base['declared_methods']:continue
        actual=method_map[method];row=rows.get(method) or {};pred=_prediction(trial,method,row)
        tasks=expand_tasks(pred,contexts,center_only=(trial.get('selection') or {}).get('prediction')=='center')
        allscore=score_group(tasks,cap=cap);truetasks=[r for r in tasks if r['a']==true];true_score=score_group(truetasks,cap=cap)
        comp=[]
        for b in sorted({r['b'] for r in truetasks}):
            result=score_group([r for r in truetasks if r['b']==b],cap=cap);comp.append((b,result))
        winner=min(comp,key=lambda x:float('inf') if x[1]['mean_self_relative'] is None else x[1]['mean_self_relative']) if comp else (None,{})
        summary=dict(**common,method=method,status=actual['status'],selected_origin=actual['selected_origin'],
            actual_M=actual['actual_M'],actual_half_M=actual['actual_half_M'],noiseless_M=actual['noiseless_M'],
            noiseless_half_M=actual['noiseless_half_M'],prediction_success=bool(pred and pred.get('success')),
            global_mean=allscore['mean_added'],global_cap_mean=allscore['mean_cap_added'],global_worst=allscore['worst_task_added'],
            true_reference_mean=true_score['mean_added'],true_reference_cap_mean=true_score['mean_cap_added'],
            true_reference_mean_self_relative=true_score['mean_self_relative'],true_reference_task_weight=true_score['task_weight'],
            true_reference_min_comp_mean_added=_minimum([s['mean_added'] for b,s in comp]),
            true_reference_min_comp_mean_self_relative=_minimum([s['mean_self_relative'] for b,s in comp]),
            true_reference_hardest_competitor=winner[0],true_reference_worst_task=true_score['worst_task_added'],
            numerical_tasks=len((pred or {}).get('pairs') or []),failed_tasks=allscore['failed_or_missing'])
        record=((trial.get('selection') or {}).get('methods') or {}).get(method) or {}
        summary.update(_raw_proxy(record,trial,row.get('power_kw')))
        for axis in sorted({r['axis'] for r in truetasks if r['axis'] is not None}):
            pieces=[]
            for b in sorted({r['b'] for r in truetasks}):
                subset=[r for r in truetasks if r['axis']==axis and r['scenario']=='center' and r['b']==b]
                if subset:pieces.append(score_group(subset,cap=cap)['mean_self_relative'])
            summary[f'true_axis{axis}_center_min_comp_self_relative']=_minimum(pieces)
        summaries.append(summary)
        for dimensions in (('a',),('a','axis'),('a','axis','scenario'),('a','b'),('a','axis','scenario','b')):
            grouped=defaultdict(list)
            for task in tasks:grouped[tuple(task[k] for k in dimensions)].append(task)
            for values,subset in grouped.items():
                key=dict(zip(dimensions,values));groups.append(dict(**common,method=method,grouping='+'.join(dimensions),
                    **key,is_true_reference=key['a']==true,**score_group(subset,cap=cap)))
        # These comparisons use the saved actual noiseless future only here.
        target=(row.get('noiseless_AC_refit') or {}).get('future_mean_observations') or {}
        meter=(trial.get('measurement_models') or {}).get('fm') or {}
        for ref in (pred or {}).get('records') or []:
            if ref.get('a')!=true:continue
            bundle=next((c for c in contexts if c['id']==ref.get('ensemble_id')),{});obs=ref.get('predicted_observations') or {}
            f=dict(**common,method=method,ensemble_id=ref.get('ensemble_id'),axis=bundle.get('axis'),scenario=bundle.get('scenario_id'),
                explanation_id=ref.get('explanation_id'),prediction_record_success=ref.get('success'))
            for channel in ('p','q','v','master_p','master_q'):
                f['forecast_actual_'+channel+'_white_RMS']=_rmse(obs.get(channel),target.get(channel),meter.get(channel+'_sigma',1.))
            state=ref.get('forecast_state') or {}
            for channel in ('source_p','source_q'):
                truth=trial['truth'].get(channel)
                value=_rmse(state.get(channel),truth[h:] if truth is not None else None)
                f['forecast_actual_'+channel+'_RMS_kw_kvar']=None if value is None else base_kw*value
            explained=explanation.get(ref.get('explanation_id')) or {};fit=explained.get('fit') or {}
            f.update(reference_R_relative=a._norm_error(fit,trial['truth'],'r'),
                reference_X_relative=a._norm_error(fit,trial['truth'],'x'),
                reference_history_half_SSE_gap=number(explained.get('history_gap_half_sse')))
            forecasts.append(f)
    by_method={r['method']:r for r in summaries};contrasts=[]
    if 'charge_mobile' in by_method:
        baseline=by_method['charge_mobile']
        fields=('global_mean','global_cap_mean','global_worst','true_reference_mean','true_reference_mean_self_relative',
            'true_reference_min_comp_mean_added','true_reference_min_comp_mean_self_relative',
            'actual_M','noiseless_M','actual_half_M','noiseless_half_M',
            'true_axis0_center_min_comp_self_relative','true_axis1_center_min_comp_self_relative')
        for method,row in by_method.items():
            if method=='charge_mobile':continue
            c=dict(**common,baseline='charge_mobile',method=method)
            for field in fields:
                left,right=number(baseline.get(field)),number(row.get(field));c['delta_'+field]=None if left is None or right is None else right-left
            g,cost,mean=c['delta_global_mean'],c['delta_actual_M'],c['delta_noiseless_M']
            c['mean_actual_reversal']=g is not None and cost is not None and g>1e-8 and cost< -1e-8
            c['mean_noiseless_reversal']=g is not None and mean is not None and g>1e-8 and mean< -1e-8
            c['true_reference_mean_also_decreases']=number(c['delta_true_reference_mean']) is not None and c['delta_true_reference_mean']< -1e-8
            c['hard_competitor_mean_also_decreases']=number(c['delta_true_reference_min_comp_mean_self_relative']) is not None and c['delta_true_reference_min_comp_mean_self_relative']< -1e-8
            contrasts.append(c)
    return dict(method_rows=summaries,group_rows=groups,forecast_rows=forecasts,contrast_rows=contrasts)


def run(inputs,output):
    tables={key:[] for key in ('method_rows','group_rows','forecast_rows','contrast_rows')};manifest=[]
    for label,directory in inputs:
        directory=Path(directory);freeze=a.core.read_json(directory/'freeze.json');files=[];seeds=[]
        if freeze.get('phase')!='development':raise ValueError('this failure diagnostic is restricted to development data')
        for path in sorted(directory.glob('trial_*.json.gz')):
            trial=a.core.read_json(path)
            if (trial.get('seed') not in freeze['seeds'] or trial.get('source_sha256')!=freeze['source_sha256']
                    or trial.get('phase')!=freeze['phase'] or trial.get('config')!=freeze.get('config')
                    or trial.get('policy_id')!=freeze.get('policy_id') or trial['seed'] in seeds):
                raise ValueError('saved trial/freeze mismatch')
            result=analyze_trial(trial,label=label)
            for key in tables:tables[key].extend(result[key])
            seeds.append(trial['seed']);files.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest.append(dict(label=label,directory=str(directory.resolve()),phase=freeze['phase'],
            available_seeds=seeds,planned_seeds=freeze['seeds'],missing_seeds=sorted(set(freeze['seeds'])-set(seeds)),trial_hashes=files))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for key,rows in tables.items():a.write_csv(output/(key+'.csv'),rows)
    a.core.write_json(output/'manifest.json',dict(inputs=manifest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='offline development diagnosis; true-reference labels never enter selector; no new AC fit or optimization',
        phases_pooled=False))
    lines=['# 预测指标与实际排错差异：离线诊断','',
        '真拓扑仅用于动作冻结后的分组。各版本、各历史分别保留，不合并开发与评价，也不改变现有策略。',
        '模型任务权重表示设计任务价值。true-reference 是“离线只查看真实拓扑作为参考的预测行”，没有回传选择器。','',
        '|版本|历史|比较（减移动充电）|Δ预测均值|Δ真实参考均值|Δ最难竞争者自相对均值|Δ实际M|Δ无噪M|',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.4g}'
    for row in tables['contrast_rows']:
        lines.append('|'+ '|'.join([row['label'],str(row['seed']),row['method'],fmt(row['delta_global_mean']),
            fmt(row['delta_true_reference_mean']),fmt(row['delta_true_reference_min_comp_mean_self_relative']),
            fmt(row['delta_actual_M']),fmt(row['delta_noiseless_M'])])+'|')
    lines+=['','分解含义：',
        '- 全部参考的平均增加、真实参考均值下降：参考模型之间存在任务价值取舍。',
        '- 真实参考均值增加、最难竞争者下降：平均竞争者目标与排除最近竞争者的目标存在取舍。',
        '- 完整AC预测的真实参考最难竞争者增加、实际无噪M下降：需要检查历史参数解释、负荷/源/追踪预测与实际未来的偏差。',
        '- 同见证代理误差和完整AC重拟合分开记录；若最终执行的是继承曲线，原始MILP点的代理检查不归到该曲线。',
        '- 单个开发历史的数值差只能定位可能机制。噪声、局部拟合与预测误差仍可同时作用。','',
        'group_rows.csv 按保存的参考、axis、scenario 和竞争者分组；精确合并情景的 alias 按其保存权重拆分。',
        'forecast_rows.csv 比较保存的真实参考预测均值与实际无噪未来，另列源预测差和历史线路解释误差。','']
    (output/'diagnostic.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(input_summaries=manifest,rows={key:len(rows) for key,rows in tables.items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();inputs=[]
    for value in args.input:
        label,separator,path=value.partition('=')
        if not separator:label=Path(value).name;path=value
        inputs.append((label,path))
    print(json.dumps(run(inputs,args.output),ensure_ascii=False))


if __name__=='__main__':main()
