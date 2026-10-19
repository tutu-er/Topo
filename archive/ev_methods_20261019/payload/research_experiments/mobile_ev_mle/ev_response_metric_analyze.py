"""Read-only strict two-candidate calibration and full-pipeline comparison.

Only standard-library analysis and the existing read-only collector helpers
are imported. No fit, policy, simulator, or solver is executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import ev_dynamic_collect as collector
from .ev_gradient_flow_analyze import stats,timing_audit as old_timing_audit

FAMILIES=('star','clade_01')
METHODS=('none','constant','highlow','square','nearest','gradac','contrast_joint','contrast_box','contrast_load')
SCHEMA='ev_strict_binary_response_analysis_v1'
CAL_SCHEMA='ev_strict_binary_response_thresholds_v1'
RUNTIMES=('planning_actual_seconds','planning_standalone_seconds','dependency_seconds',
    'operational_future_fit_seconds','common_history_pipeline_seconds',
    'end_to_end_standalone_seconds','soft_budget_overrun_seconds')


def _rank(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        for k in range(start,end):out[order[k]]=(start+end-1)/2
        start=end
    return out


def spearman(left,right):
    """Average ranks; undefined constant rankings remain unavailable."""
    if len(left)!=len(right) or len(left)<2:return None
    if not all(collector._finite(v) for v in [*left,*right]):return None
    a,b=_rank(left),_rank(right);am,bm=statistics.fmean(a),statistics.fmean(b)
    cov=sum((x-am)*(y-bm) for x,y in zip(a,b))
    den=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return cov/den if den>0 else None


def _timing_audit(trial):
    view=deepcopy(trial)
    for row in view.get('selection',{}).get('timing',{}).values():
        row['context_seconds']=row.get('context_seconds',0.)+row.get('scenario_seconds',0.)
    result=old_timing_audit(view)
    result['scenario_covariance_cost_included']=True
    return result


def rank_actions(values):
    ranking=dict(unique_executed_curves=len(values),
        scope='within-history unique executed actions; each history supplies at most one correlation')
    for metric in ('full_AC_distance','joint_contrast','box_contrast','load_contrast'):
        for target in ('M','clean_future_M'):
            valid=[r for r in values if collector._finite(r.get(metric)) and collector._finite(r.get(target))]
            consistent=opposite=ties=0
            for i,a in enumerate(valid):
                for b in valid[:i]:
                    x,y=a[metric]-b[metric],a[target]-b[target]
                    if abs(x)<=1e-6 or abs(y)<=1e-6:ties+=1
                    elif x*y>0:consistent+=1
                    else:opposite+=1
            ranking[metric+'_vs_'+target]=dict(actions=len(valid),
                spearman=spearman([r[metric] for r in valid],[r[target] for r in valid]),
                concordant=consistent,discordant=opposite,ties=ties,
                pair_counts_are_descriptive_not_independent_samples=True)
    return ranking


def _detail(trial,parsed):
    actual={row['name']:row for row in trial.get('methods',[])}
    logs=trial.get('selection',{}).get('methods',{})
    diag=trial.get('planning_diagnostics',{})
    out=dict(seed=parsed['seed'],family=parsed['family'],status=parsed['status'],
        candidate_miss=parsed['candidate_miss'],candidate_names=parsed['candidate_names'],
        seconds=trial.get('seconds'),timing_audit=_timing_audit(trial),methods={},
        actual_solver_calls=trial.get('actual_solver_calls',{}),
        diagnostic_seconds=diag.get('seconds'),diagnostic_calls=diag.get('calls',{}),
        load_pressure=trial.get('selection',{}).get('load_pressure',{}))
    unique={}
    for name in METHODS:
        row=parsed['methods'][name];raw=actual.get(name,{});log=logs.get(name,{})
        diagnostic=diag.get('methods',{}).get(name,{})
        runtime=raw.get('runtime',{})
        for key in RUNTIMES:
            val=runtime.get(key)
            if val is not None and (not collector._finite(val) or val<0):
                raise ValueError('invalid saved runtime '+name+'/'+key)
        failures=list(log.get('failures',[]))
        if log.get('failure'):failures.append(log['failure'])
        models=log.get('model',{})
        out['methods'][name]=dict(status=row['status'],raw_status=row['raw_status'],M=row['M'],
            clean_future_M=raw.get('clean_future_M') if raw.get('clean_future_complete') is True else None,
            losses=row['losses'],failure_reasons=row['failure_reasons'],energy=row['energy'],runtime=runtime,
            planning_success=raw.get('planning_success'),planning_failures=failures,
            selected_origin=log.get('selected_origin'),selected_score=log.get('selected_score'),
            selected_is_new_MILP_proposal=log.get('selected_is_new_MILP_proposal',
                str(log.get('selected_origin','')).endswith('MILP')),
            planning_profile_counters=log.get('counters',{}),
            contrast_MILP={k:log.get('MILP',{}).get(k) for k in
                ('success','proxy_value','proxy_best_bound','proxy_mip_gap','added_binary_count','seconds')},
            direction_records=models.get('direction_records',[]),
            full_AC_distance=diagnostic.get('future_profile_distance'),
            joint_contrast=diagnostic.get('contrast_joint',{}).get('score'),
            box_contrast=diagnostic.get('contrast_box',{}).get('score'),
            load_contrast=diagnostic.get('contrast_load',{}).get('score'),
            source_failures=row['source_failures'],curve_id=raw.get('curve_id',diagnostic.get('curve_id')))
        compact=out['methods'][name]
        key=compact['curve_id']
        if key is not None and row['status']=='complete':unique.setdefault(key,compact)
    out['ranking']=rank_actions(list(unique.values()))
    strong={}
    for name in ('square','nearest','gradac','contrast_joint','contrast_box','contrast_load'):
        row=out['methods'][name]
        if row['curve_id'] is not None and row['status']=='complete':
            strong.setdefault(row['curve_id'],row)
    out['ranking_strong']=rank_actions(list(strong.values()))
    out['ranking_strong']['scope']='unique fixed-site V2G baseline/design curves; excludes none/constant/highlow'
    return out


def read_phase(directory,phase):
    directory=Path(directory);freeze=collector.read_json(directory/'freeze.json')
    identity=collector._identity(freeze);config=freeze['config']
    if config.get('declared_candidates')!=list(FAMILIES):raise ValueError('exact predeclared pair required')
    expected=collector.expected_methods(config)
    if set(expected)!=set(METHODS) or len(expected)!=len(METHODS):raise ValueError('nine frozen methods required')
    planned=[str(seed) for seed in config[phase+'_seeds']]
    if len(set(planned))!=len(planned) or not planned:raise ValueError('distinct planned seeds required')
    records={};details={};files=[]
    for path in sorted(directory.glob('trial*.json.gz')):
        trial=collector.read_json(path);seed=str(trial.get('seed'))
        if seed not in planned or seed in records:raise ValueError('duplicate or unplanned independent history')
        if collector._identity(trial)!=identity:raise ValueError('trial frozen policy differs')
        if trial.get('source_unchanged') is False:raise ValueError('trial source changed during execution')
        if trial.get('family_index') not in (0,1) or isinstance(trial.get('family_index'),bool):
            raise ValueError('only declared true families 0/1 belong to this experiment')
        if trial['family_index']!=planned.index(seed)%2:
            raise ValueError('true family differs from predeclared seed-position alternation')
        names=[r['name'] for r in trial.get('candidates',[])]
        if names!=list(FAMILIES):raise ValueError('raw failure records must retain the declared two-candidate bank')
        parsed=collector.make_record(trial,list(METHODS));records[seed]=parsed
        details[seed]=_detail(trial,parsed)
        files.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed))
    n=len(planned);pending=[seed for seed in planned if seed not in records]
    result=dict(phase=phase,directory=str(directory.resolve()),**identity,planned_seeds=planned,
        observed_seeds=sorted(records),pending_seeds=pending,planned_histories=n,observed_histories=len(records),
        pending_histories=len(pending),family_counts=dict(Counter(r['family'] for r in records.values())),
        candidate_omissions=sum(r['candidate_miss'] for r in records.values()),methods={},records=details,
        input_files=files,parsed_records=list(records.values()),
        actual_history_runtime_seconds=stats([r['seconds'] for r in details.values()],n),
        diagnostic_seconds=stats([r['diagnostic_seconds'] for r in details.values()],n),
        timing_audit=dict(checks=sum(r['timing_audit']['checks'] for r in details.values()),
            failures=[dict(seed=seed,**f) for seed,r in details.items() for f in r['timing_audit']['failures']]))
    for name in METHODS:
        rows=[r['methods'][name] for r in details.values()]
        result['methods'][name]=dict(planned=n,observed=len(rows),pending=len(pending),
            complete=sum(r['status']=='complete' for r in rows),failed=sum(r['status']!='complete' for r in rows),
            planning_unsuccessful=sum(r['planning_success'] is False for r in rows),
            M=stats([r['M'] for r in rows],n),clean_future_M=stats([r['clean_future_M'] for r in rows],n),
            runtime={key:stats([r['runtime'].get(key) for r in rows],n) for key in RUNTIMES},
            standalone_AC_batches=stats([r['runtime'].get('standalone_AC_batches') for r in rows],n),
            selected_origin=dict(Counter(str(r['selected_origin']) for r in rows)),
            failure_reasons=dict(Counter(reason for r in rows for reason in set(r['failure_reasons']))),
            planning_failure_reasons=dict(Counter(reason for r in rows for reason in set(map(str,r['planning_failures'])))),
            new_MILP_actions=sum(r['selected_is_new_MILP_proposal'] is True for r in rows),
            energy={key:stats([r['energy'].get(key) for r in rows],n) for key in
                ('grid_energy_kwh','charging_kwh','discharging_kwh','throughput_kwh','driving_kwh','worker_minutes','moves')})
    result['paired_contrasts']={}
    for left in ('nearest','gradac','contrast_joint','contrast_box','contrast_load'):
        for right in ('square','constant'):
            pairs=[]
            for seed in planned:
                d=details.get(seed);a,b=(d['methods'][name] for name in (left,right)) if d else ({},{})
                valid=a.get('status')==b.get('status')=='complete'
                pairs.append(dict(seed=seed,pending=d is None,
                    delta_M=a['M']-b['M'] if valid else None,
                    delta_clean_M=a['clean_future_M']-b['clean_future_M'] if valid and
                        collector._finite(a.get('clean_future_M')) and collector._finite(b.get('clean_future_M')) else None))
            result['paired_contrasts'][left+'_vs_'+right]=dict(left=left,right=right,rows=pairs,
                delta_M=stats([r['delta_M'] for r in pairs],n),
                delta_clean_M=stats([r['delta_clean_M'] for r in pairs],n))
    result['ranking']={metric+'_vs_'+target:stats([r['ranking'][metric+'_vs_'+target]['spearman']
        for r in details.values()],n) for metric in ('full_AC_distance','joint_contrast','box_contrast','load_contrast')
        for target in ('M','clean_future_M')}
    result['ranking_strong']={metric+'_vs_'+target:stats([r['ranking_strong'][metric+'_vs_'+target]['spearman']
        for r in details.values()],n) for metric in ('full_AC_distance','joint_contrast','box_contrast','load_contrast')
        for target in ('M','clean_future_M')}
    result['planned_method_slots']=n*len(METHODS)
    result['failed_method_slots']=sum(r['failed'] for r in result['methods'].values())
    result['complete_method_slots']=sum(r['complete'] for r in result['methods'].values())
    result['pending_method_slots']=len(pending)*len(METHODS)
    result['status']='partial' if pending else 'complete_with_failures' if result['failed_method_slots'] else 'complete'
    return result


def calibrate(block,*,alpha=.1,expected_per_family=9):
    if not isinstance(expected_per_family,int) or isinstance(expected_per_family,bool) or expected_per_family<1:
        raise ValueError('positive integer expected_per_family required')
    if block['pending_seeds']:raise ValueError('calibration is missing planned histories')
    rows=block['parsed_records'];counts=Counter(r['family'] for r in rows)
    if counts!={family:expected_per_family for family in FAMILIES}:
        raise ValueError('calibration needs '+str(expected_per_family)+' independent histories per declared family')
    thresholds=dict(schema=CAL_SCHEMA,methods=list(METHODS),families=list(FAMILIES),alpha=alpha,
        expected_per_family=expected_per_family,seeds=block['observed_seeds'],trial_files=block['input_files'],
        **{key:block[key] for key in ('policy_id','config_sha256','source_manifest_sha256')},
        calibration={method:{family:collector.rank_threshold([r['methods'][method]['calibration_score']
            for r in rows if r['family']==family],alpha) for family in FAMILIES} for method in METHODS},
        score='max(0,J_true-min J); failed method/history -> inf',
        scope='method-wise family-conditional rank calibration for independent declared synthetic histories; no topology probability')
    thresholds['thresholds_id']=collector.object_hash(thresholds)
    return thresholds


def evaluate(block,thresholds):
    if thresholds.get('schema')!=CAL_SCHEMA or thresholds.get('families')!=list(FAMILIES):
        raise ValueError('strict two-family thresholds required')
    if collector.object_hash({k:v for k,v in thresholds.items() if k!='thresholds_id'})!=thresholds.get('thresholds_id'):
        raise ValueError('threshold content/hash mismatch')
    if thresholds.get('methods')!=list(METHODS):raise ValueError('threshold methods differ')
    if any(thresholds.get(key)!=block.get(key) for key in ('policy_id','config_sha256','source_manifest_sha256')):
        raise ValueError('calibration/evaluation frozen policy differs')
    if set(thresholds['seeds'])&set(block['observed_seeds']):raise ValueError('calibration/evaluation seeds overlap')
    for method in METHODS:
        if set(thresholds['calibration'].get(method,{}))!=set(FAMILIES):raise ValueError('threshold family coverage differs')
    decisions=[]
    by_seed={r['seed']:r for r in block['parsed_records']}
    for seed in block['planned_seeds']:
        row=by_seed.get(seed)
        if row is None:continue
        decision=dict(seed=seed,family=row['family'],candidate_miss=row['candidate_miss'],methods={})
        for method in METHODS:
            fit=row['methods'][method];failed=fit['status']!='complete'
            if failed:kept=list(FAMILIES)
            else:
                minimum=min(fit['losses'].values())
                kept=[name for name in FAMILIES if fit['losses'][name]-minimum<=
                    collector._decode(thresholds['calibration'][method][name]['threshold'])]
            false=row['candidate_miss'] or row['family'] not in kept
            decision['methods'][method]=dict(retained=kept,false_reject=bool(false),
                unique_correct=bool(not false and len(kept)==1),unique_incorrect=bool(false and len(kept)==1),
                unresolved=len(kept)!=1,failed_keep_all=failed)
        decisions.append(decision)
    def count(rows,method,total):
        vals=[r['methods'][method] for r in rows]
        return dict(total=total,observed=len(vals),pending=total-len(vals),
            **{key:sum(r[key] for r in vals) for key in
                ('false_reject','unique_correct','unique_incorrect','unresolved','failed_keep_all')},
            false_reject_rate=sum(r['false_reject'] for r in vals)/len(vals) if vals else None)
    results={}
    # Planned true family follows the frozen seed-position alternation. Missing
    # histories stay pending; no absent third/fourth family is manufactured.
    planned_family={seed:FAMILIES[index%2] for index,seed in enumerate(block['planned_seeds'])}
    for method in METHODS:
        results[method]=dict(overall=count(decisions,method,len(block['planned_seeds'])),
            by_true_family={family:count([r for r in decisions if r['family']==family],method,
                sum(f==family for f in planned_family.values())) for family in FAMILIES})
    return dict(thresholds_id=thresholds['thresholds_id'],decisions=decisions,evaluation=results,
        alpha=thresholds['alpha'],pending_seeds=block['pending_seeds'])


def render(report):
    lines=['# 严格两候选、固定点EV完整流程','',
        '候选预先固定为star与clade_01；每拓扑一组训练R/X。独立历史是统计单位。','']
    fmt=lambda x:'—' if x is None else format(x,'.6g')
    for phase in ('development','calibration','evaluation'):
        if phase not in report:continue
        block=report[phase]
        lines.extend(['## '+phase,'',f"历史 {block['observed_histories']}/{block['planned_histories']}；候选漏真 {block['candidate_omissions']}。",'',
            '|方法|完整/计划|规划失败|M均值|M中位数|完整运行秒数均值|',
            '|---|---:|---:|---:|---:|---:|'])
        for name in METHODS:
            row=block['methods'][name]
            lines.append(f"|{name}|{row['complete']}/{row['planned']}|{row['planning_unsuccessful']}|{fmt(row['M']['mean'])}|{fmt(row['M']['median'])}|{fmt(row['runtime']['end_to_end_standalone_seconds']['mean'])}|")
        lines.extend(['','|同历史配对|有效/计划|M差均值|胜/平/负|','|---|---:|---:|---:|'])
        for label,row in block['paired_contrasts'].items():
            s=row['delta_M'];lines.append(f"|{label}|{s['valid']}/{s['total']}|{fmt(s['mean'])}|{s['positive']}/{s['ties']}/{s['negative']}|")
        lines.append('')
    if 'calibration_thresholds' in report:
        lines.extend(['## 冻结阈值与独立评价','','阈值在打开评价数据之前保存；按方法、真实族分别校准。','',
            '|方法|star阈值|clade_01阈值|','|---|---:|---:|'])
        for name in METHODS:
            values=report['calibration_thresholds'][name]
            lines.append('|'+name+'|'+'|'.join(str(values[f]['threshold']) for f in FAMILIES)+'|')
        if 'decisions_by_method' in report:
            lines.extend(['','|方法|评价历史|误拒绝|唯一正确|唯一错误|未决|计算失败保留全部|','|---|---:|---:|---:|---:|---:|---:|'])
            for name in METHODS:
                row=report['decisions_by_method'][name]['overall']
                lines.append('|'+name+'|'+'|'.join(str(row[k]) for k in
                    ('total','false_reject','unique_correct','unique_incorrect','unresolved','failed_keep_all'))+'|')
    lines.extend(['','M为验证与未来实际SSE差；预测距离及对比分数单独保存。',
        '负荷主成分情景是训练残差产生的确定性压力测试，没有覆盖率声明。',
        '模型距离、MILP界及校准频率均不作为拓扑正确概率。失败与未完成分母保留。',''])
    return '\n'.join(lines)


def analyze(output,*,development=None,calibration=None,evaluation=None,alpha=.1,expected_per_family=9):
    out=Path(output)
    paths=[Path(p) for p in (development,calibration,evaluation) if p is not None]
    if not paths or (development is not None and (calibration is not None or evaluation is not None)):
        raise ValueError('choose development or calibration/evaluation')
    if any(out.resolve()==path.resolve() for path in paths):raise ValueError('analysis output differs from raw directories')
    report=dict(schema=SCHEMA,methods=list(METHODS),families=list(FAMILIES),independent_unit='one history',
        analyzer=dict(imports='standard library and read-only collectors',new_AC_calls=0,new_fit_calls=0,
            new_selector_calls=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        topology_probability_reported=False)
    if development is not None:
        report['development']=read_phase(development,'development');report['status']=report['development']['status']
    else:
        if calibration is None:raise ValueError('formal analysis requires calibration')
        block=read_phase(calibration,'calibration');thresholds=calibrate(block,alpha=alpha,expected_per_family=expected_per_family)
        out.mkdir(parents=True,exist_ok=True);threshold_path=out/'thresholds.json'
        if threshold_path.exists() and collector.read_json(threshold_path)!=thresholds:
            raise ValueError('existing thresholds differ; preserve prior analysis')
        if not threshold_path.exists():collector.write_json(threshold_path,thresholds)
        threshold_hash=hashlib.sha256(threshold_path.read_bytes()).hexdigest()
        report.update(calibration=block,calibration_thresholds=thresholds['calibration'],alpha=alpha,
            thresholds_id=thresholds['thresholds_id'],threshold_file_sha256=threshold_hash,
            thresholds_persisted_before_evaluation_read=True,status='calibration_only')
        if evaluation is not None:
            eval_block=read_phase(evaluation,'evaluation')
            evaluated=evaluate(eval_block,collector.read_json(threshold_path))
            if hashlib.sha256(threshold_path.read_bytes()).hexdigest()!=threshold_hash:
                raise ValueError('threshold file changed during evaluation')
            report.update(evaluation=eval_block,decisions_by_method=evaluated['evaluation'],
                decisions=evaluated['decisions'],status=eval_block['status'])
            collector.write_json(out/'evaluation.json',evaluated)
    collector.write_json(out/'summary.json',report)
    (out/'summary.md').write_text(render(report),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path);parser.add_argument('--calibration',type=Path)
    parser.add_argument('--evaluation',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--alpha',type=float,default=.1);parser.add_argument('--expected-per-family',type=int,default=9)
    args=parser.parse_args();report=analyze(args.output,development=args.development,calibration=args.calibration,
        evaluation=args.evaluation,alpha=args.alpha,expected_per_family=args.expected_per_family)
    print(json.dumps(dict(status=report['status'],output=str(args.output.resolve()))))


if __name__=='__main__':main()
