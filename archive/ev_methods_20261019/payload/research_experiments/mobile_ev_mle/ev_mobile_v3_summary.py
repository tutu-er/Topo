"""Read saved v3 pilot results; no AC, fitting, optimization or threshold reuse.

Every declared seed is one historical unit. Missing/fatal records retain all
eight method slots. Development and evaluation are never pooled.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

METHODS=('constant','highlow','square_fixed','square_mobile','scalar_fixed','scalar_mobile','projection_fixed','projection_mobile')
PHASES=('development','evaluation')
TOL=1e-7


def number(value):
    return float(value) if isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value) else None


def stats(values):
    values=[v for v in values if number(v) is not None]
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)) if values else dict(n=0,mean=None,median=None,min=None,max=None)


def minimum_set(scores,*,atol=1e-8,rtol=1e-10):
    if not scores or any(number(v) is None for v in scores.values()):return None
    best=min(scores.values());tolerance=atol+rtol*abs(best)
    return sorted(k for k,v in scores.items() if v<=best+tolerance)


def candidate_dimensions(record):
    rows=record.get('training_result',{}).get('fits',[]);result={}
    for row in rows:
        name=row.get('candidate',row.get('name'));r,x=row.get('r'),row.get('x')
        if not isinstance(name,str) or name in result or not isinstance(r,list) or not r or not isinstance(x,list) or len(r)!=len(x):return {}
        if any(number(v) is None for v in r+x):return {}
        result[name]=2*len(r)
    return result


def relative_line_error(truth,estimate):
    """Finite saved vectors only; a missing estimate never becomes zero error."""
    if not isinstance(truth,(list,tuple)) or not truth:
        return dict(valid=False,value=None,reason='missing truth vector')
    if not isinstance(estimate,(list,tuple)) or not estimate:
        return dict(valid=False,value=None,reason='missing fitted vector')
    if len(truth)!=len(estimate) or any(number(v) is None for v in list(truth)+list(estimate)):
        return dict(valid=False,value=None,reason='nonfinite or mismatched line vectors')
    denominator=sum(v*v for v in truth)
    if denominator<=0:return dict(valid=False,value=None,reason='zero truth norm')
    return dict(valid=True,value=math.sqrt(sum((v-w)**2 for v,w in zip(estimate,truth))/denominator),reason=None)


def parameter_errors(record,method):
    truth=record.get('truth') or {};label=record.get('true_name')
    matches=[r for r in record.get('training_result',{}).get('fits',[]) if r.get('candidate',r.get('name'))==label]
    training=matches[0] if len(matches)==1 else {}
    method=method if isinstance(method,dict) else {}
    comparison=(method.get('evaluation') or {}).get('comparisons',{}).get(label,{})
    post=(comparison.get('joint') or {}).get('post') or {};state=post.get('state') or {}
    errors={}
    for stage,estimate in (('training',training),('joint',state)):
        for channel in ('r','x'):
            errors[stage+'_'+channel.upper()]=relative_line_error(truth.get(channel),estimate.get(channel))
    return dict(errors=errors,training_fit_success=training.get('success'),joint_fit_success=post.get('success'),
        validity_rule='matching finite saved vectors and positive truth norm; fit/operation success remains separately reported')


def _bounds(raw):
    """Keep separate bound scopes; values from different proxies are not pooled."""
    keep=('success','objective','Q','exact_score','objective_lower','objective_upper','best_bound','gap','mip_gap','completed','unfinished','selected_origin','bound_scope','scope','seconds')
    result=dict(top={k:raw[k] for k in keep if k in raw},masters=[])
    for i,row in enumerate(raw.get('steps',[])):
        master=row.get('master') or {}
        if master:result['masters'].append(dict(path=f'steps[{i}].master',**{k:master[k] for k in keep if k in master},solver=master.get('solver')))
    for i,row in enumerate(raw.get('grids',[])):
        master=row.get('raw') or {}
        if master:result['masters'].append(dict(path=f'grids[{i}].raw',**{k:master[k] for k in keep if k in master}))
    return result


def project_method(record,name,*,candidate_count=4):
    methods=record.get('methods',{});method=methods.get(name) if isinstance(methods,dict) else None
    result=dict(name=name,present=isinstance(method,dict),usable=False,M=None,M_AIC=None,issues=[],planning_status='missing',execution_status='missing',fit_status='missing',
        raw_argmin=None,AIC_argmin=None,raw_unique_correct=None,AIC_unique_correct=None,raw_contains_truth=None,AIC_contains_truth=None,
        operation={},planning={},parameter_errors=parameter_errors(record,method))
    if not isinstance(method,dict):return result
    metadata=record.get('planning',{}).get(name,{}) or {};raw=metadata.get('raw') or {}
    planning_success=method.get('planning_success',raw.get('success'))
    result['planning_status']=('not_applicable' if name in ('constant','highlow') else
        'success' if planning_success is True else 'failed' if planning_success is False else 'unknown')
    result['planning_failure']=method.get('planning_failure',raw.get('failure'))
    result['planning']=dict(incremental_seconds=number(metadata.get('seconds')),standalone_seconds=number(metadata.get('standalone_seconds')),
        context_seconds=number(metadata.get('context_seconds')),engine_initialization_seconds=number(metadata.get('engine_initialization_seconds')),
        soft_budget_exceeded=metadata.get('soft_budget_exceeded'),selected_origin=raw.get('selected_origin'),fallback_origin=raw.get('fallback_origin'),
        selected_predictive_AC_score=number(raw.get('selected_score')) if name.startswith(('scalar_','projection_')) else None,
        parent_score=number(raw.get('parent_score')),parent_origin=raw.get('parent_origin'),
        accepted_new_steps=sum(s.get('accepted') is True for s in raw.get('steps',[])),
        final_new_projection_plan=isinstance(raw.get('selected_origin'),str) and raw['selected_origin'].startswith('shared_projection_MILP_'),
        deadline_reached=raw.get('deadline_reached',raw.get('time_budget_exceeded')),solver_bounds=_bounds(raw),
        failure_count=len(raw.get('failures',[])),
        objective_family='precision_square' if name.startswith('square_') else 'fixed_line_predictive_distance' if name.startswith('scalar_') else 'shared_line_predictive_half_gap' if name.startswith('projection_') else 'public_curve')
    op=method.get('operation') or {};audit=op.get('audit') or {};feasible=audit.get('feasible')
    result['execution_status']='success' if feasible is True else 'failed' if feasible is False else 'missing'
    throughput,grid=number(audit.get('throughput_kwh')),number(audit.get('grid_energy_kwh'))
    charge=(throughput+grid)/2 if throughput is not None and grid is not None else None
    discharge=(throughput-grid)/2 if throughput is not None and grid is not None else None
    if charge is not None and (charge < -TOL or discharge < -TOL):
        result['issues'].append('negative derived charging/discharging energy');charge=discharge=None
    result['operation']=dict(moves=number(audit.get('moves')),worker_minutes=number(audit.get('driver_cost')) if audit.get('cost_units')=='worker_minutes' else None,
        worker_cost_units=audit.get('cost_units'),driving_energy_kwh=number(audit.get('travel_energy_kwh')),grid_energy_kwh=grid,throughput_kwh=throughput,
        charging_energy_kwh=max(0.,charge) if charge is not None else None,discharging_energy_kwh=max(0.,discharge) if discharge is not None else None,
        conversion_loss_kwh=number(audit.get('conversion_loss_kwh')),route_power_error_kw=number(method.get('route_power_error_kw')),
        battery_energy_kwh=op.get('energy_kwh'),charging_slot_count=sum(any(number(v) is not None and v>TOL for v in row) for row in method.get('power_kw',[])),
        discharging_slot_count=sum(any(number(v) is not None and v<-TOL for v in row) for row in method.get('power_kw',[])))
    evaluation=method.get('evaluation') or {};losses=evaluation.get('losses') or {};truth=record.get('true_name')
    loss_ok=(isinstance(losses,dict) and len(losses)==candidate_count and truth in losses and all(number(v) is not None for v in losses.values()))
    complete=evaluation.get('complete') is True and loss_ok
    result['fit_status']='success' if complete else 'failed' if evaluation.get('complete') is False else 'invalid' if evaluation else 'missing'
    result['evaluation_failures']=evaluation.get('failures',{})
    result['evaluation_seconds']=number(evaluation.get('seconds'))
    result['usable']=bool(feasible is True and complete and method.get('success') is True)
    if method.get('success') is True and not result['usable']:result['issues'].append('saved success disagrees with operation/fit evidence')
    if complete:
        M=min(v for k,v in losses.items() if k!=truth)-losses[truth]
        if number(evaluation.get('M')) is None or abs(evaluation['M']-M)>TOL*(1+abs(M)):
            result['issues'].append('saved M mismatch');result['usable']=False
        result['saved_losses']=losses;result['diagnostic_M']=M
        if result['usable']:
            result['M']=M;result['raw_argmin']=minimum_set(losses)
            dims=candidate_dimensions(record)
            if set(dims)!=set(losses):result['issues'].append('AIC dimensions absent or inconsistent')
            else:
                scores={k:v+2*dims[k] for k,v in losses.items()}
                result['AIC_argmin']=minimum_set(scores)
                result['M_AIC']=min(v for k,v in scores.items() if k!=truth)-scores[truth]
            for tag in ('raw','AIC'):
                selected=result[tag+'_argmin']
                result[tag+'_unique_correct']=None if selected is None else selected==[truth]
                result[tag+'_contains_truth']=None if selected is None else truth in selected
    return result


def paired(rows,a,b,*,tol=TOL):
    values=[];win=tie=loss=missing=0;predicted=[];reversal=0;prediction_tie_actual_change=0
    for row in rows:
        A,B=row['methods'][a],row['methods'][b]
        if A['M'] is None or B['M'] is None:missing+=1;continue
        delta=A['M']-B['M'];values.append(delta)
        if delta>tol:win+=1
        elif delta<-tol:loss+=1
        else:tie+=1
        pa=A['planning'].get('selected_predictive_AC_score');pb=B['planning'].get('selected_predictive_AC_score')
        # Scores are comparable within the same objective family only.
        same_family=a.rsplit('_',1)[0]==b.rsplit('_',1)[0] and a.startswith(('scalar_','projection_'))
        if same_family and pa is not None and pb is not None:
            dp=pa-pb;predicted.append(dict(seed=row['spec']['seed'],delta_predicted_AC=dp,delta_actual_M=delta,
                reversed=abs(dp)>tol and abs(delta)>tol and dp*delta<0,
                predicted_gain_actual_loss=dp>tol and delta<-tol))
            reversal+=int(abs(dp)>tol and abs(delta)>tol and dp*delta<0)
            prediction_tie_actual_change+=int(abs(dp)<=tol and abs(delta)>tol)
    return dict(a=a,b=b,denominator=len(rows),complete_pairs=len(values),missing_pairs=missing,a_wins=win,ties=tie,a_losses=loss,
        delta_M=stats(values),comparison='a minus b; larger M is better',prediction_actual=dict(complete_pairs=len(predicted),
            missing_or_not_applicable=len(rows)-len(predicted),
            reversal_count=reversal,predicted_gain_actual_loss_count=sum(r['predicted_gain_actual_loss'] for r in predicted),
            predicted_tie_actual_change=prediction_tie_actual_change,rows=predicted))


def aggregate(rows):
    methods={}
    for name in METHODS:
        values=[row['methods'][name] for row in rows];good=[v for v in values if v['M'] is not None]
        operations=[v['operation'] for v in values]
        masters=[master for v in values for master in v['planning'].get('solver_bounds',{}).get('masters',[])]
        tops=[v['planning'].get('solver_bounds',{}).get('top',{}) for v in values]
        methods[name]=dict(denominator=len(rows),present=sum(v['present'] for v in values),usable=len(good),missing_or_failed=len(rows)-len(good),
            M=stats(v['M'] for v in good),M_negative=sum(v['M']<0 for v in good),M_zero_with_tolerance=sum(abs(v['M'])<=TOL for v in good),
            M_AIC=stats(v['M_AIC'] for v in values),M_AIC_negative=sum(v['M_AIC'] is not None and v['M_AIC']<0 for v in values),
            parameter_errors={key:dict(denominator=len(rows),valid=sum(v['parameter_errors']['errors'][key]['valid'] for v in values),
                missing=sum(not v['parameter_errors']['errors'][key]['valid'] for v in values),
                **stats(v['parameter_errors']['errors'][key]['value'] for v in values)) for key in ('training_R','training_X','joint_R','joint_X')},
            planning_status=dict(Counter(v['planning_status'] for v in values)),execution_status=dict(Counter(v['execution_status'] for v in values)),fit_status=dict(Counter(v['fit_status'] for v in values)),
            decisions={tag:dict(known=sum(v[tag+'_unique_correct'] is not None for v in values),unique_correct=sum(v[tag+'_unique_correct'] is True for v in values),
                contains_truth=sum(v[tag+'_contains_truth'] is True for v in values),ties=sum(v[tag+'_argmin'] is not None and len(v[tag+'_argmin'])>1 for v in values)) for tag in ('raw','AIC')},
            moved=sum(o.get('moves') is not None and o['moves']>0 for o in operations),moves=stats(o.get('moves') for o in operations),
            worker_minutes=stats(o.get('worker_minutes') for o in operations),driving_energy_kwh=stats(o.get('driving_energy_kwh') for o in operations),
            charging_energy_kwh=stats(o.get('charging_energy_kwh') for o in operations),discharging_energy_kwh=stats(o.get('discharging_energy_kwh') for o in operations),
            discharged=sum(o.get('discharging_energy_kwh') is not None and o['discharging_energy_kwh']>TOL for o in operations),
            planning_incremental_seconds=stats(v['planning'].get('incremental_seconds') for v in values),
            planning_standalone_seconds=stats(v['planning'].get('standalone_seconds') for v in values),
            soft_budget_exceeded=sum(v['planning'].get('soft_budget_exceeded') is True for v in values),
            accepted_projection_steps=sum(v['planning'].get('accepted_new_steps',0) for v in values),
            final_new_projection_plans=sum(v['planning'].get('final_new_projection_plan') is True for v in values),
            solver_summary=dict(master_attempts=len(masters),master_feasible=sum(m.get('success') is True for m in masters),
                master_completed=sum(m.get('completed') is True for m in masters),master_explicit_unfinished=sum(m.get('unfinished') is True for m in masters),
                completed_top=sum(m.get('completed') is True for m in tops),explicit_unfinished_top=sum(m.get('unfinished') is True for m in tops),
                master_proxy_gaps=stats(m.get('gap') for m in masters)),
            origins=dict(Counter(str(v['planning'].get('selected_origin',v['planning'].get('fallback_origin'))) for v in values)),
            issues=sum(bool(v['issues']) for v in values))
    pairs={a+'__vs__'+b:paired(rows,a,b) for a,b in combinations(METHODS,2)}
    # Explicit challenger-vs-parent directions for the requested sign test.
    mobility={family:paired(rows,family+'_mobile',family+'_fixed') for family in ('square','scalar','projection')}
    return dict(history_denominator=len(rows),history_record_status=dict(Counter(r['record_status'] for r in rows)),methods=methods,pairs=pairs,mobility=mobility)


def summarize_records(specs,records,*,candidate_count=4):
    """Pure function used by the file reader and synthetic tests."""
    if len({s['seed'] for s in specs})!=len(specs):raise ValueError('one independent seed required per declared history')
    if len({s['index'] for s in specs})!=len(specs):raise ValueError('duplicate declared history index')
    expected={s['index']:s for s in specs}
    if set(records)-set(expected):raise ValueError('extra record index outside frozen phase')
    rows=[]
    for index,spec in sorted(expected.items()):
        record=records.get(index);r=dict(spec=spec,record_status='missing_record' if record is None else record.get('status','unknown'),methods={})
        if record is not None and record.get('spec')!=spec:raise ValueError('record identity differs from frozen specification')
        for name in METHODS:r['methods'][name]=project_method(record or {},name,candidate_count=candidate_count)
        r['true_name']=None if record is None else record.get('true_name');r['failure']=None if record is None else record.get('failure')
        rows.append(r)
    return dict(overall=aggregate(rows),by_bank_H={f'{bank}/H{H}':aggregate([r for r in rows if r['spec']['bank']==bank and r['spec']['H']==H])
        for bank,H in sorted({(s['bank'],s['H']) for s in specs})},rows=rows)


def _read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data),dict(path=str(Path(path).resolve()),bytes=len(data),sha256=sha256(data).hexdigest())


def run(root,output,*,phases=PHASES,require_complete=False):
    root,output=Path(root),Path(output)
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('preserve previous summary; choose new output')
    freeze,freeze_source=_read(root/'freeze.json')
    if tuple(freeze['methods'])!=METHODS:raise ValueError('expected the eight declared v3 methods')
    if not set(phases)<=set(PHASES):raise ValueError('unknown phase')
    all_specs=[s for p in PHASES for s in freeze['specifications'][p]]
    if len({s['seed'] for s in all_specs})!=len(all_specs):raise ValueError('development/evaluation seed overlap or repeated independent history')
    result=dict(schema='mobile_v3_saved_pilot_summary_v1',created_utc=datetime.now(timezone.utc).isoformat(),policy_id=freeze['policy_id'],
        source_sha256=freeze['source_sha256'],freeze_source=freeze_source,methods=list(METHODS),phases={},input_files=[],
        scope='Descriptive paired pilot only; no rejection thresholds, probability interpretation or new policy selection.',
        independent_unit='one frozen history seed; eight methods are paired observations within that seed.',
        missing_rule='Each missing/fatal record retains all eight method slots and every associated paired denominator.',
        AIC_rule='Working score L+2d, d=2*actual training R/X vector length. No calibration or asymptotic claim.',
        M_AIC_rule='Secondary diagnostic min_wrong(L_b+2d_b)-(L_true+2d_true); primary M and its paired comparisons are unchanged.',
        parameter_error_rule='Offline true-candidate ||estimated R/X-true R/X||_2 / ||true R/X||_2, separately for training and saved final joint state. Missing vectors retain denominators; numeric validity and fit success are distinct.',
        comparison_tolerance=TOL,M_negative_rule='M<0 exactly; paired ties use comparison_tolerance.',
        operation_energy_rule='Charging/discharging grid-side energies=(throughput +/- net-grid energy)/2.',
        bound_rule='Saved solver bounds belong to each stated proxy; they are never compared across objective families or called AC bounds.',
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    for phase in phases:
        specs=freeze['specifications'][phase];records={};expected={str(s['index']).zfill(3)+'.json.gz':s for s in specs}
        for path in sorted((root/phase).glob('*.json.gz')):
            if path.name not in expected:raise ValueError('extra history file '+str(path))
            record,source=_read(path);result['input_files'].append(source)
            if record.get('policy_id')!=freeze['policy_id']:raise ValueError('record policy differs from freeze')
            records[expected[path.name]['index']]=record
        if require_complete and len(records)!=len(specs):raise ValueError('declared phase still has missing histories')
        result['phases'][phase]=summarize_records(specs,records,candidate_count=freeze['config']['candidate_count'])
        result['phases'][phase]['all_declared_files_present']=len(records)==len(specs)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2),encoding='utf8')
    render(result,output.with_suffix('.md'));return result


def render(data,path):
    text=['# v3 配对算例描述统计','',
        '每个种子是一条独立历史，八种方法在同一历史上配对。开发与评价分开。N 是预声明历史分母；均值和中位数仅对可执行且最终拟合完整的 M 计算，同时保留失败和缺失数。原始/AIC 最小值正确数是描述性排序，没有套用旧阈值。',
        '',r'另存补充量 $M_{\rm AIC}=\min_{b\ne a}(L_b+2d_b)-(L_a+2d_a)$，$d_k=2m_k$。它为工作评分提供差距量级，保留预定原始 $M$ 作为主要配对指标。','']
    for phase,p in data['phases'].items():
        text.extend([f'## {phase}','',
            '|银行/H|方法|N/完整|M均值|M中位|M<0|M_AIC均值|raw正确/有效；AIC正确/有效|规划失败/缺失|执行失败/缺失|拟合失败/缺失|移动数|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for group,g in p['by_bank_H'].items():
            for name,m in g['methods'].items():
                fmt=lambda v:'—' if v is None else f'{v:.3f}'
                fail=lambda d:sum(v for k,v in d.items() if k not in ('success','not_applicable'))
                decisions='；'.join(f"{m['decisions'][tag]['unique_correct']}/{m['decisions'][tag]['known']}" for tag in ('raw','AIC'))
                text.append(f"|{group}|{name}|{m['denominator']}/{m['usable']}|{fmt(m['M']['mean'])}|{fmt(m['M']['median'])}|{m['M_negative']}|{fmt(m['M_AIC']['mean'])}|{decisions}|{fail(m['planning_status'])}|{fail(m['execution_status'])}|{fail(m['fit_status'])}|{m['moved']}|")
        text.extend(['','|银行/H|移动相对固定|胜/平/负/缺失|平均 ΔM|预测分数升而实际 M 降|预测/实际完整对数|','|---|---|---:|---:|---:|---:|'])
        for group,g in p['by_bank_H'].items():
            for family,pair in g['mobility'].items():
                mean=pair['delta_M']['mean'];mean='—' if mean is None else f'{mean:.3f}'
                text.append(f"|{group}|{family}|{pair['a_wins']}/{pair['ties']}/{pair['a_losses']}/{pair['missing_pairs']}|{mean}|{pair['prediction_actual']['predicted_gain_actual_loss_count']}|{pair['prediction_actual']['complete_pairs']}|")
        text.append('')
    text.extend(['完整 JSON 还保存全部方法对的配对比较、逐种子 M/损失与最小候选集合、驾驶分钟/能量/充放电、规划时间、projection 接受与继承来源、各步求解界及其适用范围。失败回退动作的执行结果与规划失败分列。',
        '真候选训练与最终共享拟合的 R/X 相对 L2 误差另作事后诊断，按银行/H/方法保留有效数、缺失数、均值与中位数。缺真值或状态时不造零；数值有效与拟合成功分别记录，真实参数不进入选择器。',''])
    Path(path).write_text('\n'.join(text),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--phase',choices=(*PHASES,'all'),default='all')
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    out=args.output or args.root/'summary_v1.json';phases=PHASES if args.phase=='all' else (args.phase,)
    result=run(args.root,out,phases=phases,require_complete=args.require_complete)
    print(json.dumps(dict(output=str(out),phases={k:v['overall']['history_denominator'] for k,v in result['phases'].items()})),flush=True)


if __name__=='__main__':main()
