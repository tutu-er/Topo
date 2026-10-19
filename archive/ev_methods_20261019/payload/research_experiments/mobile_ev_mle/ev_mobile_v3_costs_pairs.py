"""Read-only pilot2 cost, operation and matched-domain descriptions.

Uses the final saved summary and independently reads all 16 evaluation raws.
No bootstrap, calibration, fitting, simulation or policy adjustment.
"""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import gradient_price_study as io


def stats(values,denominator):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(denominator=denominator,n=len(values),missing=denominator-len(values),
        mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None,
        min=min(values) if values else None,max=max(values) if values else None)


def error(truth,estimate):
    if estimate is None:return None
    a,b=np.asarray(truth,float),np.asarray(estimate,float)
    if a.shape!=b.shape or not np.isfinite(b).all() or np.linalg.norm(a)==0:return None
    return float(np.linalg.norm(b-a)/np.linalg.norm(a))


def project(raw,method,summary_row,tol):
    actual=raw.get('methods',{}).get(method,{})
    meta=raw.get('planning',{}).get(method,{})
    p=meta.get('raw',{});counter=p.get('counters',{});response=counter.get('response_calls',{})
    operation=actual.get('operation',{}).get('audit',{})
    truth=raw['true_name'];fits={f.get('candidate',f.get('name')):f for f in raw['training_result']['fits']}
    joint=actual.get('evaluation',{}).get('comparisons',{}).get(truth,{}).get('joint',{}).get('post',{})
    final=joint.get('state',{})
    errs={stage+'_'+key.upper():error(raw['truth'][key],fit.get(key))
        for stage,fit in (('training',fits.get(truth,{})),('joint',final)) for key in ('r','x')}
    candidates=[]
    for r in p.get('initials',p.get('initial_candidates',[])):
        power=r.get('power_kw')
        if power is None:power=(r.get('profile') or {}).get('power_kw')
        if power is None:power=(r.get('operation') or {}).get('u_kw')
        if power is not None:candidates.append((r.get('origin'),np.asarray(power,float)))
    u=np.asarray(actual.get('power_kw'),float)
    matches=[name for name,power in candidates if power.shape==u.shape and np.allclose(power,u,rtol=0,atol=tol)]
    if method.startswith('projection'):
        accepted=sum(r.get('accepted') is True for r in p.get('steps',[]))
        new=isinstance(p.get('selected_origin'),str) and p['selected_origin'].startswith('shared_projection_MILP_')
    elif method.startswith('scalar'):
        accepted=sum(r.get('accepted') is True for r in p.get('grids',[]))
        new=p.get('selected_is_new_MILP_proposal')
    else:accepted=new=None
    M=actual.get('evaluation',{}).get('M') if summary_row.get('usable') else None
    operation_small={k:operation.get(k) for k in ('moves','travel_energy_kwh','throughput_kwh','grid_energy_kwh','conversion_loss_kwh')}
    operation_small['worker_minutes']=operation.get('driver_cost') if operation.get('cost_units')=='worker_minutes' else None
    small=dict(method=method,M=M,planning_success=actual.get('planning_success'),
        execution_success=operation.get('feasible'),fit_success=actual.get('evaluation',{}).get('complete'),
        incremental_seconds=meta.get('seconds'),standalone_seconds=meta.get('standalone_seconds'),
        soft_budget_exceeded=meta.get('soft_budget_exceeded'),origin=p.get('selected_origin') or p.get('fallback_origin'),
        accepted_new_steps=accepted,final_new_optimized_plan=new,
        retained_initial_power=bool(matches) if candidates else None,matched_initial_origins=matches,
        observed_initial_action_count=len(candidates),operation=operation_small,parameter_errors=errs,
        source_AC_value_calls=response.get('ac_value_calls'),source_AC_jacobian_calls=response.get('ac_jacobian_calls'),
        source_solver_calls=counter.get('source_solver_calls'),source_total_nfev=counter.get('source_total_nfev'),
        source_pair_fit_calls=counter.get('pair_fit_calls'),joint_fit_calls=p.get('joint_fit_calls'),
        joint_profile_evaluations=p.get('profile_evaluations'))
    return small


def aggregate(rows,methods,tol):
    n=len(rows);result={}
    keys=('incremental_seconds','standalone_seconds','source_AC_value_calls','source_AC_jacobian_calls',
        'source_solver_calls','source_total_nfev','source_pair_fit_calls','joint_fit_calls','joint_profile_evaluations')
    for method in methods:
        selected=[r['methods'][method] for r in rows]
        result[method]=dict(denominator=n,
            **{key:stats([r[key] for r in selected],n) for key in keys},
            planning_success=sum(r['planning_success'] is True for r in selected),
            planning_failure=sum(r['planning_success'] is False for r in selected),
            planning_status_unavailable=sum(r['planning_success'] is None for r in selected),
            execution_success=sum(r['execution_success'] is True for r in selected),
            fit_success=sum(r['fit_success'] is True for r in selected),
            soft_budget_exceeded=sum(r['soft_budget_exceeded'] is True for r in selected),
            origins=dict(Counter(str(r['origin']) for r in selected)),
            initial_action_retained=sum(r['retained_initial_power'] is True for r in selected),
            initial_action_identity_available=sum(r['retained_initial_power'] is not None for r in selected),
            accepted_steps=stats([r['accepted_new_steps'] for r in selected],n),
            accepted_steps_sum=sum(r['accepted_new_steps'] for r in selected if r['accepted_new_steps'] is not None),
            final_new_optimized_plans=sum(r['final_new_optimized_plan'] is True for r in selected),
            final_new_flag_available=sum(r['final_new_optimized_plan'] is not None for r in selected),
            moved=sum(r['operation']['moves'] is not None and r['operation']['moves']>0 for r in selected),
            operation={k:stats([r['operation'][k] for r in selected],n) for k in selected[0]['operation']},
            parameter_errors={k:stats([r['parameter_errors'][k] for r in selected],n) for k in selected[0]['parameter_errors']})
    pairs={}
    for domain in ('fixed','mobile'):
        a,b='projection_'+domain,'square_'+domain;pair_rows=[]
        for row in rows:
            x,y=row['methods'][a],row['methods'][b]
            delta=None if x['M'] is None or y['M'] is None else x['M']-y['M']
            op={k:None if x['operation'][k] is None or y['operation'][k] is None else x['operation'][k]-y['operation'][k]
                for k in x['operation']}
            pair_rows.append(dict(index=row['spec']['index'],seed=row['spec']['seed'],delta_M=delta,
                category='missing' if delta is None else 'win' if delta>tol else 'loss' if delta < -tol else 'tie',operation_difference=op))
        pairs[domain]=dict(a=a,b=b,denominator=n,categories=dict(Counter(r['category'] for r in pair_rows)),
            delta_M=stats([r['delta_M'] for r in pair_rows],n),
            operation_difference={k:stats([r['operation_difference'][k] for r in pair_rows],n) for k in pair_rows[0]['operation_difference']},rows=pair_rows)
    return dict(history_denominator=n,methods=result,projection_vs_same_domain_square=pairs)


def render(data,path):
    f=lambda v:'—' if v is None else f'{v:.3g}'
    energy=lambda v:f(0. if v is not None and abs(v)<1e-7 else v)
    groups=data['groups'];names={'square_fixed':'平方F','square_mobile':'平方M','scalar_fixed':'标量F','scalar_mobile':'标量M','projection_fixed':'投影F','projection_mobile':'投影M'}
    lines=['# pilot2：规划费用与同域对照','',
        '独立评价16历史，n3/n4×H8/H96各4条；8方法共128槽全部保留。F为固定点，M为允许移动。仅描述保存结果，未重拟合、校准或调整策略。','',
        '时间为规划增量/独立执行时间的均值（秒）。独立执行计入所需父计划、context及engine初始化，未包括训练和最终数据推断；60秒为增量软预算。恒充/高低曲线未单独计时，保留为不可得。','',
        '|方法|n3/H8|n3/H96|n4/H8|n4/H96|','|---|---:|---:|---:|---:|']
    for method,label in names.items():
        cells=[f"{f(groups[g]['methods'][method]['incremental_seconds']['mean'])}/{f(groups[g]['methods'][method]['standalone_seconds']['mean'])}" for g in groups]
        lines.append('|'+label+'|'+'|'.join(cells)+'|')
    lines.extend(['','同域投影−平方比较：胜/平/负按原M差与1e−7容忍；电量差依次为驾驶耗电/吞吐/网侧净充电（kWh），均为每历史均差。', '',
        '|组/域|胜/平/负|ΔM均值/中位|Δ驾驶/吞吐/净电|投影新接受/最终新计划/实际移动|',
        '|---|---:|---:|---:|---:|'])
    for group,g in groups.items():
        for domain in ('fixed','mobile'):
            p=g['projection_vs_same_domain_square'][domain];c=p['categories'];d=p['operation_difference'];m=g['methods']['projection_'+domain]
            lines.append(f"|{group}/{'F' if domain=='fixed' else 'M'}|{c.get('win',0)}/{c.get('tie',0)}/{c.get('loss',0)}|{f(p['delta_M']['mean'])}/{f(p['delta_M']['median'])}|"+
                '/'.join(energy(d[k]['mean']) for k in ('travel_energy_kwh','throughput_kwh','grid_energy_kwh'))+
                f"|{m['accepted_steps_sum']}/{m['final_new_optimized_plans']}/{m['moved']}|")
    lines.extend(['','真候选R/X相对误差（L2范数比）的组均值；训练每历史仅算一组，最终按方法分别算。完整8方法、均值/中位/缺失数见JSON。','',
        '|组|训练R/X|最终平方F R/X|最终平方M R/X|最终投影F R/X|最终投影M R/X|',
        '|---|---:|---:|---:|---:|---:|'])
    for group,g in groups.items():
        e=g['methods']['projection_fixed']['parameter_errors'];cells=[f(e['training_R']['mean'])+'/'+f(e['training_X']['mean'])]
        for method in ('square_fixed','square_mobile','projection_fixed','projection_mobile'):
            e=g['methods'][method]['parameter_errors'];cells.append(f(e['joint_R']['mean'])+'/'+f(e['joint_X']['mean']))
        lines.append('|'+group+'|'+'|'.join(cells)+'|')
    lines.extend(['','标量方法保存实际source AC值/导数调用与source_solver_calls（各初值求解次数）；投影仅有joint_fit_calls和profile_evaluations，完整AC调用/重启总数不可得。N4各投影组的joint_fit_calls仅2/4日志可得，另外2条失败规划保留为未知；其耗时已计入时间统计。JSON保留这些字段的有效分母、逐历史值、origin与初值同曲线匹配；未用selected_profile倒推总费用。',''])
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


def run(root):
    root=Path(root);source=root/'summary_final_v2.json';summary=io.read_json(source)
    output=root/'costs_and_pairs_v1.json'
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('Preserve prior cost summary')
    methods=summary['methods'];tol=summary['comparison_tolerance'];rows=[];files=[];checks=[]
    for saved in summary['phases']['evaluation']['rows']:
        spec=saved['spec'];path=root/'evaluation'/(str(spec['index']).zfill(3)+'.json.gz')
        content=path.read_bytes();raw=io.read_json(path)
        if raw['spec']!=spec or raw['policy_id']!=summary['policy_id']:raise ValueError('Raw identity differs from summary')
        row=dict(spec=spec,methods={m:project(raw,m,saved['methods'][m],tol) for m in methods})
        for method in methods:
            ours=row['methods'][method];s=saved['methods'][method]
            for key in ('incremental_seconds','standalone_seconds'):
                checks.append(ours[key]==s['planning'][key])
            checks.append(ours['M']==s['M'])
            for key,value in ours['parameter_errors'].items():
                old=s['parameter_errors']['errors'][key]['value']
                checks.append(value==old if value is None or old is None else abs(value-old)<=1e-12*(1+abs(old)))
        rows.append(row);files.append(dict(path=str(path.resolve()),bytes=len(content),sha256=sha256(content).hexdigest()))
    if len(rows)!=16 or len({r['spec']['seed'] for r in rows})!=16 or len(methods)!=8:raise ValueError('Expected 16 independent histories and 8 methods')
    groups={f'{bank}/H{H}':aggregate([r for r in rows if r['spec']['bank']==bank and r['spec']['H']==H],methods,tol)
            for bank,H in sorted({(r['spec']['bank'],r['spec']['H']) for r in rows})}
    data=dict(schema='mobile_v3_pilot2_costs_pairs_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        policy_id=summary['policy_id'],source_summary=dict(path=str(source.resolve()),sha256=sha256(source.read_bytes()).hexdigest()),
        input_files=files,methods=methods,history_denominator=16,method_slot_denominator=128,
        summary_crosscheck_passed=all(checks),summary_crosscheck_count=len(checks),comparison_tolerance=tol,
        groups=groups,overall=aggregate(rows,methods,tol),rows=rows,
        parameter_error='true-candidate relative Euclidean R/X error; training counted once per history, final fit success separate',
        time_scope='saved incremental/standalone planning only; no training/inference time; incremental soft budget 60 seconds',
        count_scope='source counters are actual saved incremental fields, not ancestor totals; missing counters remain unavailable',
        initial_retention_rule='actual selected power equals saved initial/parent power within 1e-7 kW; protection flags alone do not imply inheritance',
        no_new_AC_or_fits=True,no_bootstrap_or_calibration=True,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    io.write_json(output,data);render(data,output.with_suffix('.md'));return data


if __name__=='__main__':
    answer=run(Path('outputs/mobile_v3_pilot2'))
    print(json.dumps(dict(histories=answer['history_denominator'],slots=answer['method_slot_denominator'],crosscheck=answer['summary_crosscheck_passed']),ensure_ascii=False),flush=True)
