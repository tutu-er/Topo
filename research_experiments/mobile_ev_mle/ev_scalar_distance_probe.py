"""New development-only probe of scalar source-profile chord MILPs.

Plans see only training meters/fits/forecast. Simulator truth and the common
heldout future are used after all plans have been fixed. Existing nine-method
formal policies, calibration and evaluation records are never changed or read.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import ev_scalar_distance_milp as scalar
from . import ev_gradient_flow_design as gradient
from . import ev_gradient_flow_study as previous
from . import ev_fixed_train_design as fixed
from . import ev_dynamic_study as dynamic
from . import gradient_price_study as io
from .information_context import meter_from_record
from .likelihood import Observations,RadialCandidate

HERE=Path(__file__).parent
CONFIG=dict(version='scalar_profile_chord_development_v1',histories=[994101,994102,994103,994104],
    grids=[3,5,9,17],solver_seconds=2.,planning_seconds=35.,backend='gurobi',
    source_fit_multistart=2,acceptance_atol=1e-9,acceptance_rtol=1e-10,
    acceptance='same complete AC mean directed source-profile half-SSE for baselines and proposals',
    independent_evaluation=False,existing_policy_modified=False,formal_records_read=False,
    action_domain='fixed site0 V2G; original SOC/task/access/charge-discharge constraints',
    grid3_scope='two endpoint chords via original C/D; zero additional variables',
    budget_scope='development grid audit: 2s per MILP; 35s cumulative reference logged, not a matched-budget policy test',
    bound_scope='global frozen affine L2 objective only; nonlinear AC checked separately')


def planning_view(record):
    fields=('candidate','r','x','source_p','source_q','success','loss')
    return dict(training=record['training'],training_meter=record['training_meter'],future_meter=record['future_meter'],
        candidates=record['candidates'],forecast=record['forecast'],
        training_fits=[{k:row[k] for k in fields} for row in record['training_result']['fits']],
        config={k:record['config'][k] for k in ('profile','source','horizon')})


def restore_training(view):
    obs=Observations(**view['training']);bank=[]
    for row in view['candidates']:
        family=[tuple(s) for s in row['supports'] if 1<len(s)<obs.n]
        candidate=RadialCandidate.from_supports(row['name'],obs.n,family)
        if candidate.supports!=tuple(tuple(s) for s in row['supports']):raise ValueError('edge order changed')
        bank.append(candidate)
    if [c.name for c in bank]!=['star','clade_01']:raise ValueError('strict predeclared pair required')
    fits={row['candidate']:SimpleNamespace(**{k:np.asarray(row[k],float) if k in
        ('r','x','source_p','source_q') else row[k] for k in ('r','x','source_p','source_q','success','loss')})
        for row in view['training_fits']}
    return obs,bank,fits,meter_from_record(view['training_meter']),meter_from_record(view['future_meter'])


def plan(view,*,seed):
    began=perf_counter();obs,bank,fits,tm,fm=restore_training(view);T,n=view['config']['horizon'],obs.n
    curves=previous.public_curves(T,n);domain=fixed._service(T,n,mobile=False,discharge=True)
    context=fixed.build_fixed_contexts(obs,bank,fits,tm,fm,-fixed.CONFIG['command_power_kw'],
        fixed.CONFIG['command_power_kw'],curves['analytic_square'],forecast=view['forecast'],**view['config']['source'])
    if not context['success']:raise RuntimeError('fixed-line affine context failed: '+str(context['diagnostics']))
    engine=gradient.ProfileEngine(context,fit_options={**view['config']['profile'],**view['config']['source']},seed=seed)
    baselines={name:curves[name] for name in ('constant','highlow')};baselines['square']=curves['analytic_square']
    baseline_records={};feasible=[]
    for name,u in baselines.items():
        operation=fixed._realize(domain,u);profile=engine.evaluate(u,'ac')
        value=gradient._aggregate(profile['pairs'],'binarymean') if profile['success'] else dict(score=None)
        baseline_records[name]=dict(power_kw=u,operation=operation['audit'],profile=profile,score=value['score'])
        if operation['success'] and value['score'] is not None:feasible.append((value['score'],name,u.copy(),profile))
    if not feasible:raise RuntimeError('no complete baseline AC source profile')
    best=max(feasible,key=lambda x:x[0]);initialization=perf_counter()-began
    cache={};grid_records={};choices={name:u.copy() for name,u in baselines.items()};inherited=[]
    for points in CONFIG['grids']:
        grid_began=perf_counter();before=engine.counters()
        table=scalar.build_scalar_tables(context,domain,grid_points=points,cache=cache)
        solved=scalar.solve_scalar_distance_milp(context,domain,grid_points=points,tables=table,
            initial_powers=[best[2]],solver_seconds=CONFIG['solver_seconds'],backend=CONFIG['backend'])
        row=dict(raw=solved,baseline_best_score=max(r['score'] for r in baseline_records.values() if r['score'] is not None),
            before_score=best[0],before_origin=best[1],accepted=False,uses_truth=False,uses_validation=False,
            uses_future_noise=False,initialization_dependency_seconds=initialization,
            prior_grid_dependency_seconds=sum(r['own_seconds'] for r in inherited))
        u=solved.get('power_kw')
        if solved['success'] and u is not None:
            choices['raw_scalar_'+str(points)]=u.copy()
            prediction=engine.evaluate(u,'ac')
            value=gradient._aggregate(prediction['pairs'],'binarymean') if prediction['success'] else dict(score=None)
            row.update(raw_AC_profile=prediction,raw_AC_score=value['score'])
            tolerance=CONFIG['acceptance_atol']+CONFIG['acceptance_rtol']*abs(best[0])
            if value['score'] is not None and value['score']>best[0]+tolerance:
                row['accepted']=True;best=(value['score'],'scalar_'+str(points),u.copy(),prediction)
            else:row['rejection_reason']='complete AC score did not improve or required source fit failed'
        else:row['failure']='no feasible scalar MILP proposal'
        row.update(selected_score=best[0],selected_origin=best[1],selected_power_kw=best[2].copy(),
            selected_AC_profile=best[3],inheritance_protected=best[0]>=row['before_score']-1e-10,
            own_seconds=perf_counter()-grid_began)
        row['cumulative_planning_seconds']=initialization+row['prior_grid_dependency_seconds']+row['own_seconds']
        row['planning_budget_exceeded']=row['cumulative_planning_seconds']>CONFIG['planning_seconds']
        row['profile_cache_before']=before;row['profile_cache_after']=engine.counters()
        grid_records[str(points)]=row;inherited.append(row);choices['scalar_'+str(points)]=best[2].copy()
    return choices,dict(success=True,grids=grid_records,baselines=baseline_records,
        context=context['diagnostics'],initialization_seconds=initialization,seconds=perf_counter()-began,
        counters=engine.counters(),uses_truth=False,uses_validation=False,uses_future_noise=False,
        line_parameters_fixed=True), (obs,bank,fits,tm,fm)


def evaluate_plans(record,choices,restored):
    # Truth enters here, only after the planning function has returned.
    _,bank,fits,_,_=restored;seed=record['seed']
    case=dynamic.make_case(seed,record['family_index'],record['config'])
    if not np.array_equal(case['assets']['r'],record['truth']['r']) or not np.array_equal(case['assets']['x'],record['truth']['x']):
        raise ValueError('replayed true lines differ from saved development input')
    cache={};methods={}
    for name,power in choices.items():
        key=fixed._control_key(power)
        if key not in cache:
            operation=dynamic.operation_record(power,mobile=False,discharge=name not in ('constant','highlow'))
            if not operation['success']:value=dict(status='operation_failure',M=None,operation=operation)
            else:
                try:value=dynamic.actual_curve(case,bank,fits,record['validation_result'],power,record['config'],seed=seed+500)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    value=dict(status='fit_or_physical_failure',M=None,failure=type(exc).__name__+':'+str(exc))
                value['operation']=operation
            cache[key]=value
        methods[name]=dict(power_kw=power,**cache[key])
    original={row['name']:dict(M=row.get('M'),clean_future_M=row.get('clean_future_M')) for row in record['methods']}
    return dict(methods=methods,original_methods=original,unique_executed_curves=len(cache),
        scope='same four development histories and common saved future noise; no independent-evaluation claim')


def render(histories):
    fmt=lambda value:'—' if value is None else format(value,'.7g')
    lines=['# 冻结仿射标量L2弦MILP：4份新开发试验','',
        '完整AC分数只决定接受；上下界仅针对冻结仿射L2问题。3节点仅两段特殊分支，不增加变量。',
        '表中LB/UB属于原始MILP动作；AC分数及实际M属于接受后的计划，可能继承其他网格或基线。',
        '本轮按网格开发审计，每次MILP限2秒；累计35秒只作记录，并非与旧策略严格同预算评价。','']
    for history in histories:
        lines.extend([f"## {history['seed']}",'','|网格|新增binary|冻结LB|冻结UB|冻结gap|总弦差上界|AC接受|选中来源|AC分数|实际M|',
            '|---|---:|---:|---:|---:|---:|---|---|---:|---:|'])
        for grid,row in history['planning']['grids'].items():
            raw=row['raw'];actual=history['evaluation']['methods']['scalar_'+grid]
            lines.append('|'+grid+'|'+'|'.join([str(raw['added_binary_count']),fmt(raw['frozen_objective_lower']),
                fmt(raw['frozen_objective_upper']),fmt(raw['frozen_absolute_gap']),
                fmt(raw['table']['frozen_profile_error_upper']),str(row['accepted']),row['selected_origin'],
                fmt(row['selected_score']),fmt(actual.get('M'))])+'|')
        lines.extend(['',f"全部网格累计规划 {history['planning']['seconds']:.6g} 秒；超过35秒参考的网格数 "
            +str(sum(r['planning_budget_exceeded'] for r in history['planning']['grids'].values()))+'。'])
        lines.extend(['','|比较方法|实际M|去未来噪声M|','|---|---:|---:|'])
        for name in ('constant','highlow','square','nearest','gradac','contrast_joint','contrast_box','contrast_load'):
            row=history['evaluation']['methods'].get(name,history['evaluation']['original_methods'].get(name,{}))
            lines.append('|'+name+'|'+fmt(row.get('M'))+'|'+fmt(row.get('clean_future_M'))+'|')
        lines.append('')
    lines.extend(['以上是开发数据上的新增方法探索，4份历史仍为统计单位；没有重新校准、没有读取正式评价。',''])
    return '\n'.join(lines)


def run(directory,output):
    directory,output=Path(directory),Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'freeze.json').exists():raise ValueError('existing probe retained; use a fresh output')
    original=io.read_json(directory/'freeze.json');sources=dict(original['source_sha256'])
    for name in ('ev_scalar_distance_milp.py','test_ev_scalar_distance_milp.py','ev_scalar_distance_probe.py','ev_scalar_distance_math.md'):
        sources[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name,value in sources.items():
        path=(HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise ValueError('frozen strategy source changed: '+name)
    inputs={str(seed):dict(path=str((directory/f'trial_{seed}.json.gz').resolve()),
        sha256=hashlib.sha256((directory/f'trial_{seed}.json.gz').read_bytes()).hexdigest()) for seed in CONFIG['histories']}
    frozen=dict(config=CONFIG,source_sha256=sources,input_files=inputs,original_policy_id=original['policy_id'])
    frozen['probe_id']=io.identity(CONFIG,sources);io.write_json(output/'freeze.json',frozen)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode()
        for name in sources}
    io.write_json(output/'source_snapshot.json.gz',dict(source_sha256=sources,source_base64=snapshot),compressed=True)
    histories=[]
    for seed in CONFIG['histories']:
        record=io.read_json(inputs[str(seed)]['path']);began=perf_counter()
        with dynamic.count_physics_calls() as counts:
            choices,planning,restored=plan(planning_view(record),seed=seed+200)
            evaluation=evaluate_plans(record,choices,restored)
        result=dict(seed=seed,planning=planning,evaluation=evaluation,actual_solver_calls=counts,
            seconds=perf_counter()-began,probe_id=frozen['probe_id'])
        io.write_json(output/f'probe_{seed}.json.gz',result,compressed=True);histories.append(result)
        io.write_json(output/'checkpoint.json',dict(completed_histories=len(histories),planned_histories=4,last_seed=seed))
        print(json.dumps(dict(seed=seed,seconds=result['seconds'],
            M={name:row.get('M') for name,row in evaluation['methods'].items()})),flush=True)
    io.write_json(output/'summary.json',dict(config=CONFIG,manifest=frozen,histories=histories))
    (output/'summary.md').write_text(render(histories),encoding='utf-8')
    return histories


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.directory,args.output)


if __name__=='__main__':main()
