"""One locked retry of an adapter-only failed case; original result retained.

The solver, seeds, model, domain and limits are unchanged. Raw solve output
is saved before any downstream adapter and a bounded numerical energy fix
is independently rechecked. Completed cases are never solved again.
"""
from __future__ import annotations
import argparse,pickle,traceback
from pathlib import Path
import numpy as np
from .runner import read,write,ref,checked,sha,now,warm_curves,curve_from_plan
from .day_solver import solve
from .day_model import scenario_summary,resource_audit
from .repair_tools import polish_curve
from .adapter import build_plan


def run(original,output,condition):
    old=Path(original).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    p=read(old/'protocol.json');c=next(c for c in p['cases'] if c['name']==condition)
    previous=old/'cases'/condition/'result.json';failure=read(previous)
    if failure['status']!='failed' or 'daily incremental energy' not in failure.get('message',''):
        raise ValueError('only the declared adapter energy-equality failure may be retried')
    frozen=read(old/'cases'/condition/'problem_frozen.json');problem_path=checked(frozen['problem'])
    for name,digest in p['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('original source closure drift: '+name)
    additions=[Path(__file__).resolve(),Path(__file__).with_name('repair_tools.py')]
    q=dict(status='frozen',created_utc=now(),original_protocol=ref(old/'protocol.json'),original_failure=ref(previous),
        original_problem=ref(problem_path),condition=condition,model_unchanged=True,
        seeds_unchanged=True,solver_limits_unchanged=True,planned_retries=1,
        source_sha256={**p['source_sha256'],**{str(x):sha(x) for x in additions}},
        correction='maximum1e-7kW equality restoration; original resource/fee/target re-evaluation; raw solver retained')
    write(out/'protocol.json',q)
    try:
        with problem_path.open('rb') as f:data=pickle.load(f)
        problem=data['problem'];scenarios=data['scenarios']
        pool=read(checked(c['pool']));plans={k:read(checked(v)) for k,v in c['old_plans'].items()};resources=plans['optimized']['resources']
        r=solve(problem,scenarios,seconds=p['planning_seconds'],cost_seconds=p['cost_seconds'],
            warm_curves=warm_curves(problem,plans,resources),log_path=out/'solver.log')
        write(out/'raw_solver.json',r)
        r=polish_curve(problem,scenarios,r)
        adapted=build_plan(resources,np.asarray(r['mean_u']),np.asarray(r['prices']),node_labels=pool['labels'],
            budget=problem['budget'],policy='fixed24h_minfee',metadata=dict(scope='conditional model pilot numerical repair; not deployed',case=c['case']))
        write(out/'plan.json',adapted['plan'])
        r.update(condition=condition,plan=ref(out/'plan.json'),generic_adapter_audit=adapted['audit'],
            source_protocol=ref(out/'protocol.json'),forecast_scope=problem['forecast_scope'])
        r['baselines']={k:dict(training=scenario_summary(problem,scenarios,curve_from_plan(v,resources)),
            resource_audit=resource_audit(problem,curve_from_plan(v,resources))) for k,v in plans.items()}
        write(out/'result.json',r)
        write(out/'ended.json',dict(status='complete',result=ref(out/'result.json'),raw_solver=ref(out/'raw_solver.json')))
        print(dict(stage='adapter_retry_complete',condition=condition,status=r['status'],
            passed=r['training']['passed'],slack=r['training']['feasibility_slack_micro'],
            expected_payment=r['resource_audit']['expected_payment'],maximum_polish_kw=r['numerical_polish']['maximum_change_kw']),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(status='failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',default='paper15_b2_v0005');a=p.parse_args();run(a.original,a.output,a.condition)
