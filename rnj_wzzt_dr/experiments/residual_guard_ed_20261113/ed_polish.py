"""Feasible convex ED steps for the same nominal information QCQP."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import argparse,traceback
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..linear_ac_population_20261113.solver import Model
from .model import Evidence
from .prices import economic_dispatch


class Tangent:
    def __init__(self,evidence,x,gamma):
        self.Q=np.zeros_like(evidence.Q);self.a=np.array([x@Q+a for Q,a in zip(evidence.Q,evidence.a,strict=True)])
        self.rhs=evidence.thresholds(gamma)+np.einsum('hi,kij,hj->k',x,evidence.Q,x)
    def thresholds(self,gamma):return self.rhs
    def direct(self,x):return 2*np.einsum('khi,hi->k',self.a,x)


def worker(task):
    root,item,original,out=task;e=item['entry'];score=io.read(e['scores']['path']);H=np.array([p['H'] for p in score['parameters']])
    with np.load(item['planner']['path'],allow_pickle=False) as p:F=p['forecast'].copy()
    with np.load(original['private_point']['path'],allow_pickle=False) as p:
        point=dict(cost_yuan=original['cost_yuan'],x=p['joint_x'].copy(),
            dispatch={k:p['joint_'+k].copy() for k in ('dg','shift','shed','buy','sell')})
    base=Model(H,e['target'],e['competitors'],e['scale'],score['sigma2']).prepare(F);evidence=Evidence(base)
    gamma=4.;began=perf_counter();trace=[];old_cost=point['cost_yuan'];offset=original['free_ED']['cost_yuan']
    for iteration in range(8):
        if perf_counter()-began>2.:break
        tangent=Tangent(evidence,point['x'],gamma)
        candidate=economic_dispatch(F,e['scale'],item['base_mw'],item['alpha'],fixed=None,evidence=tangent,gamma=gamma,
            seconds=min(.4,max(.001,2.-(perf_counter()-began))),allow_shed=False,objective_offset=offset,warm=point)
        if not candidate['feasible']:break
        require(min(evidence.direct(candidate['x'])-evidence.thresholds(gamma))>=-1e-7,'Tangent ED step violated original evidence')
        require(candidate['cost_yuan']<=point['cost_yuan']+1e-6,'Tangent ED cost increased')
        improvement=point['cost_yuan']-candidate['cost_yuan'];trace.append(dict(iteration=iteration,cost_yuan=candidate['cost_yuan'],seconds=candidate['seconds']))
        point=candidate
        if improvement<1e-7:break
    lower=original['lower_bound_yuan'];premium=point['cost_yuan']-offset;premium_lower=max(0.,lower-offset)
    result=dict(id=item['id'],scenario=item['scenario'],old_cost_yuan=old_cost,new_cost_yuan=point['cost_yuan'],
        old_premium_yuan=original['information_premium_yuan'],premium_yuan=premium,premium_lower_bound_yuan=premium_lower,
        premium_relative_gap=max(0.,premium-premium_lower)/max(abs(premium),1e-12),
        iterations=len(trace),seconds=perf_counter()-began,trace=trace,components=point.get('components',original['components']),
        inner_QP_bound_used=False,local_only=True,source=io.ref(__file__))
    directory=Path(out)/'points';directory.mkdir(parents=True,exist_ok=True);file=directory/f"{item['scenario']}__{item['id']}.npz"
    np.savez_compressed(file,x=point['x'],**point['dispatch']);result['private_point']=io.ref(file);return result


def run(previous,out):
    require(not out.exists(),'Preserve earlier ED improvement');out.mkdir(parents=True);cfg=io.read(previous/'protocol.json')
    records=io.read(previous/'joint_results.json');selected=[r for r in records if r['feasible']]
    lookup={(t['id'],t['scenario']):t for t in cfg['tasks'] if t['phase']=='confirmation' and t['day']==0}
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),previous=io.ref(previous/'protocol.json'),
        selected=[dict(id=r['id'],scenario=r['scenario']) for r in selected],seconds_per_problem=2.,max_iterations=8,
        local_only=True,inner_bounds_not_used=True,new_independent_days=0))
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(worker,(str(previous),lookup[r['id'],r['scenario']],r,str(out))):r for r in selected}
        for f in as_completed(futures):
            try:results.append(f.result())
            except Exception as exc:failures.append(dict(id=futures[f]['id'],error=repr(exc),traceback=traceback.format_exc()))
    io.write(out/'summary.json',results);io.write(out/'complete.json',dict(status='passed' if not failures else 'complete_with_failures',
        completed=len(results),planned=len(selected),failures=failures,new_independent_days=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.previous.resolve(),a.output.resolve())
