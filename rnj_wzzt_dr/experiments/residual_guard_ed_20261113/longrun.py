"""Reoptimize identical old QCQPs with the old verified point and valid bound."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import argparse,traceback
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..linear_ac_population_20261113.solver import Model
from .solver import solve
from .study import OLD


def select(out):
    entries={e['request']['id']:e for e in io.read(OLD/'cap30/inputs.json')};selected=[]
    def choices(identity,scenario,phase,label):
        output=[]
        for f in sorted((OLD/scenario/phase/'jobs'/identity).glob('day*/plans.json')):
            p=io.read(f)[label]
            if p['qualified'] and not p['global_optimality_verified']:output.append((f,p))
        return sorted(output,key=lambda pair:-pair[1]['relative_gap'])
    def add(pair,scenario,phase,label):
        file,plan=pair;identity=file.parent.parent.name;e=entries[identity]
        selected.append(dict(key=f'{scenario}__{phase}__{identity}__{file.parent.name}__{label}',id=identity,
            scenario=scenario,phase=phase,label=label,day=int(file.parent.name[3:]),alpha=.3 if scenario=='cap30' else .5,
            entry=e,old_plan=plan,plans=io.ref(file),planner=io.ref(file.parent/'planner.npz')))
    for identity in ('main__input121__101','main__input195__101','pressure__input026__101'):
        values=choices(identity,'cap30','confirmation','g4');require(len(values)>=3,'Need unclosed positive examples')
        retained=next((p for p in values if p[1].get('source')!='native_bounded_QCQP_incumbent'),values[-1])
        indices=[]
        for candidate in (values[0],values[len(values)//2],retained,*values):
            if candidate[0] not in [p[0] for p in indices]:indices.append(candidate)
            if len(indices)==3:break
        for pair in indices:add(pair,'cap30','confirmation','g4')
    add(choices('main__input039__101','cap50','confirmation','g4')[0],'cap50','confirmation','g4')
    multi=[]
    for identity,e in entries.items():
        if len(e['competitors'])>=2:
            v=choices(identity,'cap30','population','g2')
            if v:multi.append(v[0])
    multi.sort(key=lambda p:-p[1]['relative_gap'])
    for pair in multi[:2]:add(pair,'cap30','population','g2')
    require(len(selected)==12 and len({v['key'] for v in selected})==12,'Expected 12 distinct problems')
    sources={name:io.ref(Path(__file__).parent/name) for name in ('model.py','solver.py','longrun.py')}
    io.write(out/'protocol.json',dict(created_utc=utc(),sources=sources,selection=selected,budgets=[2.,10.],
        final_extension='at most three largest remaining relative gaps >10%, each 30 seconds',
        same_original_QCQP=True,changed_evidence_constraints=False,residual_guard=False,
        warm_start='prior verified feasible original QCQP quote; lower bound carried forward',
        timing='each additional call includes its new seed checks/model construction; old .4-second preparation is separately retained',
        native_gap=1e-5,prior_native_gap=.01,reported_global_gap=1e-5,new_independent_days=0,
        commit_at_lock=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),environment=io.environment()))
    return selected


def work(task):
    out,item,extension=task;directory=Path(out)/'jobs'/item['key'];directory.mkdir(parents=True,exist_ok=True)
    require(sha(item['planner']['path'])==item['planner']['sha256'] and sha(item['plans']['path'])==item['plans']['sha256'],'Old quote changed')
    e=item['entry'];score=io.read(e['scores']['path']);require(sha(e['scores']['path'])==e['scores']['sha256'],'Frozen candidate changed')
    H=np.array([p['H'] for p in score['parameters']])
    with np.load(item['planner']['path'],allow_pickle=False) as p:
        forecast=p['forecast'].copy();old=p['optimized_'+item['label']].copy()
    model=Model(H,e['target'],e['competitors'],e['scale'],score['sigma2']).prepare(forecast)
    gamma=2. if item['label']=='g2' else 4.;began=perf_counter()
    require(np.max(abs(old))<=item['alpha']+1e-8 and min(model.direct_gains(old))>=gamma-1e-8,'Invalid old warm point')
    if extension:
        records=io.read(directory/'results.json')
        with np.load(directory/'controls.npz',allow_pickle=False) as p:points={k:p[k].copy() for k in p.files}
        point=points['seconds10'];lower=records[-1]['lower_bound'];budgets=[30.]
    else:records=[];points={'old':old};point=old;lower=item['old_plan']['lower_bound'];budgets=[2.,10.]
    import gurobipy as gp
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    try:
        for seconds in budgets:
            result=solve(model,gamma,item['alpha'],seconds=seconds,env=env,warm=point,previous_lower=lower)
            require(result['qualified'],'A verified prior feasible quote was lost')
            point=result.pop('x');lower=result['lower_bound'];result.update(requested_seconds=seconds)
            require(result['cost']<=item['old_plan']['cost']*(1+1e-8),'Reoptimization lost old incumbent')
            records.append(result);points[f'seconds{seconds:g}']=point
    finally:env.dispose()
    io.write(directory/'results.json',records);np.savez_compressed(directory/'controls.npz',**points)
    summary=dict(key=item['key'],id=item['id'],case=e['request']['case'],alpha=item['alpha'],gamma=gamma,
        old_cost=item['old_plan']['cost'],old_lower=item['old_plan']['lower_bound'],old_gap=item['old_plan']['relative_gap'],
        old_seconds=item['old_plan']['seconds'],new_cost=records[-1]['cost'],new_lower=records[-1]['lower_bound'],
        new_gap=records[-1]['relative_gap'],global_gap_closed=records[-1]['global_optimality_verified'],
        relative_cost_improvement=(item['old_plan']['cost']-records[-1]['cost'])/item['old_plan']['cost'],
        new_call_seconds=sum(r['seconds'] for r in records),latest_worker_seconds=perf_counter()-began,
        stages=[{k:v for k,v in r.items() if k not in ('gains','guard_gains')} for r in records],status='complete')
    io.write(directory/'complete.json',summary);return summary


def run(out):
    require(not out.exists(),'Preserve earlier timing runs');out.mkdir(parents=True);selected=select(out)
    receipts=[];failures=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(work,(str(out),item,False)):item for item in selected}
        for f in as_completed(tasks):
            try:receipts.append(f.result())
            except Exception as exc:failures.append(dict(key=tasks[f]['key'],error=repr(exc),traceback=traceback.format_exc()))
            io.write(out/'checkpoint.json',dict(completed=len(receipts),failures=failures,total=12))
            print('long-QCQP',len(receipts),'/',12,'failures',len(failures),flush=True)
    extend=sorted((r for r in receipts if r['new_gap']>.1),key=lambda r:-r['new_gap'])[:3]
    mapping={i['key']:i for i in selected}
    if extend:
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(work,(str(out),mapping[r['key']],True)):r['key'] for r in extend}
            for f in as_completed(futures):
                try:
                    replacement=f.result();receipts=[replacement if r['key']==replacement['key'] else r for r in receipts]
                except Exception as exc:failures.append(dict(key=futures[f],extension=True,error=repr(exc),traceback=traceback.format_exc()))
    io.write(out/'summary.json',sorted(receipts,key=lambda r:r['key']))
    io.write(out/'complete.json',dict(status='passed' if not failures else 'complete_with_failures',problems=12,completed=len(receipts),
        extensions=len(extend),failures=failures,new_independent_days=0,source=io.ref(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
