"""Bounded initialization/feasibility repair before any physical future is read.

Expected original L1 difference is used only to initialize the exact event
probability search. Final target and validation rule are unchanged.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,sys,time,shutil,zipfile
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize
from driver import read,write,sha,ref,checked,load,dump,now,config,Devices,profiles,simple_plan
from planner import Probability


class MeanDifference:
    """Exact inner Gaussian mean of D=L_C-1.03 L_F, fixed outer integration."""
    def __init__(self,probability):
        self.p=probability;self.last_x=None

    def evaluate(self,x):
        p=self.p;d=p.devices;x=np.asarray(x).reshape(d.R,24)
        if self.last_x is not None and np.array_equal(x,self.last_x):return self.last
        u,ncache=d.nominal(x,cache=True)
        delivered,ecache=d.project(u[None]*p.multiplier,cache=True)
        ec=np.einsum('nr,srt->stn',p.packet['effect_C'],delivered).reshape(p.samples,-1)
        ef=np.einsum('nr,srt->stn',p.packet['effect_F'],delivered).reshape(p.samples,-1)
        a=p.a0+np.concatenate([ec,ef],axis=1);sigma=np.abs(p.b)
        z=np.divide(a,sigma,out=np.zeros_like(a),where=sigma>0)
        slope=np.where(sigma>0,2*ndtr(z)-1,np.sign(a))
        folded=np.where(sigma>0,2*sigma*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+a*slope,np.abs(a))
        value=float(np.mean(folded@p.w));g=slope*p.w
        gc=np.einsum('stn,nr->srt',g[:,:p.m].reshape(p.samples,96,-1),p.packet['effect_C'])
        gf=np.einsum('stn,nr->srt',g[:,p.m:].reshape(p.samples,96,-1),p.packet['effect_F'])
        gnom=np.mean(d.vjp(gc+gf,ecache)*p.multiplier,axis=0)
        gradient=d.hourly_vjp(gnom,ncache).ravel()
        self.last_x=x.copy();self.last=dict(value=value,gradient=gradient)
        return self.last


def repaired_solve(prob,old_solution,seed):
    """One mean-D warm start, two probability searches, one cost minimization."""
    d=prob.devices;shape=(d.R,24);bounds=[(-1.,0. if d.kind[r]==1 else 1.) for r in range(d.R) for _ in range(24)]
    rng=np.random.default_rng(seed);small=.03*rng.normal(size=shape)
    small[d.kind==1]=-np.abs(small[d.kind==1])
    old=np.asarray(old_solution['best']['x']).reshape(shape)
    mean=MeanDifference(prob);records=[];traces=[];begun=time.perf_counter()
    def keep(x):
        q=prob.evaluate(x);m=mean.evaluate(x)
        row=dict(x=np.asarray(x).reshape(shape).copy(),probability=q['probability'],cost=q['cost'],mean_u=q['mean_u'].copy(),invalid=q['invalid'],mean_D=m['value'])
        records.append(row);return row
    keep(np.zeros(shape));keep(old)
    class WallLimit(Exception):pass
    # A flat event probability is a numerical obstacle, not evidence that no
    # response can help. The mean of the SAME loss difference seeds the search.
    started=time.perf_counter();mw=[];scale=max(1.,abs(prob.packet['threshold']))
    def wm(x):
        if time.perf_counter()-started>60:raise WallLimit()
        return mean.evaluate(x)
    def warm_callback(x):
        r=wm(x);mw.append(dict(x=np.asarray(x).copy(),mean_D=r['value']))
    try:
        res=minimize(lambda x:-wm(x)['value']/scale,small.ravel(),jac=lambda x:-wm(x)['gradient']/scale,
            method='L-BFGS-B',bounds=bounds,callback=warm_callback,options=dict(maxiter=20,maxls=12,ftol=1e-9))
        warm_callback(res.x);status=str(res.message);success=bool(res.success)
    except WallLimit:status='time_limit';success=False
    warm=max(mw,key=lambda r:r['mean_D'])['x'] if mw else small.ravel()
    traces.append(dict(stage='mean_D_initialization_only',status=status,solver_success=success,iterations=len(mw),
        start_mean_D=mean.evaluate(small)['value'],best_mean_D=mean.evaluate(warm)['value'],elapsed=time.perf_counter()-started))
    keep(warm)
    for label,x0 in [('old_best',old),('mean_D_warm',warm)]:
        started=time.perf_counter();trace=[]
        def ev(x):
            if time.perf_counter()-started>60:raise WallLimit()
            q=prob.evaluate(x)
            if q['invalid']:raise ArithmeticError('conditional derivative unresolved')
            return q
        def cb(x):
            r=keep(x);trace.append(dict(probability=r['probability'],cost=r['cost'],mean_D=r['mean_D']))
        try:
            res=minimize(lambda x:-ev(x)['probability'],np.asarray(x0).ravel(),jac=lambda x:-ev(x)['gradient'],
                method='L-BFGS-B',bounds=bounds,callback=cb,options=dict(maxiter=20,maxls=12,ftol=1e-10,gtol=1e-8))
            cb(res.x);status=str(res.message);success=bool(res.success)
        except WallLimit:status='time_limit';success=False
        except ArithmeticError as e:status=str(e);success=False
        traces.append(dict(stage='event_feasibility_'+label,status=status,solver_success=success,trace=trace,elapsed=time.perf_counter()-started))
    qualified=[r for r in records if r['probability']>=.95-1e-7 and not r['invalid']]
    if qualified:
        start=min(qualified,key=lambda r:r['cost'])['x'];started=time.perf_counter();trace=[]
        def ev(x):
            if time.perf_counter()-started>90:raise WallLimit()
            q=prob.evaluate(x)
            if q['invalid']:raise ArithmeticError('conditional derivative unresolved')
            return q
        def cb(x):
            r=keep(x);trace.append(dict(probability=r['probability'],cost=r['cost'],mean_D=r['mean_D']))
        try:
            res=minimize(lambda x:ev(x)['cost'],start.ravel(),jac=lambda x:ev(x)['cost_gradient'],method='SLSQP',bounds=bounds,
                constraints=[dict(type='ineq',fun=lambda x:ev(x)['probability']-.95,jac=lambda x:ev(x)['gradient'])],
                callback=cb,options=dict(maxiter=25,ftol=1e-7))
            cb(res.x);status=str(res.message);success=bool(res.success)
        except WallLimit:status='time_limit';success=False
        except ArithmeticError as e:status=str(e);success=False
        traces.append(dict(stage='hard_probability_min_cost',status=status,solver_success=success,trace=trace,elapsed=time.perf_counter()-started))
    qualified=[r for r in records if r['probability']>=.95-1e-7 and not r['invalid']]
    best=min(qualified,key=lambda r:r['cost']) if qualified else max(records,key=lambda r:(r['probability'],-r['cost']))
    return dict(best=best,qualified=bool(qualified),traces=traces,elapsed=time.perf_counter()-begun,
        evaluations=prob.evaluations,global_optimum=False,zero_probability=records[0]['probability'])


def prepare_repair(previous,root):
    previous=Path(previous);root=Path(root)
    old=read(previous/'price_freeze.json');checked(old['protocol'])
    if (previous/'observations_frozen.json').exists() or (previous/'actual').exists():raise ValueError('future already observed')
    root.mkdir(parents=True,exist_ok=False);p=read(previous/'protocol.json')
    p.update(id='heterogeneous_resource_pilot_v2',created_utc=now(),planning_attempt_1=ref(previous/'planning_summary.json'),
        planning_repair=dict(reason='flat zero-probability gradient at small starts and failed first feasibility search',
            mean_D_for_initialization_only=dict(maxiter=20,seconds=60),event_feasibility=dict(starts=2,maxiter=20,seconds=60),
            min_cost=dict(maxiter=25,seconds=90),target=.95,qualification_not_used_for_selection=True,
            unchanged=['resource domain','case set','history','loss event','95% target','price law','actual path seeds','selector']))
    # Audit-only scripts are excluded from the scientific source closure.
    p['source_sha256'].update({str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py') if f.name!='post_audit.py'})
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(p['source_sha256']):z.write(f,f'{i:03d}_{Path(f).name}')
    shutil.copy2(previous/'screening.json',root/'screening.json')
    for c in p['cases']:
        folder=root/c['name'];folder.mkdir()
        for filename in ['case_protocol.json','pool.json','problem.pkl']:
            shutil.copy2(previous/c['name']/filename,folder/filename)
    write(root/'run_lock.json',dict(status='prepared',pid=None))


def run(previous,root):
    root,p,cfg,alloc=config(root);previous=Path(previous)
    write(root/'run_lock.json',dict(status='repair_optimizing',pid=os.getpid()));allplans={};rows=[]
    for i,c in enumerate(p['cases']):
        folder=root/c['name'];receipt=folder/'price_receipt.json'
        if receipt.exists():
            saved=read(receipt);checked(saved['plans']);allplans[c['name']]=saved;rows.append(saved['summary']);continue
        packet=load(folder/'problem.pkl');dev=Devices(alloc[c['case']],cfg,profiles)
        prob=Probability(packet,dev,p['planning_samples'],p['planning_seed']+i,p['response_seed']+i)
        sol=repaired_solve(prob,load(previous/c['name']/'solver_private.pkl'),p['optimizer_seed']+i+30)
        dump(folder/'solver_private.pkl',sol)
        best=sol['best'];qualification=Probability(packet,dev,p['qualification_samples'],p['qualification_seed']+i,p['qualification_response_seed']+i)
        qual=qualification.evaluate(best['x'])['probability'];fee=best['cost']
        plans={'zero':dict(mean_u=np.zeros_like(best['mean_u']),cost=0.),'optimized':dict(mean_u=best['mean_u'],cost=fee)}
        for name in ['rotation','cost_first']:
            full=simple_plan(dev,name);maximum=dev.fees(full);alpha=min(1.,np.sqrt(fee/maximum)) if maximum>0 else 0.
            value=alpha*full;plans[name]=dict(mean_u=value,cost=dev.fees(value),requested_cost=fee,scale=alpha,full_cost=maximum,cost_matched=abs(dev.fees(value)-fee)<1e-7)
        for name,item in plans.items():
            item['nominal_audit']=dev.audit(item['mean_u'])
            if not item['nominal_audit']['passed']:raise ArithmeticError('infeasible nominal '+name)
        dump(folder/'plans.pkl',plans)
        s=dict(condition=c['name'],planning_probability=best['probability'],independent_working_probability=qual,
            optimization_target_met=sol['qualified'],optimized_cost=fee,optimizer_status=[t['status'] for t in sol['traces']],
            mean_D_micro=best['mean_D'],global_optimal=False,has_dispute=packet['has_dispute'],
            policy_costs={name:v['cost'] for name,v in plans.items()},policy_nominal_audits={name:v['nominal_audit'] for name,v in plans.items()})
        saved=dict(summary=s,plans=ref(folder/'plans.pkl'),solver_trace=sol['traces']);write(receipt,saved);allplans[c['name']]=saved;rows.append(s)
        print('repaired',json_safe_summary(s),flush=True)
    write(root/'price_freeze.json',dict(status='frozen',protocol=ref(root/'protocol.json'),cases=allplans,truth_read=False))
    write(root/'planning_summary.json',dict(rows=rows));write(root/'run_lock.json',dict(status='prices_frozen',pid=None))
    write(previous/'superseded_planning.json',dict(status='planning_only_retained',replacement=ref(root/'price_freeze.json'),new_physical_days=0))


def json_safe_summary(s):
    import json
    return json.dumps({k:v for k,v in s.items() if k!='policy_nominal_audits'},ensure_ascii=False)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','run']);a.add_argument('--previous',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
    args=a.parse_args()
    if args.phase=='prepare':prepare_repair(args.previous,args.output)
    else:run(args.previous,args.output)
