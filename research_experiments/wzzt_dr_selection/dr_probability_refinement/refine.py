"""Bounded refinement of existing heterogeneous DR, no new identifier."""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,sys,time,shutil,zipfile,platform,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
sys.path.insert(0,str(REPO/'research_experiments/wzzt_dr_selection/dr_heterogeneous_pilot'))
from driver import read,write,ref,sha,checked,load,dump,now,config,Devices,profiles,simple_plan,observe,score,open_runtime
from planner import Probability
from planning_repair import MeanDifference
from conditional_integral import conditional_probability
from tail import TailStart


def solve(prob,old,limits):
    d=prob.devices;shape=(d.R,24);n=d.R*24
    bounds=[(-1.,0. if d.kind[r]==1 else 1.) for r in range(d.R) for _ in range(24)]
    records=[];traces=[];start=time.perf_counter();mean=MeanDifference(prob)
    def keep(x):
        q=prob.evaluate(x)
        records.append(dict(x=np.asarray(x).reshape(shape).copy(),probability=q['probability'],cost=q['cost'],
            mean_u=q['mean_u'].copy(),invalid=q['invalid'],outer_standard_error=q['outer_standard_error']))
        return records[-1]
    keep(np.zeros(shape));keep(old)
    class WallLimit(Exception):pass
    def run(label,x0,fun,jac,bnd,constraints=(),method='L-BFGS-B',transform=lambda x:x):
        t=time.perf_counter();trace=[];cap=limits[label]
        def f(x):
            if time.perf_counter()-t>cap['seconds']:raise WallLimit()
            return fun(x)
        def cb(x):
            r=keep(transform(x));trace.append(dict(probability=r['probability'],cost=r['cost']))
        options=dict(maxiter=cap['maxiter'],ftol=1e-10)
        if method=='L-BFGS-B':options.update(maxls=30,gtol=1e-7)
        try:
            res=minimize(f,np.asarray(x0).ravel(),jac=jac,bounds=bnd,method=method,constraints=constraints,callback=cb,options=options)
            cb(res.x);status=str(res.message);success=bool(res.success);last=res.x
        except WallLimit:status='time_limit';success=False;last=np.asarray(x0).ravel()
        except (ArithmeticError,FloatingPointError) as e:status=repr(e);success=False;last=np.asarray(x0).ravel()
        traces.append(dict(stage=label,status=status,solver_success=success,trace=trace,seconds=time.perf_counter()-t))
        return last
    scale=max(1.,abs(prob.packet['threshold']))
    mw=run('mean',old,lambda x:-mean.evaluate(x)['value']/scale,lambda x:-mean.evaluate(x)['gradient']/scale,bounds)
    tail=TailStart(prob)
    # Lower-tail start, optimized alpha is not an extra physical restriction.
    best=max(records,key=lambda r:(r['probability'],-r['cost']))['x']
    cv=run('tail',np.r_[best.ravel(),0.],lambda x:tail.evaluate(x)['value'],lambda x:tail.evaluate(x)['gradient'],
        bounds+[(None,None)],transform=lambda x:x[:-1])
    for label,x0 in [('prob_old',old),('prob_tail',max(records,key=lambda r:(r['probability'],-r['cost']))['x'])]:
        def ev(x):
            q=prob.evaluate(x)
            if q['invalid']:raise ArithmeticError('conditional gradient unresolved')
            return q
        run(label,x0,lambda x:-ev(x)['probability'],lambda x:-ev(x)['gradient'],bounds)
    eligible=[r for r in records if r['probability']>=.95-1e-7 and not r['invalid']]
    if eligible:
        # Limit dense SLSQP to effective hourly controls; fixed availability
        # variables cannot affect delivery and need no solver dimension.
        initial=min(eligible,key=lambda r:r['cost'])['x'].ravel()
        active=(d.scale.reshape(d.R,24,4).max(-1)>0).ravel();inds=np.flatnonzero(active)
        def expand(v):
            x=initial.copy();x[inds]=v;return x
        run('cost',initial[inds],lambda v:ev(expand(v))['cost'],lambda v:ev(expand(v))['cost_gradient'][inds],
            [bounds[j] for j in inds],constraints=[dict(type='ineq',fun=lambda v:ev(expand(v))['probability']-.95,
            jac=lambda v:ev(expand(v))['gradient'][inds])],method='SLSQP',transform=expand)
    eligible=[r for r in records if r['probability']>=.95-1e-7 and not r['invalid']]
    best=min(eligible,key=lambda r:r['cost']) if eligible else max(records,key=lambda r:(r['probability'],-r['cost']))
    return dict(best=best,qualified=bool(eligible),traces=traces,seconds=time.perf_counter()-start,
        global_optimal=False,probability_evaluations=prob.evaluations,retained_iterates=len(records))


def independent_probability(packet,dev,x,N,seed,response_seed):
    vals=[];u=dev.nominal(x)
    for batch in range((N+63)//64):
        count=min(64,N-len(vals));p=Probability(packet,dev,count,seed+batch,response_seed+batch)
        delivered=dev.project(u[None]*p.multiplier)
        a=p.a0+np.concatenate([np.einsum('nr,srt->stn',packet['effect_'+k],delivered).reshape(count,-1) for k in ['C','F']],axis=1)
        vals.extend(conditional_probability(row,p.b,p.w,packet['threshold'],coefficient_gradients=False).probability for row in a)
    values=np.array(vals)
    return dict(mean=float(values.mean()),standard_error=float(values.std(ddof=1)/np.sqrt(N)),samples=N,
        hoeffding_one_sided_95_lower=float(max(0.,values.mean()-np.sqrt(np.log(20)/(2*N)))),
        probability_model_only=True,used_for_search=False)


def prepare(previous,root):
    previous=Path(previous);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    p=read(previous/'protocol.json');cfg=read(checked(p['library']))
    p.update(id='heterogeneous_probability_refinement_v1',created_utc=now(),previous=ref(previous/'price_freeze.json'),
        planning_samples=128,planning_seed=2026110901,response_seed=2026111901,
        qualification_samples=4096,qualification_seed=2026112901,qualification_response_seed=2026113901,
        primary_policies=['zero','rotation','old_diagnostic','candidate'],expected_records=576,
        paths=[dict(day=j,profile_index=j%3,replicate=1130000+173*j,response_seed=2026121000+j) for j in range(24)],
        solver_limits=dict(mean=dict(maxiter=100,seconds=150),tail=dict(maxiter=100,seconds=180),
            prob_old=dict(maxiter=100,seconds=150),prob_tail=dict(maxiter=100,seconds=150),cost=dict(maxiter=80,seconds=240)),
        scientific_changes=['tail-directed initialization','bounded extra local iterations','128 planning outer points',
            '4096 independent probability checks','explicit feasible versus diagnostic labels'],
        qualification='planning target 0.95; independent estimate and lower bound reported without feeding back into search',
        actual_scope='same three cases and resource inventory; 24 unseen paired background paths per case; frozen candidate evaluation only',
        selection_scope='full historical frozen C/F and all R/X; original weighted validation and frozen output, no re-fit/new W search',
        stop_rule='one locked refinement, retain unresolved; no retuning after independent results',
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__))
    # Public postprocessing is not a scientific experiment dependency.
    for f in [Path(__file__),Path(__file__).with_name('tail.py')]:p['source_sha256'][str(f.resolve())]=sha(f)
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(p['source_sha256']):z.write(f,f'{i:03d}_{Path(f).name}')
    for c in p['cases']:
        folder=root/c['name'];folder.mkdir()
        for fn in ['case_protocol.json','pool.json','problem.pkl']:
            shutil.copy2(previous/c['name']/fn,folder/fn)
        write(folder/'inputs.json',{fn:ref(folder/fn) for fn in ['case_protocol.json','pool.json','problem.pkl']})
    write(root/'run_lock.json',dict(status='prepared',pid=None))


def optimize(previous,root):
    root,p,cfg,alloc=config(root);previous=Path(previous);rows=[];allplans={}
    write(root/'run_lock.json',dict(status='optimizing',pid=os.getpid()))
    for ci,c in enumerate(p['cases']):
        folder=root/c['name'];receipt=folder/'price_receipt.json'
        if receipt.exists():
            saved=read(receipt);checked(saved['plans']);rows.append(saved['summary']);allplans[c['name']]=saved;continue
        for r in read(folder/'inputs.json').values():checked(r)
        packet=load(folder/'problem.pkl');dev=Devices(alloc[c['case']],cfg,profiles)
        old=load(previous/c['name']/'solver_private.pkl')['best']
        prob=Probability(packet,dev,p['planning_samples'],p['planning_seed']+ci*1000,p['response_seed']+ci*1000)
        if packet['has_dispute']:
            sol=solve(prob,old['x'],p['solver_limits'])
        else:
            q=prob.evaluate(np.zeros((dev.R,24)))
            sol=dict(best=dict(x=np.zeros((dev.R,24)),mean_u=q['mean_u'],cost=0.,probability=q['probability']),
                qualified=False,traces=[],seconds=0.,global_optimal=False,retained_iterates=1,probability_evaluations=1)
        best=sol['best'];dump(folder/'solver_private.pkl',sol)
        qual=independent_probability(packet,dev,best['x'],p['qualification_samples'],p['qualification_seed']+ci*1000,p['qualification_response_seed']+ci*1000)
        fee=best['cost'];rotation=simple_plan(dev,'rotation');maximum=dev.fees(rotation)
        rotation*=min(1.,np.sqrt(fee/maximum)) if maximum else 0.
        plans={'zero':dict(mean_u=np.zeros_like(best['mean_u']),cost=0.),'candidate':dict(mean_u=best['mean_u'],cost=fee),
            'old_diagnostic':dict(mean_u=old['mean_u'],cost=old['cost']),'rotation':dict(mean_u=rotation,cost=dev.fees(rotation))}
        for name,item in plans.items():
            item['nominal_audit']=dev.audit(item['mean_u'])
            if not item['nominal_audit']['passed']:raise ArithmeticError('infeasible nominal '+name)
        dump(folder/'plans.pkl',plans)
        status='no_historical_dispute_no_purchase' if not packet['has_dispute'] else ('planning_feasible' if sol['qualified'] else 'feasibility_not_found_diagnostic_only')
        s=dict(condition=c['name'],status=status,has_dispute=packet['has_dispute'],planning_probability=best['probability'],
            independent_probability=qual,planning_target_met=sol['qualified'],verified_estimate_at_least_target=qual['mean']>=.95,
            candidate_cost=fee,old_cost=old['cost'],rotation_cost=plans['rotation']['cost'],threshold_micro=packet['threshold'],
            global_optimal=False,solve_seconds=sol['seconds'],statuses=[t['status'] for t in sol['traces']],
            nominal_audits={name:v['nominal_audit'] for name,v in plans.items()})
        saved=dict(summary=s,plans=ref(folder/'plans.pkl'),solution=ref(folder/'solver_private.pkl'))
        write(receipt,saved);rows.append(s);allplans[c['name']]=saved
        print('PLANNED',c['name'],status,best['probability'],qual['mean'],fee,flush=True)
    write(root/'planning_summary.json',dict(rows=rows));write(root/'price_freeze.json',dict(protocol=ref(root/'protocol.json'),cases=allplans,truth_read_for_planning=False,status='frozen'))
    write(root/'run_lock.json',dict(status='prices_frozen',pid=None))


def evaluate(root):
    root,p,cfg,alloc=config(root);ledger=read(root/'observations_frozen.json');checked(ledger['price_freeze']);rows=[]
    for old in ledger['rows']:
        row={k:v for k,v in old.items() if k not in ['traceback','packet']}
        if old['status']=='complete':
            result=load(checked(old['packet']));pool=read(root/old['condition']/'pool.json')
            s=score(pool,result['scene'],history_gate=True)
            chosen=next(r for r in s['rules'] if r['new_day_weight']==2 and r['new_day_fraction']==.03)
            write(root/'scores'/(old['record_id']+'.json'),s)
            loss={r['candidate_id']:r['new_validation_mae'] for r in s['loss_rows']}
            row.update(status='predicted',selected=chosen['selected_candidate_id'],coarse_excluded='coarse' not in chosen['eligible_candidate_ids'],
                D_micro=1e6*(loss['coarse']-1.03*loss['refined']),margin_micro=1e6*chosen['exit_margins']['coarse'])
        rows.append(row)
    write(root/'predictions_frozen.json',dict(rows=rows,truth_read=False))
    errors={}
    for c in p['cases']:
        rt=open_runtime(c['source_root'],c['source_metadata'],repo_root=REPO,protocol_ref=ref(root/'protocol.json'),labels=c['labels'],case=c['case'],v_sigma=c['voltage_sigma'],pq_sigma=c['pq_sigma'])
        truth=rt['legacy'].evaluate_supports([],rt['context'].net);truthset={tuple(sorted(b)) for b in truth['missed_blocks']}
        write(root/c['name']/'truth_private.json',truth);errs={}
        for m in read(root/c['name']/'pool.json')['models']:
            raw={tuple(sorted(c['labels'][i] for i in b)) for b in m['point']['supports']}
            final={tuple(sorted(b)) for b in m['frozen_output_support_labels']}
            errs[m['candidate_id']]=dict(raw_errors=len(raw^truthset),errors=len(final^truthset),missing=len(truthset-final),extra=len(final-truthset))
        errors[c['name']]=errs
    for row in rows:
        if row['status']=='predicted':row.update(status='complete',**errors[row['condition']][row['selected']])
    tables=[];pairs=[];N=len(p['paths'])
    for c in p['cases']:
        for law in p['laws']:
            for name in p['primary_policies']:
                group=[r for r in rows if r['condition']==c['name'] and r['law']==law and r['policy']==name];ok=[r for r in group if r['status']=='complete']
                table=dict(condition=c['name'],law=law,policy=name,denominator=N,complete=len(ok),unresolved=N-len(ok),
                    errors=sum(r['errors'] for r in ok),recoveries=sum(r['errors']==0 for r in ok),fee=group[0]['fee'],
                    mean_D=float(np.mean([r['D_micro'] for r in ok])),min_D=float(min(r['D_micro'] for r in ok)),
                    exclusions=sum(r['coarse_excluded'] for r in ok),mean_abs_kwh=float(np.mean([r['physical']['resource_audit']['absolute_DR_kwh'] for r in ok])),
                    mean_net_kwh=float(np.mean([r['physical']['resource_audit']['net_DR_kwh'] for r in ok])),
                    raw_final_different=sum(r['raw_errors']!=r['errors'] for r in ok))
                tables.append(table)
            for other in ['zero','rotation','old_diagnostic']:
                changes=[]
                for day in range(N):
                    rs=[r for r in rows if r['condition']==c['name'] and r['law']==law and r['day']==day]
                    a=next(r for r in rs if r['policy']=='candidate');b=next(r for r in rs if r['policy']==other)
                    if a['status']==b['status']=='complete':changes.append(a['errors']-b['errors'])
                pairs.append(dict(condition=c['name'],law=law,reference=other,better=sum(v<0 for v in changes),same=sum(v==0 for v in changes),worse=sum(v>0 for v in changes),unresolved=N-len(changes)))
    summary=dict(tables=tables,pairs=pairs,candidate_errors=errors,records=len(rows),complete=sum(r['status']=='complete' for r in rows),
        failed_unresolved=sum(r['status']!='complete' for r in rows),independent_backgrounds=N*len(p['cases']),paired_response_laws=len(p['laws']),
        AC_converged_points=sum(r.get('physical',{}).get('converged',0) for r in rows),
        resource_violations=sum(not r.get('physical',{}).get('resource_audit',{}).get('passed',True) for r in rows),
        voltage_violations=sum(r.get('physical',{}).get('voltage_violation_cells',0)>0 for r in rows),
        parameter_fits=0,new_W_search=0,scope='frozen full historical RNJ+WZZT candidates, validation and output only')
    write(root/'scored_private.json',dict(rows=rows));write(root/'summary.json',summary)
    write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json')));write(root/'run_lock.json',dict(status='complete',pid=None))
    print('COMPLETE',len(rows),summary['failed_unresolved'],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','optimize','observe','evaluate']);a.add_argument('--previous',type=Path);a.add_argument('--output',required=True,type=Path);args=a.parse_args()
    try:
        if args.phase in ['prepare','optimize']:globals()[args.phase](args.previous,args.output)
        else:globals()[args.phase](args.output)
    except Exception as e:
        if args.output.exists():write(args.output/f'failure_{args.phase}_{int(time.time())}.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
