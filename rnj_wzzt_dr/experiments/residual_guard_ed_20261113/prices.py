"""Fixed-commitment active-power ED, using illustrative yuan/MWh offers.

Only forecasts, historical power scales and nominal base MVA enter. No branch,
true impedance, future voltage or actual future load enters this calculation.
"""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,traceback,shutil
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..linear_ac_population_20261113.solver import Model,finite_attribute
from .model import Evidence
from .study import SCENARIOS,CONFIRM

PRICE=np.repeat([300.]*7+[500.]*10+[800.]*5+[500.]*2,4)


def economic_dispatch(forecast,scale,base_mw,alpha,*,fixed=None,evidence=None,gamma=2.,seconds=3.,allow_shed=False,env=None,objective_offset=0.,warm=None):
    began=perf_counter();forecast=np.asarray(forecast,float);scale=np.asarray(scale,float);n=len(scale)
    if fixed is not None and np.max(abs(fixed))>alpha+1e-8:
        return dict(status='requested_action_outside_capacity',feasible=False,infeasibility_proved=True)
    mw=scale*base_mw;dt=.25;fuel=500*np.array([.7,.9,1.1])[np.arange(n)%3]
    g0=.33;room=.27;ramp=.15;shift_cap=.15;curvature=.4*500/(.6*mw)
    gross=forecast[:,:n]/scale[None]+g0;shed_cap=.05*np.maximum(gross,0.) if allow_shed else np.zeros_like(gross)
    money_scale=max(1.,float(24*500*sum(mw)))
    import gurobipy as gp
    owns=env is None
    if owns:env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    try:
        with gp.Model('fixed_commitment_flexibility_ED',env=env) as m:
            for k,v in dict(OutputFlag=0,Threads=1,DualReductions=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,NumericFocus=1,MIPGap=1e-5,
                NonConvex=2 if evidence is not None else 0).items():m.setParam(k,v)
            x=m.addMVar((24,n),lb=-alpha,ub=alpha)
            if fixed is not None:x.LB=fixed;x.UB=fixed
            dg=m.addMVar((96,n),lb=-room,ub=room);shift=m.addMVar((96,n),lb=-shift_cap,ub=shift_cap)
            shed=m.addMVar((96,n),lb=0.,ub=shed_cap);absolute=m.addMVar((96,n),lb=0.)
            buy=m.addMVar(96,lb=0.);sell=m.addMVar(96,lb=0.)
            m.addConstr(absolute>=shift);m.addConstr(absolute>=-shift);m.addConstr(shift.sum(axis=0)==0.)
            m.addConstr(dg[1:]-dg[:-1]<=ramp);m.addConstr(dg[:-1]-dg[1:]<=ramp)
            m.addConstr(dg[0]<=ramp);m.addConstr(dg[0]>=-ramp)
            for t in range(96):
                m.addConstr(shift[t]-shed[t]-dg[t]==x[t//4])
                m.addConstr(buy[t]-sell[t]==float(base_mw*forecast[t,:n].sum())+(mw*x[t//4]).sum())
            energy=dt*(PRICE@buy-(.9*PRICE)@sell)
            generation=dt*(((dg+g0)*mw*fuel).sum()+.5*(curvature*mw*mw*dg*dg).sum())
            transfer=dt*100*(mw*absolute).sum();loss=dt*10000*(mw*shed).sum()
            objective=energy+generation+transfer+loss;m.setObjective((objective-objective_offset)/money_scale)
            m.Params.MIPGapAbs=1e-9/money_scale
            if evidence is not None:
                for Q,a,rhs in zip(evidence.Q,evidence.a,evidence.thresholds(gamma),strict=True):
                    q=sum(x[h]@Q@x[h] for h in range(24))
                    m.addConstr((q+2*(a*x).sum())/gamma>=rhs/gamma)
            if warm is not None:
                x.Start=warm['x'];dg.Start=warm['dispatch']['dg'];shift.Start=warm['dispatch']['shift'];shed.Start=warm['dispatch']['shed']
                absolute.Start=abs(warm['dispatch']['shift']);buy.Start=warm['dispatch']['buy'];sell.Start=warm['dispatch']['sell']
            m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began));m.optimize()
            status=int(m.Status);bound=finite_attribute(m,'ObjBound');value=None if not m.SolCount else float(m.ObjVal*money_scale+objective_offset)
            if not m.SolCount and warm is None:return dict(status='infeasible' if status==3 else 'unresolved',native_status=status,
                feasible=False,infeasibility_proved=status==3,seconds=perf_counter()-began,cost_yuan=None,
                lower_bound_yuan=None if bound is None else bound*money_scale+objective_offset)
            if warm is not None and (value is None or warm['cost_yuan']<=value):
                xx=warm['x'];gg=warm['dispatch']['dg'];ss=warm['dispatch']['shift'];ll=warm['dispatch']['shed']
                bb=warm['dispatch']['buy'];ee=warm['dispatch']['sell'];aa=abs(ss);value=warm['cost_yuan'];source='verified_ED_initial_point_retained'
            else:
                xx=np.array(x.X);gg=np.array(dg.X);ss=np.array(shift.X);ll=np.array(shed.X)
                bb=np.array(buy.X);ee=np.array(sell.X);aa=np.array(absolute.X);source='native_ED_QCQP_incumbent'
            net=base_mw*forecast[:,:n]+np.repeat(xx,4,axis=0)*mw
            components=dict(energy_bill_yuan=float(dt*np.sum(PRICE*bb-.9*PRICE*ee)),
                generation_bid_yuan=float(dt*np.sum(mw*fuel*(gg+g0)+.5*curvature*mw*mw*gg*gg)),
                transfer_compensation_yuan=float(dt*100*np.sum(abs(ss)*mw)),
                shedding_loss_yuan=float(dt*10000*np.sum(ll*mw)),
                bought_MWh=float(dt*bb.sum()),sold_MWh=float(dt*ee.sum()),
                transferred_absolute_MWh=float(dt*np.sum(abs(ss)*mw)),shed_MWh=float(dt*np.sum(ll*mw)))
            cost=sum(v for k,v in components.items() if k.endswith('_yuan'))
            balance=max(float(np.max(abs(ss-ll-gg-np.repeat(xx,4,axis=0)))),float(np.max(abs(bb-ee-net.sum(1)))))
            violations=max(0.,float(np.max(abs(xx)))-alpha,float(np.max(abs(gg)))-room,
                float(np.max(abs(np.diff(gg,axis=0))))-ramp,float(np.max(abs(gg[0])))-ramp,
                float(np.max(abs(ss)))-shift_cap,float(-ll.min()),float(np.max(ll-shed_cap)))
            rebound=float(np.max(abs(ss.sum(0))))
            require(balance<1e-7 and violations<1e-7 and rebound<1e-7,'ED balance/capacity/ramp/energy violation')
            # A time-limited nonconvex incumbent need not have tightened its
            # absolute-shift epigraph. Setting it to |shift| preserves every
            # physical/evidence constraint and yields a cheaper feasible quote.
            epigraph_extra=float(dt*100*np.sum(mw*(aa-abs(ss))))
            require(epigraph_extra>=-1e-6*max(1.,cost),'Invalid absolute-shift epigraph')
            require(abs(cost+epigraph_extra-value)<1e-6*max(1.,cost),'ED component/solver objective mismatch')
            if evidence is not None:require(min(evidence.direct(xx)-evidence.thresholds(gamma))>=-1e-7,'Joint ED evidence violation')
            candidate_bound=cost if bound is None and status==2 and evidence is None else None if bound is None else bound*money_scale+objective_offset
            withdrawn=candidate_bound is not None and candidate_bound>cost+1e-7*max(1.,cost)
            lower=None if withdrawn or candidate_bound is None else min(cost,candidate_bound)
            gap=None if lower is None else max(0.,cost-lower)/max(abs(cost-objective_offset),1e-12)
            return dict(status='feasible',feasible=True,native_status=status,cost_yuan=cost,lower_bound_yuan=lower,
                relative_gap=gap,objective_offset_yuan=objective_offset,source=source,
                epigraph_tightening_yuan=max(0.,epigraph_extra),raw_solver_objective_yuan=value,lower_bound_withdrawn=withdrawn,
                initial_cost_yuan=None if warm is None else warm['cost_yuan'],initialization_verified=warm is not None,
                native_infeasibility_withdrawn=status==3 and warm is not None,
                global_optimality_verified=bool(status==2 if evidence is None else gap is not None and (gap<=1e-5 or cost-lower<=1e-8)),
                seconds=perf_counter()-began,components=components,max_balance_error=balance,max_domain_violation=violations,
                max_rebound_error=rebound,x=xx,dispatch=dict(dg=gg,shift=ss,shed=ll,buy=bb,sell=ee),
                evidence_gain=None if evidence is None else evidence.direct(xx))
    finally:
        if owns:env.dispose()


def jobs(run):
    entries={e['request']['id']:e for e in io.read(run/'nominal30/inputs.json')};tasks=[]
    contexts=[('population',i,0) for i in entries]
    contexts.extend(('confirmation',i,d) for i in CONFIRM for d in range(8))
    for phase,identity,day in contexts:
        e=entries[identity];label='g2' if phase=='population' else 'g4'
        metadata=run/'nominal30/physical/source_runtime'/e['request']['cohort']/'cases'/e['request']['case']/'metadata.json'
        base=io.read(metadata)['base_mva']
        for scenario,(alpha,guard) in SCENARIOS.items():
            directory=run/scenario/phase/'jobs'/identity/f'day{day:03d}'
            tasks.append(dict(id=identity,phase=phase,day=day,scenario=scenario,alpha=alpha,entry=e,base_mw=base,
                units_source=io.ref(metadata),planner=io.ref(directory/'planner.npz'),plans=io.ref(directory/'plans.json'),label=label))
    return tasks


def price_worker(task):
    import gurobipy as gp
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    e=task['entry'];results=[];plans=io.read(task['plans']['path']);plan=plans[task['label']]
    with np.load(task['planner']['path'],allow_pickle=False) as p:forecast=p['forecast'].copy();point=p['optimized_'+task['label']].copy()
    try:
        for shed in (False,True):
            common=dict(forecast=forecast,scale=e['scale'],base_mw=task['base_mw'],alpha=task['alpha'],allow_shed=shed,env=env)
            baseline=economic_dispatch(**common,fixed=np.zeros_like(point))
            result=economic_dispatch(**common,fixed=point)
            for packet in (baseline,result):packet.pop('x',None);packet.pop('dispatch',None)
            result.update({k:task[k] for k in ('id','phase','day','scenario')});result.update(allow_shed=shed,
                original_quote_qualified=plan['qualified'],zero_action_fallback=not plan['qualified'],baseline=baseline,
                incremental_yuan=None if not result['feasible'] or not baseline['feasible'] else result['cost_yuan']-baseline['cost_yuan'])
            if result['feasible'] and baseline['feasible']:
                de=result['components']['energy_bill_yuan']-baseline['components']['energy_bill_yuan']
                result['wholesale_price_sensitivity_incremental_yuan']={str(f):result['incremental_yuan']+(f-1)*de for f in (.5,1.,1.5)}
            results.append(result)
    finally:env.dispose()
    return results


def joint_worker(task):
    e=task['entry'];score=io.read(e['scores']['path']);H=np.array([p['H'] for p in score['parameters']])
    with np.load(task['planner']['path'],allow_pickle=False) as p:forecast=p['forecast'].copy();quoted=p['optimized_'+task['label']].copy()
    base=Model(H,e['target'],e['competitors'],e['scale'],score['sigma2']).prepare(forecast)
    evidence=Evidence(base,None);gamma=2. if task['label']=='g2' else 4.
    # Free ED is the original-domain convex lower reference, with identical
    # resources and no information constraint. Its cost is not a restricted-
    # domain lower bound. The nonconvex joint problem is solved independently.
    free=economic_dispatch(forecast,e['scale'],task['base_mw'],task['alpha'],seconds=3.,allow_shed=False)
    offset=free['cost_yuan'] if free['feasible'] and free['global_optimality_verified'] else 0.
    if free['feasible'] and free['global_optimality_verified'] and min(evidence.direct(free['x'])-evidence.thresholds(gamma))>=-1e-8:
        result=dict(free,source='free_ED_optimum_already_satisfies_evidence',objective_offset_yuan=offset)
    else:
        seed=None
        if min(evidence.direct(quoted)-evidence.thresholds(gamma))>=-1e-8:
            candidate=economic_dispatch(forecast,e['scale'],task['base_mw'],task['alpha'],fixed=quoted,seconds=3.,allow_shed=False)
            if candidate['feasible']:seed=candidate
        result=economic_dispatch(forecast,e['scale'],task['base_mw'],task['alpha'],seconds=10.,allow_shed=False,evidence=evidence,gamma=gamma,objective_offset=offset,warm=seed)
    saved={}
    for name,packet in (('free',free),('joint',result)):
        if packet['feasible']:
            saved[name+'_x']=packet['x']
            saved.update({name+'_'+key:value for key,value in packet['dispatch'].items()})
    directory=Path(task['point_directory']);directory.mkdir(parents=True,exist_ok=True)
    file=directory/f"{task['scenario']}__{task['id']}__day{task['day']:03d}.npz"
    np.savez_compressed(file,**saved)
    for packet in (free,result):packet.pop('x',None);packet.pop('dispatch',None);packet.pop('evidence_gain',None)
    result.update({k:task[k] for k in ('id','phase','day','scenario')});result['free_ED']=free
    result['private_point']=io.ref(file)
    if result['feasible'] and free['feasible']:
        require(result['cost_yuan']>=free['cost_yuan']-1e-6,'Information-constrained cost below free optimum')
        result['information_premium_yuan']=result['cost_yuan']-free['cost_yuan']
        if free['global_optimality_verified']:
            result['lower_bound_yuan']=max(result['lower_bound_yuan'] or 0.,free['cost_yuan'])
            result['information_premium_lower_bound_yuan']=max(0.,result['lower_bound_yuan']-free['cost_yuan'])
            result['relative_gap']=max(0.,result['cost_yuan']-result['lower_bound_yuan'])/max(abs(result['information_premium_yuan']),1e-12)
            result['global_optimality_verified']=result['relative_gap']<=1e-5 or result['cost_yuan']-result['lower_bound_yuan']<=1e-8
    return result


def run(run,out,previous=None):
    require(not out.exists(),'Preserve prior pricing');out.mkdir(parents=True);tasks=jobs(run)
    snapshot=out/'source_snapshot';snapshot.mkdir();shutil.copyfile(__file__,snapshot/'prices.py')
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),environment=io.environment(),tasks=tasks,
        market_prices_yuan_per_MWh=PRICE,local_generation_bid_yuan_per_MWh=[350,450,550],transfer_yuan_per_MWh=100,
        shedding_yuan_per_MWh=10000,resources='all nodes hypothetical fixed-on aggregate generation .06-.60 historical peak, baseline .33; shift +/- .15 peak with daily energy equality; ramp .15 peak/15min; optional shed <=5% gross load',
        prices_are='illustrative assumptions explicitly authorized by user, not a region/date observed market calibration',
        wholesale_sensitivity='fixed-plan accounting only; local bids fixed; wholesale energy term is constant in each implementation ED',
        pricing_information='forecast and historical power scales, nominal metadata base MVA only; no branch/true R/X or actual future P/Q/V',
        free_ED_reference='same-domain operating-cost minimum without evidence constraints',joint_limit_seconds=10.,fixed_ED_limit_seconds=3.,
        reused_fixed_checks=None if previous is None else io.ref(previous/'fixed_plan_results.json'),
        development=None if previous is None else dict(completion=io.ref(previous/'complete.json'),
            correction='v1 retained 544 fixed-plan checks and three failed joint objective checks. The check had treated a loose absolute-shift epigraph at a time-limited incumbent as an inconsistent physical cost. v2 verifies the exact slack identity and tightens that auxiliary variable; resource/price/evidence equations are unchanged. Recheck all eight joint examples; no new independent AC day.')))
    records=[];failures=[]
    if previous is not None:
        old=io.read(previous/'protocol.json')
        require(len(old['tasks'])==len(tasks),'Pricing contexts changed')
        for a,b in zip(old['tasks'],tasks,strict=True):
            require(all(a[k]==b[k] for k in ('id','phase','day','scenario','alpha','base_mw','planner','plans')),'Fixed pricing input changed')
        records=io.read(previous/'fixed_plan_results.json')
        require(len(records)==2*len(tasks),'Incomplete fixed-plan source')
    else:
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(price_worker,t):t for t in tasks}
            for f in as_completed(futures):
                try:records.extend(f.result())
                except Exception as exc:failures.append(dict(id=futures[f]['id'],error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'checkpoint.json',dict(fixed_records=len(records),worker_failures=failures))
    io.write(out/'fixed_plan_results.json',records)
    # One predeclared first confirmation day for each of four diagnostic cases,
    # both capacities: eight integrated ED + identification QCQPs.
    joint_tasks=[t for t in tasks if t['phase']=='confirmation' and t['day']==0 and t['scenario'] in ('nominal30','nominal50')]
    for task in joint_tasks:task['point_directory']=str(out/'joint_points')
    joint=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(joint_worker,t):t for t in joint_tasks}
        for f in as_completed(futures):
            try:joint.append(f.result())
            except Exception as exc:failures.append(dict(joint=True,id=futures[f]['id'],error=repr(exc),traceback=traceback.format_exc()))
            io.write(out/'checkpoint.json',dict(fixed_records=len(records),joint_records=len(joint),worker_failures=failures))
    io.write(out/'joint_results.json',joint)
    io.write(out/'complete.json',dict(status='passed' if not failures else 'complete_with_failures',fixed_plan_checks=len(records),joint_checks=len(joint),
        failures=failures,new_independent_days=0,source=io.ref(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-fixed',type=Path)
    a=p.parse_args();run(a.run.resolve(),a.output.resolve(),None if a.reuse_fixed is None else a.reuse_fixed.resolve())
