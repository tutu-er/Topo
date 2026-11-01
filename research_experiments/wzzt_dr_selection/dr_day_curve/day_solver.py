"""Bounded nonconvex MIQCP for a common 24-hour curve and daily quotes.

No convexity claim: daily quote times shifted kWh is bilinear. Absolute
residuals are represented exactly, including their sign crossings.
"""
from __future__ import annotations

import math,time
import numpy as np
from .day_model import DT,canonical_prices,payment,resource_audit,scenario_summary


def _receipt(model):
    def finite(v):
        try:
            x=float(v);return x if math.isfinite(x) and abs(x)<1.e90 else None
        except Exception:return None
    return dict(status=int(model.Status),solutions=int(model.SolCount),
        objective=finite(model.ObjVal) if model.SolCount else None,
        global_bound=finite(model.ObjBound),runtime_seconds=float(model.Runtime),
        node_count=float(model.NodeCount),mip_gap=finite(model.MIPGap) if model.SolCount else None)


def solve(problem,scenarios,*,seconds=60.,cost_seconds=30.,warm_curves=(),log_path=None):
    S=len(scenarios);J=len(problem['elasticities']);T=problem['slots'];N=problem['n']
    if not S or any(np.asarray(s['base_bad_micro']).shape!=(T,N) for s in scenarios):
        raise ValueError('aligned complete-day scenarios required')
    zero=np.zeros((J,T));zero_summary=scenario_summary(problem,scenarios,zero)
    if zero_summary['passed']>=zero_summary['required']:
        return dict(status='zero_cost_optimal',mean_u=zero.tolist(),prices=[0.]*J,
            training=zero_summary,resource_audit=resource_audit(problem,zero),
            primary=None,secondary=None,cost_global_lower_bound=0.,
            proof_scope='nonnegative payments and a verified zero-cost feasible curve')
    import gurobipy as gp
    from gurobipy import GRB
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    model=gp.Model('fixed_24h_curve_quote',env=env)
    try:
        for k,v in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=2,NumericFocus=2,
            FeasibilityTol=1.e-8,OptimalityTol=1.e-8,MIPGap=1.e-6,MIPGapAbs=1.e-7,
            DualReductions=0).items():model.setParam(k,v)
        if log_path is not None:model.setParam('LogFile',str(log_path))
        lam=np.asarray(problem['elasticities']);caps=np.asarray(problem['price_caps'])
        eta_max=1+np.abs(np.asarray(problem['factors'])).sum(axis=1)
        lower=np.maximum(np.asarray(problem['power_lower'])/eta_max[:,None],-lam[:,None]*caps[:,None])
        upper=np.minimum(np.asarray(problem['power_upper'])/eta_max[:,None],lam[:,None]*caps[:,None])
        lower=np.where(problem['available'],lower,0.);upper=np.where(problem['available'],upper,0.)
        price=model.addVars(J,lb=0.,ub={j:float(caps[j]) for j in range(J)},name='daily_price')
        u=model.addVars(J,T,lb={(j,t):float(lower[j,t]) for j in range(J) for t in range(T)},
            ub={(j,t):float(upper[j,t]) for j in range(J) for t in range(T)},name='mean_kw')
        reduction=model.addVars(J,T,lb=0.,ub={(j,t):float(max(-lower[j,t],0.)) for j in range(J) for t in range(T)},name='reduced_kw')
        energy_caps={j:float(DT*np.maximum(-lower[j],0.).sum()) for j in range(J)}
        E=model.addVars(J,lb=0.,ub=energy_caps,name='shifted_kwh')
        # Finite bounds come from physical power/time and quotation limits,
        # never from an imposed monetary budget when budget is None.
        fee=model.addVars(J,lb=0.,ub={j:float(caps[j]*energy_caps[j]) for j in range(J)},name='mean_fee')
        for j in range(J):
            model.addConstr(gp.quicksum(u[j,t] for t in range(T))==0.,name=f'daily_energy_{j}')
            for t in range(T):
                model.addConstr(u[j,t]<=float(lam[j])*price[j])
                model.addConstr(-u[j,t]<=float(lam[j])*price[j])
                model.addConstr(reduction[j,t]>=-u[j,t])
                prefix=DT*gp.quicksum(u[j,s] for s in range(t+1))
                model.addConstr(prefix>=float(problem['energy_lower'][j,t]/eta_max[j]))
                model.addConstr(prefix<=float(problem['energy_upper'][j,t]/eta_max[j]))
            model.addConstr(E[j]==DT*gp.quicksum(reduction[j,t] for t in range(T)))
            model.addQConstr(fee[j]==price[j]*E[j],name=f'nonconvex_payment_{j}')
        if problem['budget'] is not None:
            model.addConstr(gp.quicksum(float(eta_max[j])*fee[j] for j in range(J))<=problem['budget'],name='optional_hard_budget')
        chosen=model.addVars(S,vtype=GRB.BINARY,name='counted_scenario')
        slack=model.addVar(lb=0.,ub=1.e6,name='target_shortfall_micro')
        model.addConstr(chosen.sum()>=math.ceil(.9*S),name='sample_90pct')
        C=np.asarray(problem['contrast']);D=np.asarray(problem['factors'])
        compilation=dict(sign_stable=0,exact_abs=0,constant=0)
        abs_records=[]
        for s,sc in enumerate(scenarios):
            eta=1+D@np.asarray(sc['z']);coefficient=C*eta[None,:]
            expression=gp.LinExpr(0.);min_sum=0.;max_sum=0.
            for t in range(T):
                for i in range(N):
                    b=float(sc['base_bad_micro'][t,i]);co=coefficient[i]
                    lo=b+float(np.minimum(co*lower[:,t],co*upper[:,t]).sum())-1.e-8
                    hi=b+float(np.maximum(co*lower[:,t],co*upper[:,t]).sum())+1.e-8
                    min_sum+=0. if lo<=0<=hi else min(abs(lo),abs(hi));max_sum+=max(abs(lo),abs(hi))
                    if not np.any(co):
                        expression+=abs(b);compilation['constant']+=1;continue
                    v=b+gp.quicksum(float(co[j])*u[j,t] for j in range(J) if co[j]!=0.)
                    if lo>=0.:
                        expression+=v;compilation['sign_stable']+=1
                    elif hi<=0.:
                        expression-=v;compilation['sign_stable']+=1
                    else:
                        x=model.addVar(lb=lo,ub=hi,name=f'residual_{s}_{t}_{i}')
                        a=model.addVar(lb=0.,ub=max(abs(lo),abs(hi)),name=f'absolute_{s}_{t}_{i}')
                        model.addConstr(x==v);model.addGenConstrAbs(a,x)
                        expression+=a;abs_records.append((x,a,s,t,i,co.copy(),b))
                        compilation['exact_abs']+=1
            const=float(problem['history_component_micro']-problem['day_weight']*(1+problem['new_tolerance'])*sc['reference_mae_micro']-problem['epsilon_micro'])
            weight=problem['day_weight']/(T*N);margin=const+weight*expression
            target=float(sc['target_micro']);low=const+weight*min_sum
            big=max(target-low,0.)+1.e-6
            model.addConstr(margin>=target-slack-big*(1-chosen[s]),name=f'exact_margin_{s}')
        curves=[zero]
        for v in warm_curves:
            a=np.asarray(v,float)
            if a.shape==(J,T) and resource_audit(problem,a)['feasible']:curves.append(a)
        starts=[]
        for a in curves:
            ss=scenario_summary(problem,scenarios,a);starts.append((ss['feasibility_slack_micro'],payment(a,canonical_prices(a,lam)),a,ss))
        starts.sort(key=lambda x:(x[0],x[1]));starts=starts[:8]
        def start(index,a,ss):
            model.Params.StartNumber=index;c=canonical_prices(a,lam);neg=np.maximum(-a,0.)
            for j in range(J):
                price[j].Start=float(c[j]);E[j].Start=float(DT*neg[j].sum())
                fee[j].Start=float(c[j]*DT*neg[j].sum())
                for t in range(T):u[j,t].Start=float(a[j,t]);reduction[j,t].Start=float(neg[j,t])
            slack.Start=ss['feasibility_slack_micro']+1.e-7
            order=sorted(range(S),key=lambda i:ss['rows'][i]['surplus_micro'],reverse=True)
            keep=set(order[:math.ceil(.9*S)])
            for s in range(S):chosen[s].Start=int(s in keep)
            for x,a_var,s,t,i,co,b in abs_records:
                residual=b+float(co@a[:,t]);x.Start=residual;a_var.Start=abs(residual)
        model.NumStart=len(starts)
        for k,(_,_,a,ss) in enumerate(starts):start(k,a,ss)
        model.setObjective(slack,GRB.MINIMIZE);model.setParam('TimeLimit',seconds)
        wall=time.perf_counter();model.optimize();primary=_receipt(model)
        primary['wall_seconds']=time.perf_counter()-wall
        result=dict(primary=primary,secondary=None,compilation=compilation,
                    warm_start_count=len(starts),best_warm_slack_micro=starts[0][0],
                    scope='finite training scenarios, fixed reference forecast; full continuous daily curves; nonconvex globally bounded solver')
        if not model.SolCount:
            result.update(status='no_incumbent',mean_u=None,prices=None);return result
        a=np.array([[u[j,t].X for t in range(T)] for j in range(J)])
        ss=scenario_summary(problem,scenarios,a);audit=resource_audit(problem,a)
        if not audit['feasible']:raise ValueError('returned curve fails original resource/fee replay')
        if ss['passed']>=ss['required']:
            slack.UB=0.;model.setObjective(fee.sum(),GRB.MINIMIZE)
            model.NumStart=1;start(0,a,ss);slack.Start=0.
            model.setParam('TimeLimit',cost_seconds);model.optimize();result['secondary']=_receipt(model)
            if model.SolCount:
                a=np.array([[u[j,t].X for t in range(T)] for j in range(J)])
                ss=scenario_summary(problem,scenarios,a);audit=resource_audit(problem,a)
            result['status']='target_feasible_cost_incumbent' if ss['passed']>=ss['required'] and audit['feasible'] else 'secondary_return_unresolved'
        else:
            bound=primary['global_bound']
            result['status']='sample_target_infeasible' if bound is not None and bound>1.e-6 else 'sample_target_unresolved'
        result.update(mean_u=a.tolist(),prices=canonical_prices(a,lam).tolist(),
                      training=ss,resource_audit=audit)
        return result
    finally:
        model.dispose();env.dispose()
