"""Physical-MSE DR diagnostic using the neighbouring pure_mse_gate algebra.

The source E gate is M0-M1 > 1e-10/K + tau*M0, so kappa=1/(1-tau).
It is an E TRAINING acceptance rule in that project, whose final selector is
strict validation MAE. Here it is explicitly a frozen-pair comparison, not a
new E proposal/search. Original3% MAE outputs are evaluated separately.

Physical MSE is unweighted: one scalar s=max(likelihood_sd) only changes units.
Per-cell voltage variances E=diag((sd/s)^2) are retained in the integral.
For model1 reference, z~N(d,C) is forecast difference noise and e~N(0,E) is
independent remaining voltage error. This independence does not describe
P/Q errors-in-variables; original metering/AC/response are mismatch evaluation.

L=||z||^2+2e'z-(kappa-1)||e||^2. With lambda=1/(4*kappa*max(E)),
 A_i=1-2lambda(kappa-1)E_i,
 t_i=lambda*(1-2lambda*kappa E_i)/A_i,
 G=I+2 diag(sqrt(t)) C diag(sqrt(t)),
 log failure_bound=lambda*delta-.5sum(log A_i)-.5logdetG
                    -||chol(G)^-1 diag(sqrt(t)) d(u)||^2.
The resulting score>=h is the SAME fixed-lambda exponential sufficient
condition, with physical SSE instead of cell-wise whitened squared loss.
Both Gaussian integrations are exact; the probability inequality is conservative.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csr_matrix

import run as base
from vendor.miqp_gate import pure_mse_gate
from vendor.solver import build_device_graph, _values, _attribute


@dataclass
class MSEGate:
    offset: np.ndarray
    factor: np.ndarray
    resources: np.ndarray
    h: float
    kappa: float
    delta: float
    beta: float
    scale: float
    K: float
    tau: float
    history_S: tuple
    history_cells: int
    kind: str = 'unweighted_physical_mse_exponential'

    def value(self,u):
        values=np.asarray(u,float)[self.resources].ravel()
        z=self.offset+self.factor@values
        return float(z@z)

    def bound(self,u):
        score=self.value(u);log_failure=math.log1p(-self.beta)+self.h-score
        return dict(score=score,h=self.h,margin=score-self.h,
            log_failure_upper=log_failure,success_lower=-math.expm1(min(0.,log_failure)),
            sufficient=score>=self.h)


def build(case,tau,*,full=False):
    case,cfg,p,a,old=base.load(case)
    if not 0<=tau<1:raise ValueError('Require0<=tau<1')
    s=float(a['likelihood_sd'].max());kappa=1/(1-tau);beta=.9
    E=(a['likelihood_sd'].ravel()/s)**2
    lam=1/(4*kappa*E.max())
    diagA=1-2*lam*(kappa-1)*E
    t=lam*(1-2*lam*kappa*E)/diagA
    if np.any(diagA<=0) or np.any(t<=0):raise ValueError('Invalid exponential-integral range')
    # Convert old normalized difference covariance back to common scalar units.
    ratio=a['likelihood_sd'].ravel()/s
    C=old['C']*ratio[:,None]*ratio[None,:]
    G=np.eye(len(E))+2*np.sqrt(t)[:,None]*C*np.sqrt(t)[None,:]
    chol=np.linalg.cholesky((G+G.T)/2)
    mean=(a['forecast_power']@(a['H1']-a['H0']).T).ravel()/s
    offset=solve_triangular(chol,np.sqrt(t)*mean,lower=True)
    # K has the same physical units as the neighbouring implementation.
    train_power=a['history_training_power']
    if 'history_training_drop' not in a:raise ValueError('Need full original training drop for K')
    rt=(a['history_training_drop']-train_power@a['H0'].T)*1e-6
    K=float(np.mean(np.abs(rt))/np.mean(rt*rt))
    rh=[(a['history_drop']-a['history_power']@a[key].T)/s for key in ('H0','H1')]
    S0h,S1h=[float(np.sum(r*r)) for r in rh]
    cells=a['history_drop'].size+p.T*p.N
    # Convert the pure gate's numerical MSE floor to normalized SSE units.
    delta_all=cells*(1e-10/K)/((s*1e-6)**2*(1-tau))
    delta=float(delta_all-(S0h-kappa*S1h))
    h=float(lam*delta-math.log1p(-beta)-.5*np.log(diagA).sum()-np.log(np.diag(chol)).sum())
    node=a['resource_node_index'].astype(int);q=a['resource_q_over_p'];dh=a['H1']-a['H0']
    effect=dh[:,node]/float(a['base_power_kw'])+dh[:,p.N+node]*q/float(a['base_reactive_kvar'])
    active=np.flatnonzero(np.any(effect!=0,axis=0))  # Exact zero only; weak nonzero effects stay.
    if full:
        D=np.zeros((p.T*p.N,len(active)*p.T))
        for ti in range(p.T):
            D[ti*p.N:(ti+1)*p.N,np.arange(len(active))*p.T+ti]=effect[:,active]/s
        factor=solve_triangular(chol,np.sqrt(t)[:,None]*D,lower=True)
    else:factor=np.zeros((len(E),len(active)*p.T))
    gate=MSEGate(offset,factor,active,h,kappa,delta,beta,s,K,tau,(S0h,S1h),a['history_drop'].size)
    return case,cfg,p,a,gate


def solve_factorized(p,gate,out,warm,limits):
    """Reuse original device graph; a bounded NonConvex2 solve of score>=h.

    Auxiliary z=offset+F*u avoids forming a dense5760-square Q. Resource
    coordinates whose model-difference effect is EXACTLY zero have optimal
    request0: constraints are resource-separable and fees nonnegative. No
    nonzero/weak effect is dropped, and original capacities are unchanged.
    """
    import gurobipy as gp
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists():raise ValueError('Existing solver attempt preserved')
    base.write(out/'request.json',dict(limits=limits,factor_shape=gate.factor.shape,h=gate.h,
       resources=gate.resources,all_resources=p.device.R,original_graph_source=base.ref(Path(__file__).parent/'vendor/solver.py'),
       source=base.ref(__file__),warm=base.contract_hash(warm)))
    began=time.perf_counter();audit_warm=p.nominal(p.requests_to_values(warm))
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model('physical_MSE_DR_factorized',env=env) as model:
            for key,value in dict(OutputFlag=1,LogToConsole=0,LogFile=str(out/'gurobi.log'),
                TimeLimit=limits['seconds'],MIPGap=limits['mip_gap'],Threads=1,Seed=0,
                NonConvex=2,NumericFocus=2,MIPFocus=1,FeasibilityTol=1e-8,IntFeasTol=1e-8,
                OptimalityTol=1e-8,DualReductions=0).items():model.setParam(key,value)
            x,u,cost=build_device_graph(model,p,warm)
            inactive=set(range(p.device.R))-set(gate.resources.tolist())
            for r in inactive:
                for h in range(p.device.H):model.addConstr(x[r,h].expr==0.)
            lo=p.device.delta_lower_kw[gate.resources].ravel()
            hi=p.device.delta_upper_kw[gate.resources].ravel()
            delivery=model.addMVar(len(lo),lb=lo,ub=hi,name='observable_delivery')
            for i,term in enumerate(u[gate.resources].ravel()):model.addConstr(delivery[i]==term.expr)
            delivery.Start=audit_warm['mean_u'][gate.resources].ravel()
            varying=np.any(gate.factor!=0,axis=1)
            constant=float(np.sum(gate.offset[~varying]**2))
            F=gate.factor[varying];b=gate.offset[varying]
            pos=np.maximum(F,0);neg=np.minimum(F,0)
            zl=b+pos@lo+neg@hi;zh=b+pos@hi+neg@lo
            z=model.addMVar(len(b),lb=zl,ub=zh,name='quadratic_factor')
            model.addConstr(z==b+csr_matrix(F)@delivery)
            z.Start=b+F@audit_warm['mean_u'][gate.resources].ravel()
            model.addConstr(z@z+constant>=gate.h)
            model.setObjective(cost,gp.GRB.MINIMIZE);model.update()
            model.write(str(out/'model.mps.gz'));model.optimize()
            result=dict(status=int(model.Status),has_solution=bool(model.SolCount),
                objective=None,lower_bound=_attribute(model,'ObjBound'),reported_gap=_attribute(model,'MIPGap') if model.SolCount else None,
                qualified=False,global_gap_target_met=False,seconds=time.perf_counter()-began,
                variables=model.NumVars,binaries=model.NumBinVars,exact_zero_effect_resources=len(inactive))
            candidates=[]
            if model.SolCount:
                requests=_values(x)
                requests=p.expand(np.clip(requests.ravel()[p.active],p.lower,p.upper))
                point=p.nominal(p.requests_to_values(requests))
                candidates.append(('solver',point))
            candidates.append(('input_warm',audit_warm))
            good=[(name,pt) for name,pt in candidates if p.audit_delivery(pt['mean_u'])['passed']
                  and gate.value(pt['mean_u'])>=gate.h-1e-7*max(1.,abs(gate.h))]
            if good:
                name,pt=min(good,key=lambda item:item[1]['fee'])
                bound=result['lower_bound'];obj=float(pt['fee'])
                gap=None if bound is None else max(0.,obj-bound)/max(obj,1e-12)
                result.update(qualified=True,objective=obj,verified_gap=gap,retained_source=name,
                    global_gap_target_met=bool(gap is not None and gap<=limits['mip_gap']),
                    gate=gate.bound(pt['mean_u']),device=p.audit_delivery(pt['mean_u']))
                np.savez_compressed(out/'quote.npz',x=pt['x'],mean_u=pt['mean_u'])
            base.write(out/'summary.json',result)
            base.write(out/'complete.json',dict(status='complete',qualified=result['qualified']))
            return result


def plan(case,out,tau,warm_path=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists():raise ValueError('Existing plan preserved')
    case,cfg,p,a,g=build(case,tau,full=False)
    source_files=[Path(__file__),Path(__file__).parent/'run.py',Path(__file__).parent/'MIQP_PROTOCOL.json',
                  *sorted((Path(__file__).parent/'vendor').glob('*.py'))]
    inputs=[case/'case.json',p.case_path,p.arrays_path,case/cfg['model_arrays'],case/cfg['candidate_file']]
    request=dict(protocol=base.ref(case/'PROTOCOL.json'),mse_protocol=base.ref(case/'MIQP_PROTOCOL.json'),
        inputs=[base.ref(x) for x in inputs],sources={str(x.relative_to(base.HERE)):base.sha(x) for x in source_files},
        tau=tau,kappa=g.kappa,K=g.K,scale_micro=g.scale,history_S=g.history_S,delta=g.delta,
        normalization='one scalar only; physical unweighted MSE preserved',warm=base.ref(warm_path) if warm_path else None,
        scope='fixed pair, borrowed E gate shape; no new MIQP topology proposal',no_actual_future_or_truth_used=True)
    base.write(out/'request.json',request)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in request['sources']:z.write(base.HERE/name,name)
    zero=np.zeros((p.device.R,p.T));initial=g.bound(zero)
    if initial['sufficient']:
        result=dict(status='CERTIFIED_COST_ZERO',objective=0.,lower_bound=0.,verified_gap=0.,qualified=True,
                    global_gap_target_met=True,solver_performed=False)
        u=zero;x=p.nominal(p.zero)['x']
    else:
        case,cfg,p,a,g=build(case,tau,full=True)
        warm=np.zeros(p.shape)
        if warm_path:
            saved=base.arrays(warm_path);small=saved['x']
            if small.shape[1]!=p.device.H or small.shape[0]>p.device.R:raise ValueError('Incompatible declared warm')
            warm[:small.shape[0]]=small
        result=solve_factorized(p,g,out/'solve',warm,base.read(case/'MIQP_PROTOCOL.json')['optimizer'])
        if not result['qualified']:
            base.write(out/'summary.json',dict(status='unresolved',solver=result,zero_bound=initial,tau=tau))
            return result
        saved=base.arrays(out/'solve/quote.npz');u,x=saved['mean_u'],saved['x']
    rotate,rot=base.rotation(p,p.device.fees(u))
    np.savez_compressed(out/'quotes.npz',optimized=u,zero=zero,rotation=rotate,optimized_hourly_requests=x)
    base.write(out/'policy_freeze.json',dict(status='frozen',quote=dict(path='quotes.npz',sha256=base.sha(out/'quotes.npz')),
        request=dict(path='request.json',sha256=base.sha(out/'request.json')),no_actual_future_or_truth_used=True))
    final=dict(status='complete',tau=tau,kappa=g.kappa,K=g.K,history_S=g.history_S,delta=g.delta,
        zero_bound=initial,optimized_bound=g.bound(u),solver=result,rotation=rot,
        fees={name:p.device.fees(v) for name,v in [('optimized',u),('zero',zero),('rotation',rotate)]},
        nominal_absolute_DR_kwh=float(.25*np.abs(u).sum()))
    base.write(out/'summary.json',final);base.write(out/'complete.json',dict(status='complete'))
    return final


def metric(case,power,drop,tau):
    """Exact borrowed physical-MSE formula, scored by the original gate function."""
    case,cfg,p,a,_=base.load(case)
    s=float(a['likelihood_sd'].max())
    rt=(a['history_training_drop']-a['history_training_power']@a['H0'].T)*1e-6
    K=float(np.mean(np.abs(rt))/np.mean(rt*rt))
    losses=[]
    for key in ('H0','H1'):
        old=(a['history_drop']-a['history_power']@a[key].T)*1e-6
        new=(drop-power@a[key].T)*1e-6
        losses.append(float((np.sum(old*old)+np.sum(new*new))/(old.size+new.size)))
    return pure_mse_gate(*losses,K=K,tau=tau)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=base.HERE)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tau',type=float,required=True)
    parser.add_argument('--warm',type=Path)
    args=parser.parse_args()
    print(json.dumps(plan(args.case,args.out,args.tau,args.warm),allow_nan=False))
