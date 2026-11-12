"""Existing physical-MSE gate and single-pair device-graph solver."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from .device_graph import build_device_graph, _values, _attribute
from .. import io as base

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


def solve_factorized(p,gate,out,warm,limits):
    """Reuse original device graph; a bounded NonConvex2 solve of score>=h.

    Auxiliary z=offset+F*u avoids forming a dense5760-square Q. Resource
    coordinates whose model-difference effect is EXACTLY zero have optimal
    request0: constraints are resource-separable and fees nonnegative. No
    nonzero/weak effect is dropped, and original capacities are unchanged.
    """
    if getattr(gate, 'zero_control_only', True):
        raise ValueError('A full control factor is required; zero-screen gates cannot price nonzero actions')
    import gurobipy as gp
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists():raise ValueError('Existing solver attempt preserved')
    base.write(out/'request.json',dict(inputs=gate.input_receipt,limits=limits,factor_shape=gate.factor.shape,h=gate.h,
       resources=gate.resources,all_resources=p.device.R,original_graph_source=base.ref(Path(__file__).with_name('device_graph.py')),
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
                  and gate.bound(pt['mean_u'])['sufficient']]
            if good:
                name,pt=min(good,key=lambda item:item[1]['fee'])
                bound=result['lower_bound'];obj=float(pt['fee'])
                consistent=bound is None or bound<=obj+1e-7*max(1.,abs(obj))
                gap=None if bound is None or not consistent else max(0.,obj-bound)/max(obj,1e-12)
                result.update(qualified=True,objective=obj,verified_gap=gap,retained_source=name,
                    bound_consistent_with_replayed_incumbent=consistent,
                    global_gap_target_met=bool(gap is not None and gap<=limits['mip_gap']),
                    gate=gate.bound(pt['mean_u']),device=p.audit_delivery(pt['mean_u']))
                np.savez_compressed(out/'quote.npz',x=pt['x'],mean_u=pt['mean_u'])
            base.write(out/'summary.json',result)
            base.write(out/'complete.json',dict(status='complete',qualified=result['qualified']))
            return result
