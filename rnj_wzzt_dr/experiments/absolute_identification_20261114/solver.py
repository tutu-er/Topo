"""Symmetric absolute-recovery QCQP over a frozen representative bank.

No truth/network/actual-future inputs. Conditional certificate applies only to
the declared candidate-as-true set and the matched Gaussian response model.
"""
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from ..customer_flexibility_20261113.placement import terms
from ..flexibility_sweep_20261113.fitting import calibrated_means


def target_set(history_sse, sigma2, mass=.99):
    if not 0 < mass <= 1:
        raise ValueError('target_mass must be in (0,1]')
    history_sse = np.asarray(history_sse, float)
    logw = -(history_sse-history_sse.min())/(2*sigma2)
    logw -= logsumexp(logw)
    order = np.argsort(-logw, kind='stable')
    if mass == 1:
        return order.tolist(), np.exp(logw)
    stop = min(int(np.searchsorted(np.cumsum(np.exp(logw[order])),mass)),len(order)-1)
    threshold = logw[order[stop]]
    return [int(i) for i in order if logw[i]>=threshold-1e-12], np.exp(logw)


class SymmetricRecovery:
    def __init__(self, case, score, validation, forecast, base_kw, mode='symmetric', target_mass=.99):
        if mode not in ('symmetric','symmetric_guard'):
            raise ValueError(mode)
        self.case, self.score, self.forecast = case, score, np.asarray(forecast,float)
        self.base_kw, self.mode = float(base_kw), mode
        self.H = np.asarray([p['H'] for p in score['parameters']],float)
        self.history = np.array([p['history_sse'] for p in score['parameters']],float)
        self.sigma2 = float(score['sigma2'])
        if self.sigma2<=0 or not np.isfinite(self.sigma2):
            raise ValueError('Positive finite score variance required')
        self.K = len(self.H)
        self.targets, self.weights = target_set(self.history,self.sigma2,target_mass)
        self.target_mass = target_mass
        selected = set(self.targets)
        self.pairs = [(i,j) for i in range(self.K) for j in range(i+1,self.K) if i in selected or j in selected]
        self.alpha = .1/max(1,self.K-1)
        self.z = float(norm.isf(self.alpha))
        cells = score['cells']+self.forecast.shape[0]*case.N
        self.tie = max(1e-7,cells*score['reference_mse']*score['validation_tie']/(2*self.sigma2))
        self.q, self.a, self.c, self.h, self.upper = [], [], [], [], []
        self.guards = []
        means = calibrated_means(score,validation) if mode=='symmetric_guard' else None
        for i,j in self.pairs:
            D = self.H[i]-self.H[j]
            d = self.forecast@D.T
            q,a = terms(case,self.H[i],self.H[j],self.H[i],self.forecast,self.sigma2,self.base_kw)
            self.q.append(q); self.a.append(a); self.c.append(float(np.sum(d*d)/(2*self.sigma2)))
            h = (self.history[j]-self.history[i])/(2*self.sigma2)
            self.h.append(min([v for target,v in ((i,h),(j,-h)) if target in selected]))
            effect = (D[:,case.node]+D[:,case.N+case.node]*case.qratio)/self.base_kw
            positive,negative=np.maximum(effect,0),np.minimum(effect,0)
            lower=np.repeat(case.lower,case.repeat,axis=0)
            upper=np.repeat(case.upper,case.repeat,axis=0)
            lo=d+lower@positive.T+upper@negative.T
            hi=d+upper@positive.T+lower@negative.T
            self.upper.append(float(np.maximum(lo*lo,hi*hi).sum()/(2*self.sigma2)))
            if means is not None:
                for left,right in ((i,j),(j,i)):
                    Q,a=terms(case,self.H[left],self.H[right],means[left],self.forecast,self.sigma2,self.base_kw)
                    self.guards.append((left,right,means[left],Q,a))
        for field in ('q','a','c','h','upper'):
            setattr(self,field,np.asarray(getattr(self,field)))

    def direct(self,u):
        P=self.forecast+self.case.shift_pq_pu(u,self.base_kw)
        J=np.array([np.sum((P@(self.H[i]-self.H[j]).T)**2)/(2*self.sigma2) for i,j in self.pairs])
        margin=self.h+J-self.z*np.sqrt(2*J)-self.tie
        pair_failure=norm.cdf(np.divide(self.tie-self.h-J,np.sqrt(2*J),
            out=np.full_like(J,-np.inf),where=J>0))
        pair_failure[(J==0)&(self.h<=self.tie)]=1.
        failure_by_target={str(i):0. for i in self.targets}
        for k,(i,j) in enumerate(self.pairs):
            h=(self.history[j]-self.history[i])/(2*self.sigma2)
            for target,offset in ((i,h),(j,-h)):
                if str(target) in failure_by_target:
                    failure=float(norm.cdf((self.tie-offset-J[k])/np.sqrt(2*J[k]))) if J[k]>0 else float(offset<=self.tie)
                    failure_by_target[str(target)]+=failure
        gains=[]
        for i,j,K,Q,a in self.guards:
            D,E=self.H[i]-self.H[j],K-self.H[i]
            def mean(power):
                d=power@D.T; bias=power@E.T
                return np.sum(d*d)/(2*self.sigma2)+np.sum(bias*d)/self.sigma2
            gains.append(float(mean(P)-mean(self.forecast)))
        minimum_margin=float(margin.min()) if len(margin) else None
        minimum_guard=min(gains) if gains else None
        feasible=bool(np.all(margin>=-1e-7) and np.all(pair_failure<=self.alpha+1e-9)
            and all(g>=-1e-7 for g in gains))
        return dict(feasible=feasible,J=J,minimum_pair_margin=minimum_margin,
            pair_failure=pair_failure,target_failure_sums=failure_by_target,
            worst_recovery_lower=max(0.,1-max(failure_by_target.values(),default=0.)),
            minimum_guard_gain=minimum_guard,guard_gains=gains)

    def verify_expansion(self,u):
        actual=self.direct(u)
        if len(self.pairs):
            expanded=np.einsum('hi,kij,hj->k',u,self.q,u)+2*np.einsum('khi,hi->k',self.a,u)+self.c
            if not np.allclose(expanded,actual['J'],atol=1e-8,rtol=1e-8):
                raise ArithmeticError('Full prediction distance and QCQP mismatch')
        for g,(_,_,_,Q,a) in zip(actual['guard_gains'],self.guards):
            expanded=np.einsum('hi,ij,hj',u,Q,u)+2*np.sum(a*u)
            if not np.isclose(expanded,g,atol=1e-8,rtol=1e-8):
                raise ArithmeticError('Original directional guard and QCQP mismatch')
        return actual


def seed_items(seed_points):
    if seed_points is None:
        return []
    if isinstance(seed_points,dict):
        return list(seed_points.items())
    return [(f'common_seed_{i}',u) for i,u in enumerate(seed_points)]


def plan(case,score,validation,forecast,base_kw,*,mode='symmetric',seconds=12.,env,
         seed_points=None,target_mass=.99):
    import gurobipy as gp
    began=perf_counter()
    evidence=SymmetricRecovery(case,score,validation,forecast,base_kw,mode,target_mass)
    zero=np.zeros(case.shape)
    baseline=evidence.verify_expansion(zero)
    common=dict(mode=mode,target_mass=target_mass,targets=evidence.targets,
        target_topology_ids=[score['models'][i]['topology_id'] for i in evidence.targets],
        historical_selected_mass=float(evidence.weights[evidence.targets].sum()),
        candidates=evidence.K,pair_count=len(evidence.pairs),guard_count=len(evidence.guards),
        individual_comparison_risk=evidence.alpha,total_comparison_risk=.1,baseline=baseline,
        constrained_bank='frozen representative models; native-bank outcomes reported separately',
        confidence_scope='conditional on true frozen model belonging to selected target set; given forecast and nominal response; iid Gaussian residual variance equals declared score variance; no rejection guarantee',
        selection_scope='historical cumulative probability target set with cutoff ties; its truth coverage is not guaranteed by the conditional recovery bound',
        time_scope='includes model, seed auditing and native solve; caller must include external common-seed generation separately')
    if baseline['feasible']:
        return dict(common,feasible=True,status='zero_sufficient',source='zero',cost_yuan=0.,
            lower_bound_yuan=0.,relative_gap=0.,global_optimality_verified=True,seconds=perf_counter()-began),zero
    initial_records=[]; point=None; warm=None; warm_merit=-np.inf; seed_name=None
    tick=perf_counter()
    for name,u in seed_items(seed_points):
        try:
            u=np.asarray(u,float)
            domain=case.audit(u)
            info=evidence.verify_expansion(u)
            valid=bool(domain['passed'] and info['feasible'])
            fee=case.fee(u)
            merit=min(v for v in (info['minimum_pair_margin'],info['minimum_guard_gain']) if v is not None)
            initial_records.append(dict(name=name,resource_feasible=domain['passed'],qcqp_feasible=info['feasible'],cost_yuan=fee,minimum_margin=merit))
            if valid and (point is None or fee<case.fee(point)):
                point=u.copy(); seed_name=name
            if domain['passed'] and merit>warm_merit:
                warm=u.copy(); warm_merit=merit
        except Exception as exc:
            initial_records.append(dict(name=name,error=repr(exc),qcqp_feasible=False))
    if point is not None:
        warm=point.copy()
    seed=None if point is None else point.copy()
    initial=dict(source='caller_common_seed_pool',feasible=seed is not None,name=seed_name,
        cost_yuan=None if seed is None else case.fee(seed),audit_seconds=perf_counter()-tick,candidates=initial_records)
    with gp.Model('symmetric_absolute_recovery',env=env) as m:
        for name,value in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=2,FeasibilityTol=1e-9,NumericFocus=1,MIPGap=1e-4).items():
            m.setParam(name,value)
        u,fee=case.attach(m)
        warm_info=None if warm is None else evidence.direct(warm)
        for k in range(len(evidence.pairs)):
            J=sum(u[h]@evidence.q[k]@u[h] for h in range(case.H))+2*(evidence.a[k]*u).sum()+evidence.c[k]
            s=m.addVar(lb=0.,ub=float(np.sqrt(evidence.upper[k])*(1+1e-9)),name='pair_distance_'+str(k))
            m.addConstr(J<=s*s)
            m.addConstr(evidence.h[k]+J>=evidence.z*np.sqrt(2)*s+evidence.tie+1e-8)
            if warm_info is not None:
                s.Start=float(np.sqrt(warm_info['J'][k]))
        for left,right,K,Q,a in evidence.guards:
            gain=sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()
            m.addConstr(gain>=0.)
        m.setObjective(fee)
        if warm is not None:
            u.Start=warm
        m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began))
        m.optimize()
        try:
            bound=float(m.ObjBound)
            if not np.isfinite(bound):
                bound=None
        except (AttributeError,gp.GurobiError):
            bound=None
        source='verified_initial_retained' if seed is not None else None
        if m.SolCount:
            candidate=np.asarray(u.X)
            if case.audit(candidate)['passed'] and evidence.verify_expansion(candidate)['feasible']:
                if point is None or case.fee(candidate)<case.fee(point)-1e-9:
                    point=candidate; source='native_found' if seed is None else 'native_improved_initial'
        if point is None:
            return dict(common,feasible=False,status='native_infeasible' if m.Status==3 else 'unresolved',
                native_status=int(m.Status),initial=initial,infeasibility_proved=m.Status==3,
                cost_yuan=None,lower_bound_yuan=bound,global_optimality_verified=False,
                seconds=perf_counter()-began),None
        cost=case.fee(point)
        conflict=bound is not None and bound>cost+1e-7*max(1.,cost)
        lower=0. if bound is None or conflict else max(0.,min(cost,bound))
        gap=(cost-lower)/max(cost,1e-12)
        return dict(common,feasible=True,status='feasible',source=source,native_status=int(m.Status),
            initial=initial,cost_yuan=cost,lower_bound_yuan=lower,relative_gap=gap,
            lower_bound_withdrawn=conflict,native_infeasibility_withdrawn=m.Status==3,
            global_optimality_verified=gap<=1e-4,
            improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
            final=evidence.verify_expansion(point),resource_audit=case.audit(point),seconds=perf_counter()-began),point
