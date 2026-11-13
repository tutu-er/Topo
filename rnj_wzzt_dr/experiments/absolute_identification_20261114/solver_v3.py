"""Freeze per-target unequal error allocations before the absolute QCQP."""
from time import perf_counter
import numpy as np
from scipy.stats import norm

from .solver import SymmetricRecovery, seed_items
from .solver_v2 import CalibratedSymmetricRecovery


class WeightedRecovery(CalibratedSymmetricRecovery):
    def __init__(self,case,score,validation,forecast,base_kw,mode='weighted_nominal',target_mass=.99):
        if mode not in ('weighted_nominal','weighted_capped'):
            raise ValueError(mode)
        self.requested_mode=mode
        super().__init__(case,score,validation,forecast,base_kw,
            mode='capped_absolute' if mode=='weighted_capped' else 'multiplicative_absolute',target_mass=target_mass)
        if mode=='weighted_nominal':
            self.means=self.H.copy()
            self.variance=np.full(self.K,self.sigma2)
            self.calibration=dict(method='nominal_frozen_candidate_means',variance=self.sigma2,
                guards='Original multiplicative-mean increment guards remain; they do not alter the absolute nominal mean')
            for c in self.absolute:
                k=c['pair']
                c.update(Q=self.q[k],a=self.a[k],rho0=0.,c=c['h']+self.c[k],ratio=1.)
        self.allocation={}
        for i in self.targets:
            comparisons=[c for c in self.absolute if c['left']==i]
            raw=[]
            for c in comparisons:
                J=self.c[c['pair']]
                sd=np.sqrt(2*c['ratio']*J)
                raw.append(float(norm.cdf((self.tie-c['c'])/sd)) if sd>0 else float(c['c']<=self.tie))
            raw=np.asarray(raw)
            if not len(raw):
                self.allocation[str(i)]=dict(total=0.,comparisons=[])
                continue
            floor=.001/len(raw)
            fallback=bool(raw.sum()<=1e-300)
            alpha=np.full(len(raw),.1/len(raw)) if fallback else floor+.099*raw/raw.sum()
            for c,a in zip(comparisons,alpha):
                c['alpha']=float(a);c['z']=float(norm.isf(a))
            self.allocation[str(i)]=dict(total=float(alpha.sum()),uniform_fallback=fallback,
                floor_per_comparison=floor,comparisons=[dict(other=c['right'],baseline_failure=float(p),alpha=float(a))
                    for c,p,a in zip(comparisons,raw,alpha)])
            if abs(alpha.sum()-.1)>1e-12 or np.any(alpha<=0) or np.any(alpha>=.5):
                raise ArithmeticError('Invalid prelocked comparison risk allocation')

    def direct(self,u):
        result=SymmetricRecovery.direct(self,u)
        P=self.forecast+self.case.shift_pq_pu(u,self.base_kw)
        actual_mu=[];used_mu=[];failure=[];actual_failure=[];margin=[];bias_changes=[]
        failure_by_target={str(i):0. for i in self.targets}
        actual_by_target={str(i):0. for i in self.targets}
        for c in self.absolute:
            k,i,j=c['pair'],c['left'],c['right']
            d=P@(self.H[i]-self.H[j]).T;bias=P@(self.means[i]-self.H[i]).T
            rho=float(np.sum(bias*d)/self.sigma2)
            mu=c['h']+result['J'][k]+rho;cap=c['h']+result['J'][k]+c['rho0']
            used=min(mu,cap) if self.mode=='capped_absolute' else mu
            sd=float(np.sqrt(2*c['ratio']*result['J'][k]))
            error=float(norm.cdf((self.tie-used)/sd)) if sd>0 else float(used<=self.tie)
            actual_error=float(norm.cdf((self.tie-mu)/sd)) if sd>0 else float(mu<=self.tie)
            actual_mu.append(mu);used_mu.append(used);failure.append(error);actual_failure.append(actual_error)
            margin.append(used-c['z']*sd-self.tie);bias_changes.append(rho-c['rho0'])
            failure_by_target[str(i)]+=error;actual_by_target[str(i)]+=actual_error
        valid_pair=all(p<=c['alpha']+1e-9 for p,c in zip(failure,self.absolute))
        feasible=bool(valid_pair and all(v>=-1e-7 for v in margin) and all(v>=-1e-7 for v in result['guard_gains']))
        result.update(feasible=feasible,actual_mean=np.asarray(actual_mu),certificate_mean=np.asarray(used_mu),
            minimum_pair_margin=min(margin) if margin else None,pair_failure=np.asarray(failure),
            actual_mean_pair_failure=np.asarray(actual_failure),target_failure_sums=failure_by_target,
            actual_mean_target_failure_sums=actual_by_target,
            worst_recovery_lower=max(0.,1-max(failure_by_target.values(),default=0.)),
            bias_changes=bias_changes,cap_active_count=sum(a>b+1e-10 for a,b in zip(actual_mu,used_mu)))
        return result


def plan(case,score,validation,forecast,base_kw,*,mode='weighted_nominal',seconds=12.,env,
         seed_points=None,target_mass=.99):
    import gurobipy as gp
    began=perf_counter()
    evidence=WeightedRecovery(case,score,validation,forecast,base_kw,mode,target_mass)
    zero=np.zeros(case.shape);baseline=evidence.verify_expansion(zero)
    common=dict(mode=mode,target_mass=target_mass,targets=evidence.targets,
        target_topology_ids=[score['models'][i]['topology_id'] for i in evidence.targets],
        historical_selected_mass=float(evidence.weights[evidence.targets].sum()),
        candidates=evidence.K,pair_count=len(evidence.pairs),absolute_comparison_count=len(evidence.absolute),
        guard_count=len(evidence.guards),total_comparison_risk=.1,risk_allocation=evidence.allocation,
        risk_rule='Per target:0.001/(K-1) floor +0.099 proportional to baseline comparison failure; uniform if all baseline failures underflow; frozen before optimization',
        baseline=baseline,calibration=evidence.calibration,
        constrained_bank='frozen representative models; native-bank outcomes reported separately',
        confidence_scope='conditional on selected true hypothesis and declared Gaussian mean/variance; given forecast and nominal response; no rejection guarantee',
        selection_scope='Historical cumulative weight target set has no guaranteed truth coverage',
        time_scope='includes calibration, risk allocation, construction, seed auditing and native solve; caller charges shared seed generation separately')
    if baseline['feasible']:
        return dict(common,feasible=True,status='zero_sufficient',source='zero',cost_yuan=0.,lower_bound_yuan=0.,
            relative_gap=0.,global_optimality_verified=True,seconds=perf_counter()-began),zero
    initial_records=[];point=None;warm=None;warm_merit=-np.inf;seed_name=None;tick=perf_counter()
    for name,u in seed_items(seed_points):
        try:
            u=np.asarray(u,float);domain=case.audit(u);info=evidence.verify_expansion(u);fee=case.fee(u)
            valid=bool(domain['passed'] and info['feasible'])
            merit=min(v for v in (info['minimum_pair_margin'],info['minimum_guard_gain']) if v is not None)
            initial_records.append(dict(name=name,resource_feasible=domain['passed'],qcqp_feasible=info['feasible'],cost_yuan=fee,minimum_margin=merit))
            if valid and (point is None or fee<case.fee(point)):
                point=u.copy();seed_name=name
            if domain['passed'] and merit>warm_merit:
                warm=u.copy();warm_merit=merit
        except Exception as exc:initial_records.append(dict(name=name,error=repr(exc),qcqp_feasible=False))
    if point is not None:warm=point.copy()
    seed=None if point is None else point.copy()
    initial=dict(source='caller_common_seed_pool',feasible=seed is not None,name=seed_name,
        cost_yuan=None if seed is None else case.fee(seed),audit_seconds=perf_counter()-tick,candidates=initial_records)
    with gp.Model('weighted_absolute_recovery',env=env) as m:
        for name,value in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=2,FeasibilityTol=1e-9,NumericFocus=1,MIPGap=1e-4).items():m.setParam(name,value)
        u,fee=case.attach(m);distances=[];scales=[]
        warm_info=None if warm is None else evidence.direct(warm)
        for k in range(len(evidence.pairs)):
            J=sum(u[h]@evidence.q[k]@u[h] for h in range(case.H))+2*(evidence.a[k]*u).sum()+evidence.c[k]
            s=m.addVar(lb=0.,ub=float(np.sqrt(evidence.upper[k])*(1+1e-9)),name='pair_distance_'+str(k))
            m.addConstr(J<=s*s);distances.append(J);scales.append(s)
            if warm_info is not None:s.Start=float(np.sqrt(warm_info['J'][k]))
        for c in evidence.absolute:
            k=c['pair'];mu=sum(u[h]@c['Q']@u[h] for h in range(case.H))+2*(c['a']*u).sum()+c['c']
            required=c['z']*np.sqrt(2*c['ratio'])*scales[k]+evidence.tie+1e-8
            m.addConstr(mu>=required)
            if evidence.mode=='capped_absolute':m.addConstr(c['h']+distances[k]+c['rho0']>=required)
        for left,right,K,Q,a in evidence.guards:
            m.addConstr(sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()>=0.)
        m.setObjective(fee)
        if warm is not None:u.Start=warm
        m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began));m.optimize()
        try:
            bound=float(m.ObjBound)
            if not np.isfinite(bound):bound=None
        except (AttributeError,gp.GurobiError):bound=None
        source='verified_initial_retained' if seed is not None else None
        if m.SolCount:
            candidate=np.asarray(u.X)
            if case.audit(candidate)['passed'] and evidence.verify_expansion(candidate)['feasible']:
                if point is None or case.fee(candidate)<case.fee(point)-1e-9:
                    point=candidate;source='native_found' if seed is None else 'native_improved_initial'
        if point is None:
            return dict(common,feasible=False,status='native_infeasible' if m.Status==3 else 'unresolved',
                native_status=int(m.Status),initial=initial,infeasibility_proved=m.Status==3,cost_yuan=None,
                lower_bound_yuan=bound,global_optimality_verified=False,seconds=perf_counter()-began),None
        cost=case.fee(point);conflict=bound is not None and bound>cost+1e-7*max(1.,cost)
        lower=0. if bound is None or conflict else max(0.,min(cost,bound));gap=(cost-lower)/max(cost,1e-12)
        return dict(common,feasible=True,status='feasible',source=source,native_status=int(m.Status),initial=initial,
            cost_yuan=cost,lower_bound_yuan=lower,relative_gap=gap,lower_bound_withdrawn=conflict,
            native_infeasibility_withdrawn=m.Status==3,global_optimality_verified=gap<=1e-4,
            improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
            final=evidence.verify_expansion(point),resource_audit=case.audit(point),seconds=perf_counter()-began),point
