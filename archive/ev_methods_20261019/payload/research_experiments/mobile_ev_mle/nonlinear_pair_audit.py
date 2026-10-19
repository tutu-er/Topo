"""Full nonlinear AC refitting of a few history-only design explanations.

This diagnostic never changes the frozen/calibrated policy. Its best feasible
cost is an upper bound on a nonlinear explanation infimum, not a global MLE,
design-benefit lower bound, or topology probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import inspect
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

try:
    from .adaptive_history import build_history_context,build_frozen_pairs,_predict_path
    from .symmetric_ac_local import mean_and_jacobian,whiten_observations
    from .ev_information_milp import FrozenPair,solve_pair
    from .adaptive_domain_audit import load_history_actions,complete_bank,history_seed,history_input_hash
    from .adaptive_result_audit import _dump,_csv
    from .simulation import model
except ImportError:
    from research_experiments.mobile_ev_mle.adaptive_history import build_history_context,build_frozen_pairs,_predict_path
    from research_experiments.mobile_ev_mle.symmetric_ac_local import mean_and_jacobian,whiten_observations
    from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair,solve_pair
    from adaptive_domain_audit import load_history_actions,complete_bank,history_seed,history_input_hash
    from adaptive_result_audit import _dump,_csv
    from research_experiments.mobile_ev_mle.simulation import model


METHODS=('constant_fixed','refit_milp','fixed_mle_milp')
DOMAINS=('same_local','full_physical')


def _hash(*arrays):
    digest=hashlib.sha256()
    for array in arrays:
        array=np.ascontiguousarray(array,dtype='<f8')
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def source_hashes():
    functions=(build_history_context,build_frozen_pairs,mean_and_jacobian,whiten_observations,
               solve_pair,load_history_actions,model)
    files={Path(inspect.getfile(function)) for function in functions}
    files.add(Path(__file__))
    files.add(Path(inspect.getfile(type(complete_bank()[0]))))
    actual_ac=mean_and_jacobian.__globals__.get('ac_predict')
    if actual_ac is not None:files.add(Path(inspect.getfile(actual_ac)))
    return {file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(files)}


class NonlinearPairProblem:
    """Exact residual and analytic Jacobian in original delta coordinates."""
    def __init__(self,context,raw_pair,u_kw,domain='same_local'):
        if domain not in DOMAINS:raise ValueError('unsupported nonlinear explanation domain')
        self.context,self.raw,self.domain=context,raw_pair,domain
        self.md=raw_pair['metadata'];cfg=context['config'];self.cfg=cfg
        self.u=np.array(u_kw,float,copy=True).reshape(cfg['horizon'],context['history'].n)
        if not np.isfinite(self.u).all() or np.any(self.u < -2e-6):
            raise ValueError('charging-only audit permits only declared 2e-6kW numerical tolerance')
        if np.asarray(raw_pair['F']).size:
            raise ValueError('this audit requires actual H residuals in c/A and an empty F')
        if len(self.md['candidate_names'])!=2:
            raise ValueError('exactly two candidate explanations are required')
        self.scenario=context['scenarios'][self.md['scenario_index']]
        self.centres=[context['centres'][name][index] for name,index in
                      zip(self.md['candidate_names'],self.md['centre_indices'])]
        self.observed=whiten_observations(context['history'],context['model'],context['channels'])
        self.full_count=int(self.md['full_variable_count'])
        self.original_free=np.asarray(self.md['free_indices'],int)
        fixed=np.asarray(self.md['fixed_values'],float)
        if fixed.shape!=(self.full_count,) or len(np.unique(self.original_free))!=len(self.original_free):
            raise ValueError('invalid original coordinate map')
        full_scale=np.full(self.full_count,.02)
        full_scale[self.original_free]=np.asarray(raw_pair['scale'],float)
        physical_lower=np.full(self.full_count,np.nan);physical_upper=np.full(self.full_count,np.nan)
        self.nominal=[]
        for centre,layout in zip(self.centres,self.md['variable_layout']):
            candidate=centre['candidate'];n=candidate.n;T=cfg['horizon']
            beta,state=centre['beta'],centre['history_state']
            a,b=layout['beta']
            if b-a!=2*candidate.m:raise ValueError('line-coordinate layout mismatch')
            physical_lower[a:b]=centre['beta_physical_lower']-beta
            physical_upper[a:b]=centre['beta_physical_upper']-beta
            full_scale[a:b]=np.maximum(beta,1e-4)
            a,b=layout['history_state']
            if b-a!=state.size:raise ValueError('historical state-coordinate layout mismatch')
            physical_lower[a:b]=centre['state_physical_lower'].ravel()-state.ravel()
            physical_upper[a:b]=centre['state_physical_upper'].ravel()-state.ravel()
            p=self.scenario['p0']+self.scenario['tracking_gain']*self.u/cfg['base_kva']
            q=self.scenario['q0']+cfg['q_over_p']*self.scenario['tracking_gain']*self.u/cfg['base_kva']
            sp=_predict_path(centre['fit'].source_p,T,self.scenario['source_baseline'])+self.scenario['source_shift_p']
            sq=_predict_path(centre['fit'].source_q,T,self.scenario['source_baseline'])+self.scenario['source_shift_q']
            self.nominal.append(dict(p=p,q=q,source=np.c_[sp,sq]))
            a,b=layout['future_source']
            if b-a not in (0,2*T):raise ValueError('future source-coordinate layout mismatch')
            if domain=='full_physical' and b-a==0:
                raise ValueError('full-physical source refit requires raw local_refit source coordinates')
            if b>a:
                baseline=np.c_[sp,sq].ravel()
                physical_lower[a:b]=cfg['bounds']['source'][0]-baseline
                physical_upper[a:b]=cfg['bounds']['source'][1]-baseline
        if np.isnan(physical_lower).any() or np.isnan(physical_upper).any():
            raise ValueError('unmapped nonlinear coordinates')
        original_lower=fixed.copy();original_upper=fixed.copy()
        original_lower[self.original_free]=raw_pair['lower'];original_upper[self.original_free]=raw_pair['upper']
        if np.any(original_lower<physical_lower-1e-13) or np.any(original_upper>physical_upper+1e-13):
            raise ValueError('original local box lies outside the declared physical coordinate domain')
        if domain=='same_local':
            self.free=self.original_free.copy();self.template=fixed.copy()
            self.lower=np.array(raw_pair['lower'],float);self.upper=np.array(raw_pair['upper'],float)
        else:
            self.free=np.flatnonzero(physical_lower<physical_upper)
            self.template=(physical_lower+physical_upper)/2
            self.lower=physical_lower[self.free];self.upper=physical_upper[self.free]
        if self.lower.shape!=(len(self.free),) or np.any(self.lower>self.upper):
            raise ValueError('invalid nonlinear box')
        # least_squares needs strict free bounds; preserve equal coordinates.
        unequal=self.lower<self.upper
        self.template[self.free[~unequal]]=self.lower[~unequal]
        self.free=self.free[unequal];self.lower=self.lower[unequal];self.upper=self.upper[unequal]
        self.scale=full_scale[self.free]
        self.history_widths=[len(self.observed)]*2
        self.future_start=2*len(self.observed)
        self.residual_count=self.future_start+cfg['horizon']*sum(
            context['history'].n if channel in ('p','q','v') else 1 for channel in context['channels'])
        self.cache=None;self.best=None;self.ac_calls=0;self.evaluations=0
        self.current_origin='unlabelled_evaluation'
        self.feasible_evaluations=0;self.infeasible_evaluations=0

    def full(self,x):
        x=np.asarray(x,float)
        if x.shape!=(len(self.free),) or not np.isfinite(x).all():raise ValueError('invalid nonlinear point')
        full=self.template.copy();full[self.free]=x
        return full

    def from_original(self,z):
        full=np.asarray(self.md['fixed_values'],float).copy()
        full[self.original_free]=np.asarray(z,float)
        return np.clip(full[self.free],self.lower,self.upper)

    def _physical_feasible(self,physical):
        bounds=self.cfg['bounds'];v=abs(physical['voltage'])
        valid=bool(v.min()>=bounds['voltage'][0] and v.max()<=bounds['voltage'][1])
        for key,bound in (('p','p'),('q','q'),('source_p','source'),('source_q','source')):
            valid &= bool(physical[key].min()>=bounds[bound][0] and physical[key].max()<=bounds[bound][1])
        return valid

    def evaluate(self,x):
        x=np.asarray(x,float)
        if self.cache is not None and np.array_equal(x,self.cache['x']):return self.cache
        full=self.full(x);H=[];N=[];geometries=[]
        for centre,layout,nominal in zip(self.centres,self.md['variable_layout'],self.nominal):
            candidate=centre['candidate'];n,m=candidate.n,candidate.m
            beta=centre['beta']+full[slice(*layout['beta'])]
            state=centre['history_state']+full[slice(*layout['history_state'])].reshape(centre['history_state'].shape)
            self.ac_calls+=1
            historic=mean_and_jacobian(candidate,beta[:m],beta[m:],state[:,:n],state[:,n:2*n],
                state[:,2*n],state[:,2*n+1],self.context['history'].root_v,self.context['model'],
                channels=self.context['channels'])
            delta=full[slice(*layout['future_source'])]
            delta=delta.reshape(self.cfg['horizon'],2) if len(delta) else np.zeros_like(nominal['source'])
            source=nominal['source']+delta
            self.ac_calls+=1
            future=mean_and_jacobian(candidate,beta[:m],beta[m:],nominal['p'],nominal['q'],
                source[:,0],source[:,1],self.scenario['root_v'],self.context['model'],
                channels=self.context['channels'])
            H.append(historic['mean']-self.observed);N.append(future['mean'])
            geometries.append(dict(historic=historic,future=future,beta=beta,state=state,source=source))
        residual=np.r_[H[0],H[1],N[0]-N[1]]
        jac=np.zeros((len(residual),self.full_count));offset=0
        for sign,centre,layout,geometry,width in zip((1.,-1.),self.centres,self.md['variable_layout'],geometries,self.history_widths):
            hs=slice(offset,offset+width);offset+=width;ns=slice(self.future_start,len(residual))
            beta,states,source=[slice(*layout[key]) for key in ('beta','history_state','future_source')]
            jac[hs,beta]=geometry['historic']['line_jacobian']
            jac[hs,states]=geometry['historic']['state_jacobian']
            jac[ns,beta]=sign*geometry['future']['line_jacobian']
            if source.stop>source.start:
                n=centre['candidate'].n
                ids=(np.arange(self.cfg['horizon'])[:,None]*(2*n+2)+np.array([2*n,2*n+1])).ravel()
                jac[ns,source]=sign*geometry['future']['state_jacobian'][:,ids]
        feasible=bool(np.all(x>=self.lower) and np.all(x<=self.upper) and all(
            self._physical_feasible(g[key]['physical']) for g in geometries for key in ('historic','future')))
        loss=float(residual@residual);self.evaluations+=1
        self.feasible_evaluations+=int(feasible);self.infeasible_evaluations+=int(not feasible)
        result=dict(x=x.copy(),full=full,residual=residual,jacobian=csr_matrix(jac[:,self.free]),
            origin=self.current_origin,
            loss=loss,feasible=feasible,geometries=geometries,
            history_costs=[float(r@r) for r in H],future_distance_squared=float(np.sum((N[0]-N[1])**2)))
        self.cache=result
        if feasible and (self.best is None or loss<self.best['loss']):self.best=result
        return result

    def residual(self,x):return self.evaluate(x)['residual']
    def jacobian(self,x):return self.evaluate(x)['jacobian']


def _best_summary(problem):
    best=problem.best
    if best is None:return None
    explanations=[];physical=[]
    for centre,geometry in zip(problem.centres,best['geometries']):
        candidate=centre['candidate'];m,n=candidate.m,candidate.n;state=geometry['state']
        explanations.append(dict(candidate=candidate.name,r_pu=geometry['beta'][:m],x_pu=geometry['beta'][m:],
            historical_p=state[:,:n],historical_q=state[:,n:2*n],historical_source_p=state[:,2*n],
            historical_source_q=state[:,2*n+1],future_source_p=geometry['source'][:,0],
            future_source_q=geometry['source'][:,1],future_total_p=geometry['future']['physical']['p'],
            future_total_q=geometry['future']['physical']['q'],source_node=1,
            beta_shared_history_future=True))
        physical.extend(geometry[key]['physical'] for key in ('historic','future'))
    md=problem.md;fixed=np.asarray(md['fixed_values'],float);original=np.zeros(problem.full_count,bool)
    original[problem.original_free]=True
    fixed_unchanged=bool(np.array_equal(best['full'][~original],fixed[~original]))
    reduced=best['full'][problem.original_free]
    original_feasible=fixed_unchanged and bool(np.all(reduced>=problem.raw['lower']) and np.all(reduced<=problem.raw['upper']))
    affine=None;taylor=None
    if fixed_unchanged:
        proxy=problem.raw['c']+problem.raw['B']@problem.u.ravel()+problem.raw['A']@reduced
        affine=float(proxy@proxy);taylor=float(np.linalg.norm(best['residual']-proxy))
    return dict(cost_upper=best['loss'],history_costs=best['history_costs'],
        witness_origin=best['origin'],
        future_distance_squared=best['future_distance_squared'],explanations=explanations,
        full_delta=best['full'],free_delta=best['x'],within_original_local_box=original_feasible,
        affine_cost_at_best_point=affine,Taylor_residual_norm=taylor,
        affine_at_expanded_point_is_original_box_witness=original_feasible,
        max_ac_equation_residual=max(g['ac_equation_residual'] for g in physical),
        max_power_balance_residual=max(g['power_balance_residual'] for g in physical),
        voltage_min=min(float(abs(g['voltage']).min()) for g in physical),
        voltage_max=max(float(abs(g['voltage']).max()) for g in physical),
        physically_feasible=True,nonlinear_infimum_computed=False,global_optimality_certified=False)


def recover_inherited_point(problem,witness):
    """Validate an old feasible explanation and recover all absolute parameters.

    Identical raw-context fingerprints permit exact delta-coordinate reuse;
    absolute beta/H-state/source checks still validate coordinate semantics.
    Original fixed coordinates can therefore become new free coordinates.
    """
    if not isinstance(witness,dict) or witness.get('best') is None:
        raise ValueError('no inherited feasible explanation')
    if witness.get('pair')!=problem.raw['name'] or witness.get('control_hash')!=_hash(problem.u):
        raise ValueError('inherited pair/control context differs')
    if witness.get('historical_inputs_hash')!=problem.md['historical_inputs_hash']:
        raise ValueError('inherited historical context differs')
    best=witness['best']
    if not best.get('physically_feasible',False):raise ValueError('inherited witness was not physically feasible')
    given={item['candidate']:item for item in best['explanations']}
    if set(given)!=set(problem.md['candidate_names']):raise ValueError('inherited candidate labels differ')
    full=problem.template.copy()
    for name,centre,layout,nominal in zip(problem.md['candidate_names'],problem.centres,
                                        problem.md['variable_layout'],problem.nominal):
        old=given[name];candidate=centre['candidate'];m,n=candidate.m,candidate.n
        beta=np.r_[old['r_pu'],old['x_pu']]
        state=np.c_[old['historical_p'],old['historical_q'],old['historical_source_p'],old['historical_source_q']]
        source=np.c_[old['future_source_p'],old['future_source_q']]
        if beta.shape!=(2*m,) or state.shape!=centre['history_state'].shape or source.shape!=nominal['source'].shape:
            raise ValueError('inherited absolute parameter shapes differ')
        if not np.array_equal(old['future_total_p'],nominal['p']) or not np.array_equal(old['future_total_q'],nominal['q']):
            raise ValueError('inherited future prediction differs')
        if old.get('source_node')!=1 or not old.get('beta_shared_history_future'):
            raise ValueError('inherited physical source/line semantics differ')
        full[slice(*layout['beta'])]=beta-centre['beta']
        full[slice(*layout['history_state'])]=(state-centre['history_state']).ravel()
        a,b=layout['future_source']
        if b>a:full[a:b]=(source-nominal['source']).ravel()
        elif not np.array_equal(source,nominal['source']):raise ValueError('fixed future source changed')
    exact_reuse=witness.get('raw_pair_fingerprint')==FrozenPair(**problem.raw).fingerprint()
    if exact_reuse:
        original=np.asarray(best['full_delta'],float)
        if original.shape!=full.shape or not np.allclose(original,full,rtol=0,atol=1e-12):
            raise ValueError('inherited delta and absolute parameters disagree')
        full=original.copy()
    fixed=np.ones(problem.full_count,bool);fixed[problem.free]=False
    if not np.array_equal(full[fixed],problem.template[fixed]):raise ValueError('inherited fixed coordinates differ')
    x=full[problem.free]
    if not np.isfinite(x).all() or np.any(x<problem.lower) or np.any(x>problem.upper):
        raise ValueError('inherited point outside expanded coordinate box')
    return x,dict(source_domain=witness.get('domain'),source_optimizer_success=witness.get('success'),
        source_cost_upper=best['cost_upper'],exact_delta_reuse=exact_reuse,
        absolute_parameters_recovered=True,released_original_fixed_columns=len(set(problem.free)-set(problem.original_free)))


def solve_nonlinear_pair(context,raw_pair,u_kw,domain='same_local',starts=2,max_nfev=300,*,inherited_witness=None):
    """Refit both explanations; save local attempts and a feasible upper bound."""
    if not isinstance(starts,int) or not 1<=starts<=2 or not isinstance(max_nfev,int) or max_nfev<1:
        raise ValueError('one/two declared starts and positive integer max_nfev required')
    began=perf_counter();problem=NonlinearPairProblem(context,raw_pair,u_kw,domain)
    inheritance=dict(status='not_requested',optimizer_success_not_inherited=True)
    if inherited_witness is not None:
        before=problem.ac_calls
        try:
            point,provenance=recover_inherited_point(problem,inherited_witness)
            problem.current_origin='inherited_'+str(provenance['source_domain'])+'_feasible_witness'
            checked=problem.evaluate(point)
            if not checked['feasible']:raise ValueError('inherited point failed current full AC physical checks')
            inheritance.update(provenance,status='feasible_witness_recomputed',
                recomputed_cost_upper=checked['loss'],source_cost_difference=checked['loss']-provenance['source_cost_upper'])
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            inheritance.update(status='inherited_witness_rejected',error_type=type(error).__name__,error=str(error))
        inheritance['ac_predict_calls']=problem.ac_calls-before
    qp=None;warm_failure=None
    try:
        qp=solve_pair(FrozenPair(**raw_pair),u_kw)
        if not qp['success']:warm_failure='warm QP did not converge; feasible point retained'
    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
        warm_failure=str(error)
    zero=problem.from_original(np.zeros(len(problem.original_free)))
    initial=[('QP_witness',problem.from_original(qp['z']) if qp is not None else zero),('zero_adjustment',zero)][:starts]
    restarts=[]
    for index,(kind,x0) in enumerate(initial):
        problem.current_origin=kind+'_restart_'+str(index)
        before_calls=problem.ac_calls;before_eval=problem.evaluations;before_bad=problem.infeasible_evaluations
        row=dict(restart=index,start_kind=kind,initial_hash=_hash(x0),duplicate_previous_start=bool(
            index and np.array_equal(x0,initial[0][1])),success=False)
        try:
            start=problem.evaluate(x0);row['initial_cost']=start['loss'];row['initial_physically_feasible']=start['feasible']
            if len(problem.free):
                solved=least_squares(problem.residual,x0,jac=problem.jacobian,
                    bounds=(problem.lower,problem.upper),x_scale=problem.scale,
                    method='trf',tr_solver='lsmr',max_nfev=max_nfev,
                    ftol=1e-9,xtol=1e-9,gtol=1e-7)
                final=problem.evaluate(solved.x)
                gradient=2*np.asarray(final['jacobian'].T@final['residual']).ravel()
                row.update(success=bool(solved.success and final['feasible']),optimizer_success=bool(solved.success),
                    physical_feasible=final['feasible'],final_cost=final['loss'],status=int(solved.status),
                    message=str(solved.message),nfev=int(solved.nfev),njev=int(solved.njev),
                    scipy_optimality=float(solved.optimality),gradient_inf_norm=float(np.max(abs(gradient),initial=0.)),
                    active_bound_count=int(np.count_nonzero(solved.active_mask)))
            else:
                row.update(success=start['feasible'],optimizer_success=True,physical_feasible=start['feasible'],
                    final_cost=start['loss'],status=3,message='no free nonlinear coordinates',nfev=1,njev=0,
                    scipy_optimality=0.,gradient_inf_norm=0.,active_bound_count=0)
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
            row.update(error_type=type(error).__name__,error=str(error),status='optimization_or_AC_failure')
        row.update(ac_predict_calls=problem.ac_calls-before_calls,evaluations=problem.evaluations-before_eval,
                   physically_infeasible_evaluations=problem.infeasible_evaluations-before_bad)
        restarts.append(row)
    best=_best_summary(problem);converged=any(row['success'] for row in restarts)
    return dict(pair=raw_pair['name'],domain=domain,status='local_optimization_complete' if converged else
        'feasible_upper_with_optimization_failure' if best is not None else 'required_optimization_or_physical_failure',
        success=converged,feasible_witness_available=best is not None,best=best,restarts=restarts,
        warm_QP_failure=warm_failure,proxy_QP_lower=None if qp is None else qp['q_lower'],
        inherited_feasible_witness=inheritance,historical_inputs_hash=problem.md['historical_inputs_hash'],
        raw_pair_fingerprint=FrozenPair(**raw_pair).fingerprint(),
        proxy_QP_upper=None if qp is None else qp['q_upper'],
        domain_hash=_hash(problem.lower,problem.upper,problem.free),control_hash=_hash(problem.u),
        variable_count=len(problem.free),residual_count=problem.residual_count,
        source_node=1,shared_history_future_lines=True,future_total_pq_fixed_by_common_forecast_and_EV=True,
        future_source_nominal_frozen_from_H_centre=True,historical_constants_retained=True,
        objective='two historical AC costs plus future whitened mean difference',
        historical_data_are_not_two_independent_samples=True,
        ac_predict_calls=problem.ac_calls,evaluations=problem.evaluations,
        physically_infeasible_evaluations=problem.infeasible_evaluations,seconds=perf_counter()-began,
        global_optimality_certified=False,nonlinear_lower_bound=None,topology_probability=False,
        primary_frozen_policy_modified=False)


def audit_history(history,actions,*,starts=2,max_nfev=300,extra_pairs=2):
    began=perf_counter();horizon=len(actions[METHODS[0]]['u_kw']);u0=np.zeros((horizon,3));u0[:,0]=3.
    context=build_history_context(history,complete_bank(),model(3),horizon=horizon,max_centers=2,
        max_attempts_per_candidate=3,max_nfev=300,multistart=2,line_trust_fraction=.3,
        source_delta_bound=.05,center_loss_slack=12.,seed=history_seed(history))
    result=dict(history_hash=history_input_hash(history),history_context=context['summary'],records=[],
        selector_uses_truth=False,selector_uses_future_noise=False,selector_uses_evaluation_seed=False,
        primary_frozen_policy_modified=False,scope='six topology pairs at centre0, plus H-only proxy-active extra centres')
    if context['status']!='ready':
        result.update(status='required_history_failure',seconds=perf_counter()-began);return result
    raw=build_frozen_pairs(context,u0,line_trust_fraction=.3,source_delta_bound=.05)
    mandatory=[i for i,row in enumerate(raw) if row['metadata']['centre_indices']==[0,0]]
    if len(mandatory)!=6:raise ValueError('all six centre0 topology pairs must be preserved')
    choices=set(mandatory);activity=[];extra_costs={}
    for method in METHODS:
        values=[]
        for i,row in enumerate(raw):
            try:
                answer=solve_pair(FrozenPair(**row),actions[method]['u_kw'])
                values.append((answer['q_upper'],i,answer['success']))
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                activity.append(dict(method=method,pair=row['name'],status='required_proxy_failure',error=str(error)))
                values.append((np.inf,i,False))
        ordered=sorted(values)
        for cost,index,success in ordered:
            if index not in choices and success:
                extra_costs[index]=min(cost,extra_costs.get(index,np.inf))
        for cost,index,success in ordered[:extra_pairs]:
            activity.append(dict(method=method,pair=raw[index]['name'],proxy_primal=cost,
                success=success,role='proxy activity; partial diagnostic never excludes an unreviewed competitor'))
    choices.update(index for index,cost in sorted(extra_costs.items(),key=lambda item:(item[1],item[0]))[:extra_pairs])
    result['diagnostic_pair_names']=[raw[i]['name'] for i in sorted(choices)]
    result['omitted_pair_names']=[row['name'] for i,row in enumerate(raw) if i not in choices]
    result['partial_centre_family_scope']=len(choices)<len(raw);result['proxy_activity']=activity
    for method in METHODS:
        for index in sorted(choices):
            local_result=None
            for domain in DOMAINS:
                try:
                    row=solve_nonlinear_pair(context,raw[index],actions[method]['u_kw'],domain,starts,max_nfev,
                        inherited_witness=local_result if domain=='full_physical' else None)
                    row['method']=method
                    if domain=='same_local' and row.get('best') is not None:local_result=row
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                    row=dict(method=method,pair=raw[index]['name'],domain=domain,
                        status='required_pair_failure',error_type=type(error).__name__,error=str(error),
                        success=False,feasible_witness_available=False,best=None)
                result['records'].append(row)
    result.update(status='diagnostic_complete_with_failures' if any(not row['success'] for row in result['records'])
                  else 'diagnostic_complete',seconds=perf_counter()-began,
                  all_required_pairs_preserved=True,nonlinear_global_design_bound_computed=False)
    return result


def run_audit(input_folder,out,*,limit=2,starts=2,max_nfev=300,extra_pairs=2):
    out,input_folder=Path(out),Path(input_folder);out.mkdir(parents=True,exist_ok=True)
    if (out/'progress.json').exists():raise ValueError('use a fresh nonlinear audit output directory')
    files=sorted(input_folder.glob('trial_[0-9]*.json'))[:limit];records=[];flat=[]
    if not files:raise ValueError('no completed pilot histories')
    config=dict(limit=limit,starts=starts,max_nfev=max_nfev,extra_pairs=extra_pairs,
        source_sha256=source_hashes(),runtime=dict(python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__),methods=METHODS,domains=DOMAINS)
    _dump(out/'config.json',config)
    for index,file in enumerate(files):
        began=perf_counter()
        try:
            history,actions=load_history_actions(file)
            result=audit_history(history,actions,starts=starts,max_nfev=max_nfev,extra_pairs=extra_pairs)
        except Exception as error:
            result=dict(status='required_history_audit_failure',error_type=type(error).__name__,error=str(error),
                        history_hash=None,records=[],seconds=perf_counter()-began)
        result['input_trial_filename']=file.name
        filename=f'audit_{file.stem}.json';_dump(out/filename,result);records.append(result)
        for row in result['records']:
            best=row.get('best') or {}
            flat.append(dict(history_hash=result['history_hash'],input_trial_filename=file.name,
                method=row['method'],pair=row['pair'],domain=row['domain'],status=row['status'],
                nonlinear_cost_upper=best.get('cost_upper'),history_costs=best.get('history_costs'),
                future_distance_squared=best.get('future_distance_squared'),proxy_QP_lower=row.get('proxy_QP_lower'),
                proxy_QP_upper=row.get('proxy_QP_upper'),Taylor_residual_norm=best.get('Taylor_residual_norm'),
                within_original_local_box=best.get('within_original_local_box'),
                seconds=row.get('seconds'),ac_predict_calls=row.get('ac_predict_calls'),
                nonlinear_lower_bound=None,global_optimality_certified=False))
        _dump(out/'progress.json',dict(completed=len(records),selected_file_count=len(files),last_file=file.name))
        print(json.dumps(dict(history=index+1,count=len(files),status=result['status'],seconds=result['seconds'])),flush=True)
    _csv(out/'nonlinear_explanations.csv',flat)
    summary=dict(history_count=len(records),selected_file_count=len(files),methods=METHODS,domains=DOMAINS,
        starts=starts,max_nfev=max_nfev,extra_proxy_active_pairs=extra_pairs,
        selection='first sorted pilot histories; action shortlist fixed before reading outcomes',
        histories=[dict(input_trial_filename=row['input_trial_filename'],history_hash=row['history_hash'],
            status=row['status'],seconds=row['seconds']) for row in records],
        failure_denominator_preserved=True,primary_frozen_policy_modified=False,
        nonlinear_infimum_computed=False,nonlinear_global_design_bound_computed=False,
        config=config)
    _dump(out/'summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--limit',type=int,default=2)
    parser.add_argument('--starts',type=int,default=2);parser.add_argument('--max-nfev',type=int,default=300)
    parser.add_argument('--extra-pairs',type=int,default=2);args=parser.parse_args()
    run_audit(args.input,args.out,limit=args.limit,starts=args.starts,max_nfev=args.max_nfev,extra_pairs=args.extra_pairs)


if __name__=='__main__':main()
