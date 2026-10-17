"""Prior-anchored AC-history Gauss--Newton with a small convex line QP.

State prior is whitened once: z=L v. All accepted steps decrease the original
nonlinear objective. Success denotes first-order numerical convergence, not
global MAP or a second-order local-optimality certificate.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular
from scipy.optimize import nnls

from .. import likelihood as lk
from ..symmetric_ac_local import mean_and_jacobian, whiten_observations
from ..ev_bayes_response_truncation import eta_jacobian, eta_to_rx


DEFAULTS = dict(max_iter=25, armijo=1e-4, beta=.5, max_backtracks=20,
    scaled_kkt_tol=1e-6, relative_obj_tol=1e-8, qp_feasibility_tol=1e-8,
    qp_kkt_tol=1e-6, qp_gap_tol=1e-7, linear_tol=1e-9,
    qp_time_limit=30., source_p_bounds=(-.5,.5), source_q_bounds=(-.5,.5),
    voltage_bounds=(.85,1.15))


def _options(options):
    given={} if options is None else dict(options)
    if set(given)-set(DEFAULTS):
        raise ValueError('unknown MAP options: '+str(sorted(set(given)-set(DEFAULTS))))
    result={**DEFAULTS,**given}
    for key in ('max_iter','max_backtracks'):
        if isinstance(result[key],bool) or int(result[key])!=result[key] or result[key]<1:
            raise ValueError(key+' must be a positive integer')
    for key in ('armijo','beta'):
        if not 0<float(result[key])<1:raise ValueError(key+' must be in (0,1)')
    for key in ('scaled_kkt_tol','relative_obj_tol','qp_feasibility_tol',
                'qp_kkt_tol','qp_gap_tol','linear_tol','qp_time_limit'):
        if not np.isfinite(result[key]) or result[key]<=0:raise ValueError(key+' must be positive finite')
    return result


def line_constraints(prior):
    """Return A eta <= h; original physical boxes become linear in eta."""
    mean=np.asarray(prior['mean'],float);d=len(mean)
    if 'A' in prior and 'b' in prior:
        A=np.asarray(prior['A'],float);h=np.asarray(prior['b'],float)
    else:
        if d%2:raise ValueError('physical eta must have an even dimension')
        bounds=prior['bounds'];rows=[];rhs=[]
        for e in range(d//2):
            for indices,key in (((2*e,),'log_R'),((2*e+1,),'log_ratio'),
                                ((2*e,2*e+1),'log_X')):
                row=np.zeros(d);row[list(indices)]=1
                lo,hi=np.asarray(bounds[key],float)[e]
                rows.extend((row.copy(),-row));rhs.extend((hi,-lo))
        A,h=np.asarray(rows),np.asarray(rhs)
    if (A.ndim!=2 or A.shape[1]!=d or h.shape!=(len(A),)
            or not np.isfinite(A).all() or not np.isfinite(h).all()):
        raise ValueError('finite compatible eta polytope required')
    return A,h


class GaussianHistoryProblem:
    """Generic measurement evaluator plus original Gaussian prior."""
    def __init__(self,evaluator,observed,line_prior,process_covariance,*,latent_shape=None):
        self.evaluator=evaluator;self.observed=np.asarray(observed,float).copy()
        self.line_prior=deepcopy(line_prior)
        self.line_dimension=len(line_prior['mean'])
        self.line_mean=np.asarray(line_prior['mean'],float).copy()
        self.line_covariance=np.asarray(line_prior['cov'],float).copy()
        self.process_covariance=np.asarray(process_covariance,float).copy()
        self.A,self.h=line_constraints(line_prior)
        d=self.line_dimension;s=len(self.process_covariance)
        if (self.line_covariance.shape!=(d,d) or self.process_covariance.shape!=(s,s)
                or not all(np.isfinite(a).all() for a in
                    (self.observed,self.line_mean,self.line_covariance,self.process_covariance))):
            raise ValueError('finite compatible original prior required')
        if not np.allclose(self.line_covariance,self.line_covariance.T,atol=1e-13,rtol=1e-12):
            raise ValueError('line covariance must be symmetric')
        if not np.allclose(self.process_covariance,self.process_covariance.T,atol=1e-13,rtol=1e-12):
            raise ValueError('process covariance must be symmetric')
        self.line_factor=cho_factor(self.line_covariance,lower=True)
        self.line_precision=cho_solve(self.line_factor,np.eye(d))
        self.state_cholesky=np.linalg.cholesky(self.process_covariance)
        self.latent_shape=(s,) if latent_shape is None else tuple(latent_shape)
        if np.prod(self.latent_shape)!=s:raise ValueError('latent shape/prior dimension disagree')
        self.prior_mean=np.r_[self.line_mean,np.zeros(s)]
        self.counters=dict(evaluations=0,Jacobian_calls=0)
        self._prior_covariance=None

    @property
    def prior_covariance(self):
        if self._prior_covariance is None:
            self._prior_covariance=block_diag(self.line_covariance,self.process_covariance)
        return self._prior_covariance

    def evaluate(self,w,with_jacobian=True):
        self.counters['evaluations']+=1
        self.counters['Jacobian_calls']+=int(with_jacobian)
        result=self.evaluator(np.asarray(w,float),with_jacobian=with_jacobian)
        result=dict(result)
        if with_jacobian and 'Jacobian' not in result:
            result['Jacobian']=np.c_[result['J_eta'],result['J_state']]
        result.setdefault('physical_ok',True);result.setdefault('physical_failures',[])
        return result


class ACHistoryProblem(GaussianHistoryProblem):
    """Original P/Q/V meter likelihood with latent P/Q and same-interface source."""
    def __init__(self,candidate,history,meter,line_prior,process_covariance,options=None):
        if meter.fit_bias or meter.calibration is not None:
            raise ValueError('known independently calibrated biases required')
        if history.master_p is None or history.master_q is None:
            raise ValueError('source P/Q require master P/Q anchors')
        self.candidate=candidate;self.history=history;self.meter=meter
        self.options=_options(options);self.ac_calls=0;self.ac_states=0
        d=len(line_prior['mean']);shape=(history.t,2*candidate.n+2)
        if d!=2*candidate.m:raise ValueError('candidate and original line-prior dimensions disagree')
        observed=whiten_observations(history,meter)
        super().__init__(self._AC_evaluate,observed,line_prior,process_covariance,latent_shape=shape)

    def _AC_evaluate(self,w,with_jacobian=True):
        d=self.line_dimension;n=self.candidate.n
        r,x=eta_to_rx(w[:d]);z=w[d:].reshape(self.latent_shape)
        self.ac_calls+=1;self.ac_states+=self.history.t
        out=mean_and_jacobian(self.candidate,r,x,z[:,:n],z[:,n:2*n],z[:,-2],z[:,-1],
            self.history.root_v,self.meter,with_jacobian=with_jacobian)
        phy=out['physical'];voltage=abs(phy['voltage']);fail=[]
        lo,hi=self.options['voltage_bounds']
        if voltage.min()<lo or voltage.max()>hi:fail.append('voltage_domain')
        for j,name in ((-2,'source_p'),(-1,'source_q')):
            lo,hi=self.options[name+'_bounds']
            if np.any(z[:,j]<lo) or np.any(z[:,j]>hi):fail.append(name+'_domain')
        result=dict(mean=out['mean'],physical=phy,physical_ok=not fail,physical_failures=fail)
        if with_jacobian:
            Jeta=out['line_jacobian']@eta_jacobian(r,x,ordering='stacked')
            result.update(J_eta=Jeta,J_state=out['state_jacobian'],
                          Jacobian=np.c_[Jeta,out['state_jacobian']])
        return result


def _objective(problem,eta,v,response):
    residual=np.asarray(response['mean'])-problem.observed
    delta=eta-problem.line_mean
    terms=dict(measurement=.5*float(residual@residual),
        line_prior=.5*float(delta@problem.line_precision@delta),
        process_prior=.5*float(v@v))
    return sum(terms.values()),terms,residual


def _current_kkt(problem,eta,g_eta,g_v,scale_eta,scale_v,tolerance,
                 legacy_eta_scale,legacy_v_scale):
    """KKT in explicit local GN column units, retaining the original audit.

    D_j=sqrt(G_jj) changes numerical variable units, not stationary points.
    Physical feasibility and complementarity remain separately checked.
    """
    slack=problem.h-problem.A@eta
    active=np.flatnonzero(slack<=10*tolerance*(1+abs(problem.h)))
    dual=np.zeros(len(slack))
    if len(active):
        dual[active]=nnls(problem.A[active].T/scale_eta[:,None],
                          -g_eta/scale_eta,maxiter=1000)[0]
    stationary=g_eta+problem.A.T@dual
    line=float(np.max(abs(stationary)/scale_eta,initial=0))
    state=float(np.max(abs(g_v)/scale_v,initial=0))
    complement=float(np.max(abs(dual*slack),initial=0)/
                     (1+np.max(abs(dual),initial=0)*np.max(abs(problem.h),initial=0)))
    legacy_dual=np.zeros(len(slack))
    if len(active):legacy_dual[active]=nnls(problem.A[active].T,-g_eta,maxiter=1000)[0]
    legacy_line=float(np.max(abs(g_eta+problem.A.T@legacy_dual),initial=0)/
        (legacy_eta_scale+np.max(abs(problem.A.T@legacy_dual),initial=0)))
    legacy_state=float(np.max(abs(g_v),initial=0)/legacy_v_scale)
    return dict(scaled=max(line,state,complement),line_scaled=line,state_scaled=state,
        complementarity_scaled=complement,dual=dual,active_indices=active,
        eta_primal_violation=float(np.max(-slack,initial=0)),
        gradient_eta=g_eta.copy(),gradient_white_state=g_v.copy(),
        column_scale_eta=scale_eta.copy(),column_scale_white_state=scale_v.copy(),
        scaling='D_j=sqrt(G_jj), G=prior_precision+J_white.T@J_white',
        original_scaled_kkt=dict(scaled=max(legacy_line,legacy_state,complement),
            line_scaled=legacy_line,state_scaled=legacy_state,dual=legacy_dual,
            scope='retained development-v1 raw-gradient normalization, not the stopping test'))


def solve_line_qp(S,b,A,h,options=None):
    """Continuous strictly convex QP, with independently evaluated dual bound."""
    import gurobipy as gp
    from gurobipy import GRB
    opt=_options(options);began=perf_counter();S=(np.asarray(S)+np.asarray(S).T)/2
    b=np.asarray(b);A=np.asarray(A);h=np.asarray(h);d=len(b)
    chol=cho_factor(S,lower=True)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model(env=env) as model:
            model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.NonConvex=0
            model.Params.FeasibilityTol=1e-9;model.Params.OptimalityTol=1e-9
            model.Params.BarConvTol=1e-11;model.Params.TimeLimit=opt['qp_time_limit']
            x=model.addMVar(d,lb=-GRB.INFINITY,ub=GRB.INFINITY,name='eta_step')
            con=model.addMConstr(A,x,GRB.LESS_EQUAL,h,name='original_eta_domain')
            model.setObjective(.5*x@S@x+b@x,GRB.MINIMIZE)
            model.optimize();status=int(model.Status)
            if model.SolCount<1:
                return dict(success=False,status=status,failure='no_QP_attainment',seconds=perf_counter()-began)
            step=np.asarray(x.X);raw_dual=-np.asarray(con.Pi)
    dual=np.maximum(raw_dual,0.)
    direction=b+A.T@dual;dual_stationary=cho_solve(chol,direction)
    lower=-.5*float(direction@dual_stationary)-float(h@dual)
    primal=.5*float(step@S@step)+float(b@step)
    gradient=S@step+b+A.T@dual
    primal_error=float(np.max(A@step-h,initial=0))
    stat=float(np.max(abs(gradient),initial=0)/(1+np.max(abs(b),initial=0)+np.max(abs(A.T@dual),initial=0)))
    comp=float(np.max(abs(dual*(A@step-h)),initial=0)/
               (1+abs(primal)+np.max(abs(dual),initial=0)*np.max(abs(h),initial=0)))
    gap=primal-lower
    gate=bool(status==GRB.OPTIMAL and primal_error<=opt['qp_feasibility_tol']
        and stat<=opt['qp_kkt_tol'] and comp<=opt['qp_kkt_tol']
        and -opt['qp_gap_tol']*(1+abs(primal))<=gap<=opt['qp_gap_tol']*(1+abs(primal)))
    return dict(success=gate,status=status,step=step,dual=dual,raw_dual=raw_dual,
        primal=primal,dual_bound=lower,primal_dual_gap=gap,primal_violation=primal_error,
        stationarity_scaled=stat,complementarity_scaled=comp,
        certificate_scope='floating-point convex-QP primal/dual/KKT audit',
        seconds=perf_counter()-began,failure=None if gate else 'QP_numerical_gate')


def solve_problem(problem,initial_eta,initial_latent,options=None):
    """Solve the prior-anchored objective; all data/prior terms are immutable."""
    opt=_options(options);began=perf_counter();d=problem.line_dimension
    eta=np.asarray(initial_eta,float).copy();z=np.asarray(initial_latent,float).reshape(-1)
    if eta.shape!=(d,) or z.shape!=(len(problem.process_covariance),):
        raise ValueError('initial eta/latent dimensions disagree with prior')
    if not np.isfinite(eta).all() or not np.isfinite(z).all():raise ValueError('finite initial state required')
    if np.max(problem.A@eta-problem.h,initial=0)>opt['qp_feasibility_tol']:
        raise ValueError('initial eta outside original physical domain')
    v=solve_triangular(problem.state_cholesky,z,lower=True)
    trace=[];status='maximum_iterations';success=False;last_relative=np.inf
    counts=dict(objective_evaluations=0,qp_solves=0,accepted_steps=0,rejected_trials=0)
    final=None;initial_objective=None
    for iteration in range(opt['max_iter']+1):
        w=np.r_[eta,problem.state_cholesky@v]
        try:
            counts['objective_evaluations']+=1;response=problem.evaluate(w,with_jacobian=True)
            if not response['physical_ok']:
                status='initial_or_current_physical_failure';final=response;break
            objective,terms,r=_objective(problem,eta,v,response)
            if initial_objective is None:initial_objective=objective
            Jeta=np.asarray(response['J_eta']);Jstate=np.asarray(response['J_state'])
            Jv=Jstate@problem.state_cholesky
            data_eta=Jeta.T@r;prior_eta=problem.line_precision@(eta-problem.line_mean)
            data_v=Jv.T@r;g_eta=data_eta+prior_eta;g_v=data_v+v
            kkt=_current_kkt(problem,eta,g_eta,g_v,
                np.sqrt(np.diag(problem.line_precision)+np.sum(Jeta*Jeta,axis=0)),
                np.sqrt(1+np.sum(Jv*Jv,axis=0)),opt['qp_feasibility_tol'],
                1+np.max(abs(data_eta),initial=0)+np.max(abs(prior_eta),initial=0),
                1+np.max(abs(data_v),initial=0)+np.max(abs(v),initial=0))
            final=dict(response,objective=objective,terms=terms,kkt=kkt)
            if kkt['scaled']<=opt['scaled_kkt_tol'] and (iteration==0 or last_relative<=opt['relative_obj_tol']):
                success=True;status='converged_first_order';break
            if iteration==opt['max_iter']:break
            Hvv=np.eye(len(v))+Jv.T@Jv;Hve=Jv.T@Jeta
            factor=cho_factor(Hvv,lower=True);rhs=np.c_[g_v,Hve]
            solved=cho_solve(factor,rhs)
            relative=float(np.linalg.norm(Hvv@solved-rhs)/max(np.linalg.norm(rhs),1e-30))
            step_record=dict(iteration=iteration,objective_before=objective,terms_before=terms,
                kkt_before=kkt,linear_relative_residual=relative,trials=[])
            if relative>opt['linear_tol']:
                trace.append(step_record);status='state_linear_solve_failed';break
            S=problem.line_precision+Jeta.T@Jeta-Hve.T@solved[:,1:]
            S=(S+S.T)/2;b=g_eta-Hve.T@solved[:,0]
            qp=solve_line_qp(S,b,problem.A,problem.h-problem.A@eta,opt)
            counts['qp_solves']+=1;step_record['qp']=qp
            if not qp['success']:
                trace.append(step_record);status='line_QP_failed';break
            pe=qp['step'];pv=-solved[:,0]-solved[:,1:]@pe
            slope=float(g_eta@pe+g_v@pv)
            response_direction=Jeta@pe+Jv@pv
            quadratic_change=slope+.5*float(pe@problem.line_precision@pe+pv@pv+
                                            response_direction@response_direction)
            step_record.update(direction_eta=pe,direction_white_state=pv,directional_derivative=slope,
                GN_directional_decrement=-slope,GN_predicted_decrease=-quadratic_change,
                GN_quadratic_model_change=quadratic_change)
            if kkt['scaled']<=opt['scaled_kkt_tol'] and (
                    max(np.max(abs(pe),initial=0),np.max(abs(pv),initial=0))<opt['relative_obj_tol']):
                step_record.update(objective_after=objective,terms_after=terms,alpha=0.,relative_objective_change=0.)
                trace.append(step_record);success=True;status='converged_first_order';break
            if not slope<0:
                trace.append(step_record);status='non_descent_direction';break
            alpha=1.;accepted=False
            for backtrack in range(opt['max_backtracks']):
                candidate_eta=eta+alpha*pe;candidate_v=v+alpha*pv
                trial=dict(alpha=alpha,backtrack=backtrack,accepted=False)
                try:
                    trial_w=np.r_[candidate_eta,problem.state_cholesky@candidate_v]
                    counts['objective_evaluations']+=1;pred=problem.evaluate(trial_w,with_jacobian=False)
                    trial.update(physical_ok=pred['physical_ok'],physical_failures=pred['physical_failures'])
                    if pred['physical_ok']:
                        candidate_objective,candidate_terms,_=_objective(problem,candidate_eta,candidate_v,pred)
                        trial.update(objective=candidate_objective,terms=candidate_terms)
                        accepted=bool(candidate_objective<=objective+opt['armijo']*alpha*slope)
                        trial['accepted']=accepted
                except Exception as exc:
                    trial.update(physical_ok=False,error=type(exc).__name__+': '+str(exc))
                step_record['trials'].append(trial)
                if accepted:
                    last_relative=abs(objective-candidate_objective)/(1+abs(objective))
                    eta=candidate_eta;v=candidate_v;counts['accepted_steps']+=1
                    final=dict(pred,objective=candidate_objective,terms=candidate_terms,kkt=None)
                    step_record.update(alpha=alpha,objective_after=candidate_objective,
                        terms_after=candidate_terms,relative_objective_change=last_relative)
                    break
                counts['rejected_trials']+=1;alpha*=opt['beta']
            trace.append(step_record)
            if not accepted:
                status=('physical_boundary_blocked' if all(not t.get('physical_ok') for t in step_record['trials'])
                        else 'Armijo_backtracking_failed');break
        except Exception as exc:
            status='numerical_failure';trace.append(dict(iteration=iteration,error=type(exc).__name__+': '+str(exc)));break
    w=np.r_[eta,problem.state_cholesky@v]
    if final is None or 'objective' not in final:
        objective=None;terms=None;Jeta=Jstate=None;kkt=None
    else:
        # Every accepted point is reevaluated at the next iteration, including
        # the final permitted iteration. Failure never substitutes the surrogate.
        objective=final['objective'];terms=final['terms'];kkt=final.get('kkt')
        Jeta=final.get('J_eta');Jstate=final.get('J_state')
    counts.update(problem.counters)
    if isinstance(problem,ACHistoryProblem):
        counts.update(ac_calls=problem.ac_calls,AC_states=problem.ac_states)
    return dict(eta=eta,latent=w[d:].reshape(problem.latent_shape),map_w=w,
        initial_eta=np.asarray(initial_eta,float).copy(),initial_latent=np.asarray(initial_latent,float).copy(),
        objective=objective,objective_terms=terms,initial_objective=initial_objective,
        success=success,converged=success,status=status,trace=trace,
        J_eta=Jeta,J_state=Jstate,Jacobian=None if Jeta is None else np.c_[Jeta,Jstate],
        kkt=kkt,relative_objective_change=None if not np.isfinite(last_relative) else last_relative,
        prior_info=dict(line_prior=deepcopy(problem.line_prior),
            process_covariance_sha256=sha256(problem.process_covariance.astype('<f8').tobytes()).hexdigest(),
            state_whitening=True,original_prior_reused_once=True,history_likelihood_count=1),
        counters=counts,seconds=perf_counter()-began,options=opt,
        global_optimality_certified=False,local_optimality_certified=False,
        termination_scope='numerical first-order constrained stationarity; saddle points are not excluded',
        physical_failures=[] if final is None else final.get('physical_failures',[]),
        posterior_updated=False)


def fit_map(candidate,D,meter,line_prior,process_covariance,initial_eta,initial_latent,options=None):
    problem=ACHistoryProblem(candidate,D,meter,line_prior,process_covariance,options)
    return solve_problem(problem,initial_eta,initial_latent,options)


__all__=['DEFAULTS','line_constraints','GaussianHistoryProblem','ACHistoryProblem',
         'solve_line_qp','solve_problem','fit_map']
