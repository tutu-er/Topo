"""Independent physical finite differences and nonlinear explanation checks."""
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from . import nonlinear_pair_audit as audit
except ImportError:
    import nonlinear_pair_audit as audit

from research_experiments.mobile_ev_mle.likelihood import Observations,ac_predict
from research_experiments.mobile_ev_mle.candidates import RadialCandidate
from research_experiments.mobile_ev_mle.adaptive_history import _center
from research_experiments.mobile_ev_mle.simulation import model


def fixture():
    star=RadialCandidate.from_supports('star',3,())
    cherry=RadialCandidate.from_supports('cherry',3,((0,1),))
    p=np.array([[.10,.12,.11],[.11,.13,.09]])
    q=np.array([[.035,.041,.038],[.040,.034,.042]])
    sp=np.array([.10,.12]);sq=np.array([.030,.040]);meter=model(3)
    r=np.array([.005,.030,.030,.030]);x=np.array([.0035,.011,.011,.011])
    physical=ac_predict(star,r,x,p,q,sp,sq)
    history=Observations(p=p+meter.bias_p,q=q+meter.bias_q,v=physical['v']+meter.bias_v,
        master_p=physical['master_p'],master_q=physical['master_q'],root_v=np.ones(2))
    bounds=dict(r=(1e-6,.1),x=(1e-6,.1),p=(-1.,1.),q=(-1.,1.),source=(-.5,.5),voltage=(.85,1.15))
    centres={}
    for candidate,rr,xx in [(star,r,x),(cherry,np.array([.005,.004,.030,.030,.030]),
        np.array([.0035,.002,.011,.011,.011]))]:
        fit=SimpleNamespace(r=rr,x=xx,p=p.copy(),q=q.copy(),source_p=sp.copy(),source_q=sq.copy())
        centre=_center(candidate,fit,history,meter,('p','q','v','master_p','master_q'),bounds,.002)
        centre['id']=candidate.name+'_centre_0';centre['centre_kind']='test_physical_point';centre['optimized']=False
        centres[candidate.name]=[centre]
    context=dict(status='ready',history=history,candidates=[star,cherry],model=meter,
        channels=('p','q','v','master_p','master_q'),centres=centres,
        config=dict(horizon=2,base_kva=20.,q_over_p=0.,line_trust_fraction=.3,source_delta_bound=.05,
                    history_state_radius=None,bounds=bounds),summary={'history_hash':'independent_noiseless_test'})
    context['scenarios']=[dict(name='mean',baseline='mean',source_baseline='mean',tracking_gain=1.,
        source_shift_p=0.,source_shift_q=0.,provenance='test_H_only',p0=p.mean(axis=0)[None,:].repeat(2,axis=0),
        q0=q.mean(axis=0)[None,:].repeat(2,axis=0),root_v=np.ones(2))]
    u=np.array([[6.,0.,0.],[0.,6.,0.]])
    raw=audit.build_frozen_pairs(context,u)[0]
    return context,raw,u


def independently_whiten(physical,meter,channels):
    pieces=[]
    for key in channels:
        value=physical[key]
        if key in ('p','q','v'):value=value+getattr(meter,'bias_'+key)
        sigma=np.broadcast_to(getattr(meter,key+'_sigma'),value.shape)
        pieces.append((value/sigma).ravel())
    return np.concatenate(pieces)


def independent_residual(problem,x):
    # Re-evaluate AC directly; never call the analytic derivative helper or
    # problem.evaluate. Layout declarations alone identify physical variables.
    full=problem.template.copy();full[problem.free]=x
    history,future=[],[];ctx=problem.context
    observed=np.concatenate([(getattr(ctx['history'],k)/np.broadcast_to(
        getattr(ctx['model'],k+'_sigma'),getattr(ctx['history'],k).shape)).ravel() for k in ctx['channels']])
    for centre,layout in zip(problem.centres,problem.md['variable_layout']):
        c=centre['candidate'];m,n=c.m,c.n;beta=centre['beta']+full[slice(*layout['beta'])]
        state=centre['history_state']+full[slice(*layout['history_state'])].reshape(2,2*n+2)
        h=ac_predict(c,beta[:m],beta[m:],state[:,:n],state[:,n:2*n],state[:,2*n],state[:,2*n+1],
                     ctx['history'].root_v)
        history.append(independently_whiten(h,ctx['model'],ctx['channels'])-observed)
        scenario=problem.scenario
        p=scenario['p0']+scenario['tracking_gain']*problem.u/ctx['config']['base_kva']
        q=scenario['q0']+ctx['config']['q_over_p']*scenario['tracking_gain']*problem.u/ctx['config']['base_kva']
        source=np.c_[np.repeat(centre['fit'].source_p.mean(),2),np.repeat(centre['fit'].source_q.mean(),2)]
        source+=full[slice(*layout['future_source'])].reshape(2,2)
        f=ac_predict(c,beta[:m],beta[m:],p,q,source[:,0],source[:,1],scenario['root_v'])
        future.append(independently_whiten(f,ctx['model'],ctx['channels']))
    return np.r_[history[0],history[1],future[0]-future[1]]


def test_full_analytic_jacobian_against_independent_AC_finite_differences():
    context,raw,u=fixture();problem=audit.NonlinearPairProblem(context,raw,u)
    x=np.zeros(len(problem.free));analytic=problem.jacobian(x).toarray()
    eps=2e-7;numeric=np.empty_like(analytic)
    for column in range(len(x)):
        plus=x.copy();minus=x.copy();plus[column]+=eps;minus[column]-=eps
        numeric[:,column]=(independent_residual(problem,plus)-independent_residual(problem,minus))/(2*eps)
    np.testing.assert_allclose(analytic,numeric,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(problem.residual(x),independent_residual(problem,x),rtol=0,atol=1e-10)


def test_shared_lines_and_single_physical_future_source():
    context,raw,u=fixture();problem=audit.NonlinearPairProblem(context,raw,u)
    x=np.zeros(len(problem.free));jac=problem.jacobian(x).toarray();layout=raw['metadata']['variable_layout'][0]
    line=int(np.flatnonzero(problem.free==layout['beta'][0])[0])
    assert np.linalg.norm(jac[:len(problem.observed),line])>0
    assert np.linalg.norm(jac[problem.future_start:,line])>0
    source=int(np.flatnonzero(problem.free==layout['future_source'][0])[0])
    # Future P/Q meter rows are unaffected by source amplitude at node1;
    # future slack and voltage responses change through AC physics.
    assert np.all(jac[problem.future_start:problem.future_start+12,source]==0)
    assert np.linalg.norm(jac[problem.future_start+12:,source])>0
    history_source=int(np.flatnonzero(problem.free==layout['history_state'][0]+6)[0])
    assert np.all(jac[problem.future_start:,history_source]==0)


def test_same_local_box_exact_and_full_physical_expands_it():
    context,raw,u=fixture();local=audit.NonlinearPairProblem(context,raw,u,'same_local')
    np.testing.assert_array_equal(local.lower,raw['lower']);np.testing.assert_array_equal(local.upper,raw['upper'])
    full=audit.NonlinearPairProblem(context,raw,u,'full_physical')
    for index,original in enumerate(local.free):
        new=int(np.flatnonzero(full.free==original)[0])
        assert full.lower[new]<=local.lower[index] and full.upper[new]>=local.upper[index]
    assert full.domain=='full_physical'


def test_full_physical_can_release_originally_fixed_line_coordinates():
    context,raw,u=fixture()
    fixed=audit.build_frozen_pairs(context,u,line_trust_fraction=0.,source_delta_bound=.05)[0]
    local=audit.NonlinearPairProblem(context,fixed,u,'same_local')
    full=audit.NonlinearPairProblem(context,fixed,u,'full_physical')
    beta=[j for layout in fixed['metadata']['variable_layout'] for j in range(*layout['beta'])]
    assert not any(j in local.free for j in beta)
    assert all(j in full.free for j in beta)
    assert len(full.free)==len(local.free)+len(beta)


def test_noiseless_star_cherry_near_collapse_is_feasible_compensation():
    context,raw,u=fixture();problem=audit.NonlinearPairProblem(context,raw,u,'full_physical')
    x=np.zeros(len(problem.free));layout=raw['metadata']['variable_layout'][1]
    centre=problem.centres[1];m=centre['candidate'].m
    internal=next(i for i,s in enumerate(centre['candidate'].supports) if len(s)==2)
    for index in (internal,internal+m):
        full_index=layout['beta'][0]+index;column=int(np.flatnonzero(problem.free==full_index)[0])
        x[column]=1e-6-centre['beta'][index]
    point=problem.evaluate(x)
    assert point['feasible'] and point['loss']<2e-4
    assert point['history_costs'][0]==pytest.approx(0.,abs=1e-15)
    # The positive physical lower bound gives a small distance, not a proof
    # that the declared nonlinear minimum is exactly zero.
    assert point['loss']>0 and point['future_distance_squared']<1e-3


def test_nonlinear_refit_improves_full_domain_without_global_claim():
    context,raw,u=fixture()
    result=audit.solve_nonlinear_pair(context,raw,u,'full_physical',starts=2,max_nfev=120)
    assert result['feasible_witness_available']
    assert result['best']['cost_upper']<.02
    assert len(result['restarts'])==2 and result['ac_predict_calls']>0
    assert result['best']['max_ac_equation_residual']<2e-10
    assert not result['global_optimality_certified'] and result['nonlinear_lower_bound'] is None
    assert all(row['beta_shared_history_future'] and row['source_node']==1
               for row in result['best']['explanations'])


def test_operator_failure_is_kept_for_both_restarts(monkeypatch):
    context,raw,u=fixture()
    def failed(*args,**kwargs):raise RuntimeError('independent injected AC failure')
    monkeypatch.setattr(audit,'mean_and_jacobian',failed)
    result=audit.solve_nonlinear_pair(context,raw,u,starts=2,max_nfev=2)
    assert result['status']=='required_optimization_or_physical_failure'
    assert result['best'] is None and not result['feasible_witness_available']
    assert len(result['restarts'])==2 and all(row['error_type']=='RuntimeError' for row in result['restarts'])
    assert result['ac_predict_calls']==2


def test_short_budget_keeps_feasible_upper_even_without_convergence():
    context,raw,u=fixture();result=audit.solve_nonlinear_pair(context,raw,u,starts=1,max_nfev=1)
    assert result['feasible_witness_available']
    assert not result['success'] and result['status']=='feasible_upper_with_optimization_failure'
    assert result['best']['cost_upper']>=0 and result['best']['physically_feasible']


def test_predictive_inputs_have_no_truth_or_future_observation_argument():
    import inspect
    signature=inspect.signature(audit.solve_nonlinear_pair)
    assert tuple(signature.parameters)==('context','raw_pair','u_kw','domain','starts','max_nfev')
    context,raw,u=fixture();p=audit.NonlinearPairProblem(context,raw,u)
    assert not hasattr(p,'truth') and not hasattr(p,'future_observations')
    np.testing.assert_array_equal(p.nominal[0]['p'],p.nominal[1]['p'])
