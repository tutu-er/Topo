"""Analytic independent targets, active bounds, bilinear saddle, physical replay."""
from __future__ import annotations
import numpy as np
import pytest
from scipy.linalg import block_diag

from research_experiments.mobile_ev_mle.ac_history_refresh.optimizer import (
    GaussianHistoryProblem,ACHistoryProblem,solve_problem,solve_line_qp)
from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle.symmetric_ac_local import whiten_observations


def prior(mean,cov,lower,upper):
    d=len(mean)
    return dict(mean=np.asarray(mean,float),cov=np.asarray(cov,float),
        A=np.r_[np.eye(d),-np.eye(d)],b=np.r_[upper,-np.asarray(lower)])


def linear_problem(A,y,line,K,physical=None):
    d=len(line['mean'])
    def evaluate(w,with_jacobian=True):
        result=dict(mean=A@w,physical_ok=True if physical is None else bool(physical(w)),
                    physical_failures=[] if physical is None or physical(w) else ['declared_state_domain'])
        if with_jacobian:result.update(J_eta=A[:,:d],J_state=A[:,d:])
        return result
    return GaussianHistoryProblem(evaluate,y,line,K)


def test_linear_model_matches_original_prior_closed_form_and_monotone_objective():
    pytest.importorskip('gurobipy')
    A=np.array([[1.,2.,.2],[.5,-1.,2.],[2.,.5,1.]])
    y=np.array([1.,-.5,.7]);line=prior([.2],[[.7]],[-3.],[3.])
    K=np.array([[.4,.1],[.1,.8]]);problem=linear_problem(A,y,line,K)
    C=block_diag(line['cov'],K);mu=np.array([.2,0.,0.])
    # Direct dense posterior is independent of the Schur implementation.
    target=np.linalg.solve(np.linalg.inv(C)+A.T@A,np.linalg.solve(C,mu)+A.T@y)
    result=solve_problem(problem,[.1],[0.,0.])
    assert result['success'],result
    np.testing.assert_allclose(result['map_w'],target,atol=1e-7,rtol=1e-7)
    assert result['objective']<result['initial_objective']
    assert not result['local_optimality_certified']
    accepted=[s for s in result['trace'] if s.get('alpha',0)>0]
    assert accepted
    for row in accepted:
        assert row['objective_after']<=row['objective_before']
        assert row['linear_relative_residual']<=1e-9
        assert row['qp']['dual_bound']<=row['qp']['primal']+1e-9
        assert row['qp']['primal_dual_gap']<=1e-7*(1+abs(row['qp']['primal']))
    first=result['trace'][0]
    # With affine f and quadratic priors, the full first step is the solution.
    full=np.r_[first['direction_eta'],problem.state_cholesky@first['direction_white_state']]
    np.testing.assert_allclose(full,target-np.array([.1,0,0]),atol=1e-7)


def test_active_line_bound_has_correct_constrained_solution_and_dual_certificate():
    pytest.importorskip('gurobipy')
    line=prior([0.],[[1.]],[-1.],[.2]);A=np.array([[1.,.2]])
    problem=linear_problem(A,np.array([2.]),line,np.eye(1))
    result=solve_problem(problem,[0.],[0.])
    assert result['success'],result
    expected=np.array([.2,.2*(2.-.2)/(1.+.2**2)])
    np.testing.assert_allclose(result['map_w'],expected,atol=2e-7)
    assert result['kkt']['scaled']<=1e-6
    assert len(result['kkt']['active_indices'])>=1
    qp=solve_line_qp(np.array([[2.]]),np.array([-4.]),
                     np.array([[1.],[-1.]]),np.array([.2,1.]))
    assert qp['success']
    assert qp['dual_bound']==pytest.approx(qp['primal'],abs=1e-8)


def test_equivalent_line_units_preserve_the_likelihood_and_stationarity_test():
    """Ohm/milliohm coordinate changes must not change the inferred response."""
    pytest.importorskip('gurobipy')
    A=np.array([[1.,2.],[.5,-1.],[2.,.5]]);y=np.array([1.,-.5,.7])
    results=[]
    for scale in (1.,1000.):
        line=prior([.2*scale],[[.7*scale**2]],[-3.*scale],[3.*scale])
        problem=linear_problem(A@np.diag([1/scale,1.]),y,line,np.eye(1)*.4)
        result=solve_problem(problem,[.1*scale],[0.])
        assert result['success'],result
        assert result['kkt']['scaled']<=1e-6
        assert result['kkt']['gradient_eta'].shape==(1,)
        assert result['kkt']['original_scaled_kkt']['state_scaled']>=0
        results.append((np.asarray(result['map_w'])/np.array([scale,1.]),result))
    np.testing.assert_allclose(results[0][0],results[1][0],atol=1e-9)
    assert results[0][1]['objective']==pytest.approx(results[1][1]['objective'],abs=1e-12)
    np.testing.assert_allclose(results[0][1]['kkt']['column_scale_eta'],
                               1000*results[1][1]['kkt']['column_scale_eta'])


def bilinear_problem():
    def evaluate(w,with_jacobian=True):
        a,z=w;out=dict(mean=np.array([a*z]))
        if with_jacobian:out.update(J_eta=np.array([[z]]),J_state=np.array([[a]]))
        return out
    return GaussianHistoryProblem(evaluate,[2.],prior([0.],[[1.]],[-3.],[3.]),np.eye(1))


def test_bilinear_nonzero_start_reaches_known_minimum_without_linear_model_claim():
    pytest.importorskip('gurobipy')
    result=solve_problem(bilinear_problem(),[.5],[.5])
    assert result['success'],result
    np.testing.assert_allclose(result['map_w'],[1.,1.],atol=3e-6)
    assert result['objective']==pytest.approx(1.5,abs=1e-9)
    assert all(t.get('objective_after',t['objective_before'])<=t['objective_before']+1e-12
               for t in result['trace'])


def test_zero_gradient_bilinear_saddle_is_only_first_order_convergence():
    result=solve_problem(bilinear_problem(),[0.],[0.])
    assert result['converged'] and result['status']=='converged_first_order'
    assert result['objective']==2.
    assert not result['local_optimality_certified']
    # Explicit counterexample: a nearby feasible direction lowers the objective.
    eps=.01
    nearby=.5*(eps**2-2.)**2+eps**2
    assert nearby<result['objective']


def test_physical_boundary_cannot_be_reported_as_unconstrained_MAP():
    pytest.importorskip('gurobipy')
    problem=linear_problem(np.array([[0.,1.]]),np.array([2.]),
        prior([0.],[[1.]],[-1.],[1.]),np.eye(1),physical=lambda w:abs(w[1])<1e-15)
    result=solve_problem(problem,[0.],[0.],dict(max_backtracks=5))
    assert not result['success']
    assert result['status']=='physical_boundary_blocked'
    np.testing.assert_array_equal(result['map_w'],[0.,0.])
    assert result['objective']==2.
    assert len(result['trace'][0]['trials'])==5


def test_ac_adapter_replays_PQV_and_source_Q_and_exact_eta_direction():
    candidate=lk.RadialCandidate.from_supports('star',3,[])
    r=np.array([.003,.005,.007,.009]);x=r*np.array([.2,.25,.3,.35])
    eta=np.c_[np.log(r),np.log(x/r)].ravel()
    p=np.array([[.05,.08,.1],[.07,.06,.12],[.08,.09,.07]])
    q=np.array([[.02,.03,.04],[.03,.025,.05],[.035,.02,.025]])
    sp=np.array([.02,.03,.015]);sq=np.array([.012,.018,.01]);root=np.ones(3)
    phy=lk.ac_predict(candidate,r,x,p,q,sp,sq,root)
    meter=lk.MeasurementModel(.001,.0008,.0001,.001,.001)
    D=lk.Observations(p,q,phy['v'],root,phy['master_p'],phy['master_q'])
    line=prior(eta,np.eye(len(eta))*.25,eta-.3,eta+.3)
    z=np.c_[p,q,sp,sq];K=np.eye(z.size)*.3
    problem=ACHistoryProblem(candidate,D,meter,line,K)
    w=np.r_[eta,z.ravel()];actual=problem.evaluate(w)
    np.testing.assert_allclose(actual['mean'],whiten_observations(D,meter),atol=1e-10)
    assert actual['physical_ok']
    direction=np.zeros(len(w));direction[0]=.7;direction[1]=-.4
    direction[len(eta)+7]=.3  # First-time source Q is an independent physical channel.
    h=1e-6
    plus=problem.evaluate(w+h*direction,with_jacobian=False)['mean']
    minus=problem.evaluate(w-h*direction,with_jacobian=False)['mean']
    np.testing.assert_allclose((plus-minus)/(2*h),actual['Jacobian']@direction,
                               rtol=2e-6,atol=2e-6)
    result=solve_problem(problem,eta,z)
    assert result['success'],result
    assert result['objective']<=result['initial_objective']
    assert result['counters']['ac_calls']>=2
    assert result['J_state'].shape[1]==z.size
