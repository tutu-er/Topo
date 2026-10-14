"""Small analytic aggregation checks; no AC fits or future-policy experiment."""
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import LinearConstraint
from scipy.stats import norm
from .ev_bilinear_support_milp import BilinearPair
from .ev_bilinear_score_exchange_v2 import exchange_bilinear_scores_v2
from .ev_profile_chord_milp import FrozenQuadratic,solve_profile_chord_milp
from .ev_toy_predictive_eig import gaussian_mixture_eig_1d


def distances(epsilon,x):
    return np.array([epsilon**2*x*x,(1-x)**2,(1-(1+epsilon)*x)**2])


@pytest.mark.parametrize('epsilon',[.02,.1,.3,.8])
def test_two_regime_worst_global_and_local_peaks(epsilon):
    left=1/(1+2*epsilon);right=2/(2+epsilon)
    left_value=epsilon**2/(1+2*epsilon)**2
    right_value=epsilon**2/(2+epsilon)**2
    assert min(distances(epsilon,left))==pytest.approx(left_value)
    assert min(distances(epsilon,right))==pytest.approx(right_value)
    assert left_value>right_value
    split=1/(1+epsilon)
    # Independent fixed-point audit of the two exact monotonic branches.
    for x in np.linspace(0,split,19):
        d=distances(epsilon,x)
        assert d[1]>=d[2]-1e-14
        assert min(d)<=left_value+1e-14
    for x in np.linspace(split,1,19):
        d=distances(epsilon,x)
        assert d[0]>=d[2]-1e-14
        assert min(d)<=right_value+1e-14


@pytest.mark.parametrize('epsilon',[.02,.1,.3,.8])
def test_mean_target_has_endpoint_optimum_and_keeps_weak_pair(epsilon):
    quadratic_coefficient=2*(epsilon**2+epsilon+1)/3
    assert quadratic_coefficient>0
    for x in np.linspace(0,1,13):
        closed=(2*(epsilon**2+epsilon+1)*x*x-2*(2+epsilon)*x+2)/3
        assert distances(epsilon,x).mean()==pytest.approx(closed)
        assert closed<=2/3+1e-14
    assert distances(epsilon,0)==pytest.approx([0.,1.,1.])
    assert distances(epsilon,1).mean()==pytest.approx(2*epsilon**2/3)


def test_fixed_gaussian_expected_log_likelihood_ratio_and_variance():
    a,b,sigma=.2,1.1,.3
    def ratio(y):return ((y-b)**2-(y-a)**2)/(2*sigma*sigma)
    delta=(a-b)/sigma
    mean=quad(lambda z:ratio(a+sigma*z)*norm.pdf(z),-np.inf,np.inf)[0]
    variance=quad(lambda z:(ratio(a+sigma*z)-mean)**2*norm.pdf(z),-np.inf,np.inf)[0]
    assert mean==pytest.approx(.5*delta*delta,rel=1e-10)
    assert variance==pytest.approx(delta*delta,rel=1e-10)


def test_bounded_gaussian_profile_minimum_and_expectation_are_strictly_different():
    c=1.
    analytic=2*((1+c*c)*norm.cdf(-c)-c*norm.pdf(c))
    numerical=2*quad(lambda y:(y-c)**2*norm.pdf(y),c,np.inf)[0]
    assert analytic==pytest.approx(numerical,rel=1e-10)
    assert analytic==pytest.approx(.15067956668754157)
    assert analytic<1.  # min E(Y-z)^2 attained at z=0 equals 1


def test_tiny_continuous_milp_bounds_worst_target_and_solves_fixed_mean_target():
    epsilon=.1;con=LinearConstraint([[1.,1.]],1.,1.)
    vectors=[np.array([[epsilon,0.]]),np.array([[0.,1.]]),np.array([[epsilon,-1.]])]
    pairs=[BilinearPair(str(i),[0.],B,np.zeros((1,0)),np.zeros((2,1,0)),[],[])
           for i,B in enumerate(vectors)]
    worst=exchange_bilinear_scores_v2(pairs,[0.,0.],[1.,1.],con,
        score_scale=1.,initial_u=[.5,.5],segments=8,max_segments=64,
        max_iterations=4,absolute_gap=1e-4,relative_gap=0.,solver_seconds=2.)
    truth=epsilon**2/(1+2*epsilon)**2
    assert worst['objective_lower']<=truth+1e-8<=worst['objective_upper']+1e-8
    mean=FrozenQuadratic('all_three_mean',np.zeros(3),np.vstack(vectors)/np.sqrt(3))
    solved=solve_profile_chord_milp([mean],[0.,0.],[1.,1.],con,
        segments=4,time_limit=2.,feasible_incumbents=[[.5,.5]])
    assert solved['u']==pytest.approx([0.,1.],abs=1e-6)
    assert solved['objective_lower']==pytest.approx(2/3,abs=1e-7)
    assert solved['objective_upper']>=2/3-1e-8


def test_saved_analytic_evidence_contains_every_pair():
    path=Path(__file__).resolve().parent/'evidence'/'ev_cross_all'/'diagnostics'/'aggregation_toy.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['AC_fit_calls']==0 and data['independent_AC_histories']==0
    assert data['weights_meaning']=='predeclared task values, not topology truth probabilities'
    for row in data['conditions']:
        assert len(row['worst']['distances'])==len(row['average']['distances'])==3
        assert row['average']['weak_pair_unresolved']
        assert sum(row['average']['u'])==pytest.approx(1.)


def test_proper_eig_is_zero_for_equal_means_and_bounded_by_prior_entropy():
    for prior in ([1/3]*3,[.49,.49,.02],[.49995,.49995,.0001]):
        zero=gaussian_mixture_eig_1d([.2,.2,.2],prior,.05,degree=80)
        assert zero['value_nats']==pytest.approx(0.,abs=1e-13)
        for means in ([0.,.01,.02],[0.,0.,1.]):
            eig=gaussian_mixture_eig_1d(means,prior,.05,degree=160)
            assert -1e-12<=eig['value_nats']<=eig['prior_entropy_nats']+1e-12
            assert not eig['integration_error_certified']


def test_small_noise_limit_gives_group_entropy_when_two_models_stay_identical():
    prior=[1/3]*3
    eig=gaussian_mixture_eig_1d([0.,0.,1.],prior,1e-3,degree=80)
    grouped=np.array([2/3,1/3]);group_entropy=float(-grouped@np.log(grouped))
    assert eig['value_nats']==pytest.approx(group_entropy,abs=1e-12)
    separate=gaussian_mixture_eig_1d([0.,.01,1.],prior,1e-4,degree=80)
    assert separate['value_nats']==pytest.approx(separate['prior_entropy_nats'],abs=1e-12)


def test_declared_prior_changes_order_and_quadrature_agreement_is_only_diagnostic():
    eps=.01;x=1/(1+2*eps)
    worst=[0.,eps*x,1-x];average=[0.,0.,1.]
    ranking=[]
    for prior in ([1/3]*3,[.49,.49,.02],[.49995,.49995,.0001]):
        vals={name:[gaussian_mixture_eig_1d(means,prior,.05,degree=n)['value_nats']
            for n in (40,80,160)] for name,means in [('worst',worst),('average',average)]}
        assert max(abs(v[2]-v[1]) for v in vals.values())<1e-9
        ranking.append(max(vals,key=lambda name:vals[name][2]))
    assert ranking==['average','average','worst']


def test_saved_proper_eig_records_declared_priors_and_all_integration_levels():
    path=Path(__file__).resolve().parent/'evidence'/'ev_cross_all'/'diagnostics'/'aggregation_toy.json'
    eig=json.loads(path.read_text(encoding='utf-8'))['proper_predictive_EIG']
    assert eig['AC_fit_calls']==0 and not eig['integration_difference_is_error_bound']
    assert eig['integration_levels']==[40,80,160]
    assert [r['degree160_preferred'] for r in eig['conditions']]==['average','average','worst']
