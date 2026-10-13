"""Independent two-time/two-site discrete and analytic toy audits. No actual H."""
from pathlib import Path
from types import SimpleNamespace
import itertools,json,os,sys
import numpy as np
import pytest

FROZEN=Path(os.environ.get('FROZEN_TOPO_ROOT',r'C:\Users\23761\.codex\worktrees\36a1\Topo'))
sys.path.insert(0,str(FROZEN))
from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair
from research_experiments.mobile_ev_mle.ev_operations_milp import EVConfig,check_plan
from shared_power_square_upper import (build_shared_upper_model,certify_time_block_support,check_control,
    coefficients,decomposed_cost,model_snapshot,score_from_result,secant_error_bounds,secant_value,upper_master_shared,zero_knots)


def toy_pairs():
    # Same-time columns are deliberately non-orthogonal. Cross-time row support is disjoint.
    B=np.array([[0,0,0,0],[0,0,0,0],[1,2,0,0],[.3,.4,0,0],[0,0,1.5,.5],[0,0,.2,2.]])
    other=B.copy();other[2:4,:2]=[[1.7,.6],[.1,.8]];other[4:,2:]=[[.7,1.4],[.9,.3]]
    return [FrozenPair(name,np.array([5.,-3.,.2,-.1,.4,.1]),matrix,
                       np.array([[1.],[-.5],[.2],[0.],[.1],[-.2]]),np.array([[.4]]),
                       np.array([-1.]),np.array([1.])) for name,matrix in [('first',B),('second',other)]]


def direct_cost(pair,z,u):
    residual=pair.c+pair.B@np.asarray(u).ravel()+pair.A@np.asarray(z)
    regularizer=pair.F@np.asarray(z)
    return float(sum(x*x for x in residual)+sum(x*x for x in regularizer))


def scalar_profile(pair,u):
    # Independent closed form for scalar box nuisance; does not use solve_pair.
    r=pair.c+pair.B@np.asarray(u).ravel();a=pair.A[:,0];f=pair.F[:,0]
    alpha=sum(x*x for x in a)+sum(x*x for x in f)
    z=float(np.clip(-sum(x*y for x,y in zip(a,r))/alpha,pair.lower[0],pair.upper[0]))
    return direct_cost(pair,[z],u)


def one_site_signed_grid():
    local=[np.zeros(2)]
    for site,power in itertools.product(range(2),np.linspace(-2,2,17)):
        if power==0:continue
        u=np.zeros(2);u[site]=power;local.append(u)
    return (np.r_[a,b] for a,b in itertools.product(local,repeat=2))


@pytest.mark.parametrize('segments',[2,4,8])
def test_independent_discrete_decomposition_and_uniform_secant_bound(segments):
    knots=[zero_knots(-2,2,segments) for _ in range(4)]
    for pair in toy_pairs():
        constant,linear,d,_=coefficients(pair,[.3],2,2)
        error=secant_error_bounds(d,knots,2,2)
        assert error['single_ev_physical']<error['general']
        for u in one_site_signed_grid():
            true=direct_cost(pair,[.3],u)
            assert decomposed_cost(pair,[.3],u,2,2)==pytest.approx(true,abs=2e-13)
            surrogate=constant+linear@u+sum(dj*secant_value(uj,k) for dj,uj,k in zip(d,u,knots))
            assert -2e-13<=surrogate-true<=error['single_ev_physical']+2e-13
            # Exact square graph at inactive zero, required for physical bound.
            assert all(secant_value(0,k)==0 for k in knots)


def test_scalar_profile_finite_witness_upper_and_shared_secants_not_lower():
    pairs=toy_pairs();witnesses=[[-1.],[0.],[1.]];knots=[zero_knots(-2,2,4) for _ in range(4)]
    maximum_error=max(secant_error_bounds(coefficients(p,[0],2,2)[2],knots,2,2)['single_ev_physical'] for p in pairs)
    for u in one_site_signed_grid():
        q=min(scalar_profile(p,u) for p in pairs)
        w=min(direct_cost(p,z,u) for p in pairs for z in witnesses)
        s=min(c+l@u+sum(dj*secant_value(uj,k) for dj,uj,k in zip(d,u,knots))
              for p in pairs for z in witnesses for c,l,d,_ in [coefficients(p,z,2,2)])
        assert q<=w+2e-13<=s+4e-13
        assert -2e-13<=s-w<=maximum_error+2e-13


def test_exact_support_rejects_cross_time_and_tiny_nonzero():
    B=toy_pairs()[0].B.copy();B[2,2]=1e-300
    with pytest.raises(ValueError,match='non-block'):certify_time_block_support(B,2,2)
    # Cancellation could give an orthogonal Gram matrix; conservative support check still rejects.
    with pytest.raises(ValueError,match='non-block'):certify_time_block_support([[1,0,1,0],[1,0,-1,0]],2,2)


def test_nonorthogonal_same_time_columns_need_single_ev_and_multi_ev_rejected():
    pair=toy_pairs()[0]
    assert float(pair.B[:,0]@pair.B[:,1])!=0
    with pytest.raises(ValueError,match='at most one'):check_control([[1,1],[0,0]],2,2)
    with pytest.raises(ValueError,match='single-EV'):certify_time_block_support(pair.B,2,2,fleet_size=2)
    with pytest.raises(ValueError,match='single-EV'):build_shared_upper_model(EVConfig(n_slots=2,n_sites=2),[pair],[[[0.]]],fleet_size=2)
    u=np.array([1,1,0,0]);c,l,d,_=coefficients(pair,[0],2,2)
    assert abs(direct_cost(pair,[0],u)-(c+l@u+d@(u*u)))>1


def test_zero_nodes_signed_asymmetric_and_odd_signed_segments_rejected():
    assert np.array_equal(zero_knots(-2,2,4),[-2,-1,0,1,2])
    assert 0 in zero_knots(-1,3,4)
    assert 0 in zero_knots(0,2,3)
    assert secant_value(0,zero_knots(-1,3,4))==0
    with pytest.raises(ValueError,match='even'):zero_knots(-2,2,3)


def toy_operations_config():
    return EVConfig(n_slots=2,n_sites=2,p_max_kw=2,p_discharge_max_kw=2,dt_hours=1,
                    net_grid_energy_kwh=2,battery_capacity_kwh=20,initial_soc=.5,soc_min=0,soc_max=1,
                    eta_charge=1,eta_discharge=1,travel_slots=1,travel_energy_kwh=0)


def operation_discrete_audit_actions():
    # Independent route enumeration: stay site 0 twice, or travel at t0 then charge site 1 at t1.
    actions=[np.array([[x,0],[2-x,0]]) for x in np.linspace(0,2,9)]
    actions.append(np.array([[0.,0.],[0.,2.]]))
    return actions


def test_toy_operation_model_best_bound_and_snapshot_stability():
    cfg=toy_operations_config();pairs=toy_pairs();witnesses=[[np.array([-1.]),np.array([0.]),np.array([1.])] for _ in pairs]
    built=build_shared_upper_model(cfg,pairs,witnesses,segments=4)
    again=build_shared_upper_model(cfg,pairs,witnesses,segments=4)
    assert built['snapshot_sha256']==again['snapshot_sha256']
    json.dumps(built['snapshot'],allow_nan=False)
    assert built['shared_graph_variables']==4*(3*4+1)
    assert built['n_variables']==built['operation_variables']+built['shared_graph_variables']+1
    assert len(built['secant_errors'])==6
    result=upper_master_shared(cfg,pairs,witnesses,segments=4,time_limit=5)
    assert result['result'].status==0 and result['q_proxy_upper'] is not None
    assert result['q_proxy_upper']>=result['q_incumbent']-1e-8
    assert check_plan(result['plan'],cfg,tolerance=1e-6)['feasible']
    assert all(np.count_nonzero(abs(row)>1e-7)<=1 for row in result['plan'].u_kw)
    discrete_true=max(min(scalar_profile(p,u) for p in pairs) for u in operation_discrete_audit_actions())
    assert result['q_proxy_upper']>=discrete_true-1e-8
    true_at_selected=min(scalar_profile(p,result['plan'].u_kw) for p in pairs)
    assert result['q_incumbent']>=true_at_selected-1e-8
    assert result['incumbent_is_full_proxy_upper'] is False


def test_incumbent_without_global_bound_is_never_promoted_to_proxy_upper():
    result=score_from_result(SimpleNamespace(objective_value=7.,objective_bound=None),100)
    assert result=={'q_incumbent':700.,'q_proxy_upper':None}
    result=score_from_result(SimpleNamespace(objective_value=7.,objective_bound=9.),100)
    assert result=={'q_incumbent':700.,'q_proxy_upper':900.}
    assert score_from_result(SimpleNamespace(objective_value=7.,objective_bound=np.inf),100)['q_proxy_upper'] is None


def test_fixed_zero_coordinates_and_infeasible_witness_rejected():
    pair=toy_pairs()[0]
    with pytest.raises(ValueError,match='feasible'):coefficients(pair,[2.],2,2)
    assert np.array_equal(zero_knots(0,0,2),[0])
    assert secant_error_bounds([2],[np.array([0])],1,1)=={'general':0.,'single_ev_physical':0.}
