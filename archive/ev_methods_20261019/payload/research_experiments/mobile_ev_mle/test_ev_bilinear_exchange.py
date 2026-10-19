"""Analytic and small independent-grid checks of the witness exchange loop."""
import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from research_experiments.mobile_ev_mle.ev_bilinear_support_milp import BilinearPair,evaluate_qp
from research_experiments.mobile_ev_mle.ev_bilinear_exchange import witness_quadratic,exchange_bilinear


def anchored(h):
    return BilinearPair('anchored',[0.,0.],[[1.],[0.]],[[0.],[h]],[[[-1.],[0.]]],[0.],[2.])


def verify_records(out):
    lowers=[e['q_lower'] for e in out['evaluations']]
    if lowers:assert out['objective_lower']==pytest.approx(max(lowers))
    uppers=[out['initial_fixed_witness_upper']]+[r['global_upper'] for r in out['iterations']]
    assert np.all(np.diff(uppers)<=1e-10)
    assert not out['full_curve_enumeration'] and not out['self_profile_subtracted']
    assert not out['original_global_optimum_certified']
    for e in out['evaluations']:
        assert e['physical_validation']['feasible']
        assert e['q_lower'] <= e['q_upper']+1e-10
        assert all(s['q_tangent_lower']<=s['q_primal_upper']+1e-10 for s in e['qp_results'])


def test_cancelled_cross_term_true_zero_without_spurious_gain():
    out=exchange_bilinear([anchored(0.)],[-2.],[2.],initial_u=[1.],absolute_gap=1e-9)
    assert out['completed']
    assert out['objective_lower']==pytest.approx(0.,abs=1e-12)
    assert out['objective_upper']<1e-12
    verify_records(out)


@pytest.mark.parametrize('h',[.2,1.,3.])
def test_anchored_analytic_optimum_and_refit_witness(h):
    out=exchange_bilinear([anchored(h)],[0.],[2.],initial_u=[0.],
        max_iterations=8,segments=4,absolute_gap=1e-6,relative_gap=1e-5)
    truth=4*h*h/(4+h*h)
    assert out['objective_lower'] <= truth+1e-8
    assert out['objective_upper'] >= truth-1e-8
    assert out['completed'],out['status']
    assert out['gap'] <= out['target_gap']+1e-8
    assert out['witness_counts'][0]>=2
    verify_records(out)


def test_two_competitors_have_interior_optimum_and_segment_refinement():
    # No nuisance: min(u^2,(1-u)^2) has exact maximum 1/4 at u=1/2.
    a=BilinearPair('a',[0.],[[1.]],np.zeros((1,0)),np.zeros((1,1,0)),[],[])
    b=BilinearPair('b',[1.],[[-1.]],np.zeros((1,0)),np.zeros((1,1,0)),[],[])
    out=exchange_bilinear([a,b],[0.],[1.],segments=1,max_segments=8,
                          max_iterations=8,absolute_gap=1e-6,relative_gap=0.)
    assert out['completed']
    assert out['u'][0]==pytest.approx(.5,abs=1e-6)
    assert out['objective_lower']==pytest.approx(.25,abs=1e-7)
    assert out['objective_upper']==pytest.approx(.25,abs=1e-7)
    assert any(x['refinement_next'] for x in out['iterations'])
    verify_records(out)


def test_box_boundary_competition_and_fixed_witness_identity():
    p=BilinearPair('boundary',[0.],[[1.]],[[1.]],[[[.2]]],[.3],[.7])
    for z in [.3,.5,.7]:
        f=witness_quadratic(p,[z])
        for u in np.linspace(0,2,17):
            assert f.value([u])==pytest.approx(float(p.residual([u],[z])@p.residual([u],[z])))
            assert evaluate_qp(p,[u])['q_primal_upper'] <= f.value([u])+1e-10
    out=exchange_bilinear([p],[0.],[2.],initial_u=[0.],absolute_gap=1e-6)
    truth=(2.+1.4*.3)**2
    assert out['completed']
    assert out['objective_lower']==pytest.approx(truth,abs=1e-6)
    assert out['objective_upper']>=truth-1e-7
    verify_records(out)


def test_small_two_time_equal_energy_grid_only_audits_bounds():
    a=BilinearPair('a',[.1,0.],[[1.,-.4],[0.,0.]],[[.2],[.8]],[[[-.5],[0.]],[[.3],[0.]]],[-.2],[1.2])
    b=BilinearPair('b',[-.2,.1],[[-.3,1.],[0.,0.]],[[.5],[.7]],[[[.4],[0.]],[[.1],[0.]]],[-.5],[.8])
    out=exchange_bilinear([a,b],[0.,0.],[1.,1.],LinearConstraint([[1.,1.]],1.,1.),
        initial_u=[.5,.5],max_iterations=3,segments=2,max_segments=8,absolute_gap=1e-7,relative_gap=1e-5)
    gridmax=max(min(evaluate_qp(p,[x,1-x])['q_primal_upper'] for p in (a,b)) for x in np.linspace(0,1,401))
    assert out['objective_upper']>=gridmax-1e-7
    assert out['objective_lower'] <= min(evaluate_qp(p,out['u'])['q_primal_upper'] for p in (a,b))+1e-8
    assert sum(out['u'])==pytest.approx(1.,abs=1e-7)
    for r in out['iterations']:
        assert r['global_upper']>=gridmax-1e-7
    verify_records(out)


def test_one_iteration_budget_is_honestly_unfinished_with_valid_bounds():
    a=BilinearPair('a',[0.],[[1.]],np.zeros((1,0)),np.zeros((1,1,0)),[],[])
    b=BilinearPair('b',[1.],[[-1.]],np.zeros((1,0)),np.zeros((1,1,0)),[],[])
    out=exchange_bilinear([a,b],[0.],[1.],max_iterations=1,segments=1,max_segments=1,
                          absolute_gap=1e-9,relative_gap=0.)
    assert out['unfinished'] and not out['completed']
    assert out['objective_lower']<=.25+1e-8<=out['objective_upper']+1e-8
    assert out['gap']>.2
    verify_records(out)


def test_infeasible_operations_no_false_lower_bound():
    out=exchange_bilinear([anchored(1.)],[0.],[1.],LinearConstraint([[1.]],2.,3.))
    assert out['status']=='operations_infeasible'
    assert out['u'] is None and out['objective_lower'] is None
    assert out['unfinished']


def test_invalid_initial_u_witness_and_domain_rejected():
    p=anchored(1.)
    with pytest.raises(ValueError,match='operations'):
        exchange_bilinear([p],[0.],[1.],initial_u=[2.])
    with pytest.raises(ValueError,match='witness'):
        witness_quadratic(p,[3.])
    with pytest.raises(ValueError,match='max_segments'):
        exchange_bilinear([p],[0.],[1.],segments=4,max_segments=2)
    with pytest.raises(ValueError,match='unique'):
        exchange_bilinear([p,p],[0.],[1.])
