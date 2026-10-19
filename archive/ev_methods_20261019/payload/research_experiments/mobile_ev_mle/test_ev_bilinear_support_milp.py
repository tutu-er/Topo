"""Independent analytic, support-LP and domain audits; no AC experiments."""
import numpy as np
import pytest
from scipy.optimize import LinearConstraint, linprog
from research_experiments.mobile_ev_mle.ev_bilinear_support_milp import (
    BilinearPair, evaluate_qp, fixed_witness_upper, solve_bilinear_support_milp,
    support_value, unit_direction,
)


def cancelling_pair(history=0.):
    return BilinearPair('cancel', [0.,0.], [[1.],[0.]], [[0.],[history]],
                         [[[-1.],[0.]]], [0.], [2.])


def test_cross_term_cancels_frozen_spurious_separation():
    pair=cancelling_pair()
    for u in np.linspace(-2,2,31):
        out=evaluate_qp(pair,[u])
        assert out['q_primal_upper'] < 1e-20
        # Freezing A at u0=0 falsely leaves the positive square u*u.
        frozen=(pair.c+pair.B@np.array([u]))
        assert frozen@frozen == pytest.approx(u*u)
    out=solve_bilinear_support_milp([pair],[[[1.,0.],[-1.,0.]]],[-2.],[2.])
    assert out['u'] is not None
    assert out['distance_lower'] == pytest.approx(0.,abs=1e-10)
    assert out['evaluated_distance_primal_upper'] < 1e-8


@pytest.mark.parametrize('history',[.1,1.,3.])
def test_history_anchor_analytic_profile(history):
    pair=cancelling_pair(history)
    for u in np.linspace(-2,2,41):
        expected=history**2*u**2/(history**2+u**2)
        solved=evaluate_qp(pair,[u])
        assert solved['q_primal_upper'] == pytest.approx(expected,abs=1e-10)
        assert solved['q_tangent_lower'] <= expected+1e-10
        assert solved['numerical_gap'] < 1e-9


def test_anchor_direction_finds_nonzero_action_and_retains_qp():
    pair=cancelling_pair(1.)
    direction=unit_direction(evaluate_qp(pair,[2.])['residual'])
    out=solve_bilinear_support_milp([pair],[[direction]], [0.],[2.])
    assert out['physical_validation']['feasible']
    assert out['distance_lower'] > 0
    assert out['distance_lower'] <= out['evaluated_distance_primal_upper']+1e-9
    assert out['evaluated_distance_primal_upper'] <= np.sqrt(.8)+1e-9
    assert not out['lower_master_best_bound_is_true_distance_upper']


def test_random_box_support_matches_independent_linprog():
    rng=np.random.default_rng(781)
    pair=BilinearPair('random',rng.normal(size=4),rng.normal(size=(4,2)),
        rng.normal(size=(4,3)),rng.normal(size=(2,4,3)),[-1.,.1,-.4],[.4,2.,.7])
    for _ in range(30):
        u=rng.uniform(-1,1,2);v=unit_direction(rng.normal(size=4))
        linear=v@pair.matrix(u)
        solved=linprog(linear,bounds=list(zip(pair.lower,pair.upper)),method='highs')
        expected=float(v@(pair.c+pair.B@u)+solved.fun)
        assert support_value(pair,v,u) == pytest.approx(expected,abs=1e-10)
        qp=evaluate_qp(pair,u)
        assert support_value(pair,v,u) <= qp['distance_primal_upper']+1e-9


def test_big_m_and_abs_domain_valid_on_random_control_box():
    rng=np.random.default_rng(105)
    pair=BilinearPair('random',rng.normal(size=3),rng.normal(size=(3,2)),
        rng.normal(size=(3,2)),rng.normal(size=(2,3,2)),[-.7,.1],[.8,1.2])
    directions=[unit_direction(rng.normal(size=3)) for _ in range(5)]
    out=solve_bilinear_support_milp([pair],[directions],[-1.,0.],[1.,2.],
                                   LinearConstraint([[1.,1.]],.4,1.4))
    assert out['physical_validation']['feasible']
    for row in out['direction_audits']:
        for u in rng.uniform([-1.,0.],[1.,2.],size=(100,2)):
            ell=support_value(pair,row['direction'],u)
            assert ell >= row['ell_box_lower']-1e-10
            assert ell+row['big_m']-row['numeric_guard'] >= out['fixed_witness_global_upper']-1e-10
    assert out['distance_lower'] <= out['evaluated_distance_primal_upper']+1e-8


def test_multiple_competitors_and_zero_only_master():
    p=cancelling_pair(1.)
    other=BilinearPair('other',[.5],[[.5]],[[1.]],[[[.2]]],[-.1],[.1])
    out=solve_bilinear_support_milp([p,other],[[],[]],[0.],[2.])
    assert out['u'] is not None
    assert out['distance_lower'] == 0
    assert out['lower_master_best_bound'] < 1e-8
    # The optimum of a zero-only lower master says nothing about true optimum.
    assert min(evaluate_qp(x,[2.])['distance_primal_upper'] for x in (p,other)) > .5


def test_direction_validation_and_explicit_normalization():
    p=cancelling_pair()
    assert np.linalg.norm(unit_direction([3.,4.])) == pytest.approx(1.)
    assert np.array_equal(unit_direction([0.,0.]),[0.,0.])
    with pytest.raises(ValueError,match='norm'):
        support_value(p,[2.,0.],[1.])
    with pytest.raises(ValueError,match='dimension'):
        support_value(p,[1.],[1.])
    with pytest.raises(ValueError,match='finite ordered'):
        solve_bilinear_support_milp([p],[[]],[0.],[np.inf])


def test_fixed_nuisance_coordinate_and_witness_validation():
    p=BilinearPair('fixed',[.2],[[1.]],[[.5]],[[[.3]]],[.7],[.7])
    for u in [-1.,0.,2.]:
        out=evaluate_qp(p,[u])
        assert out['q_primal_upper'] == pytest.approx((.2+u+(.5+.3*u)*.7)**2)
    upper=fixed_witness_upper(p,[-1.],[2.])
    assert upper >= max(abs(.2+u+(.5+.3*u)*.7) for u in [-1.,2.])
    with pytest.raises(ValueError,match='witness'):
        fixed_witness_upper(p,[-1.],[2.],z=[0.])


def test_infeasible_operations_preserved():
    out=solve_bilinear_support_milp([cancelling_pair(1.)],[[]],[0.],[1.],
                                   LinearConstraint([[1.]],2.,3.))
    assert out['status']==2
    assert out['u'] is None and out['distance_lower'] is None
