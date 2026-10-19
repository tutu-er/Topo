"""Analytic score, QR, original-domain and service tests independent of AC fits."""
import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from research_experiments.mobile_ev_mle.ev_bilinear_support_milp import BilinearPair, evaluate_qp
from research_experiments.mobile_ev_mle.ev_bilinear_score_exchange import (
    all_ordered_pairs, history_score_offsets, difference_interval,
    witness_score_quadratic, exchange_bilinear_scores, evaluate_action_box, charging_service_domain)
from research_experiments.mobile_ev_mle.ev_service_milp import (
    EVServiceConstraints, canonical_service_plan, check_service_plan)


def direct(name, c, B):
    c, B = np.asarray(c, float), np.asarray(B, float)
    return BilinearPair(name, c, B, np.zeros((len(c), 0)), np.zeros((B.shape[1], len(c), 0)), [], [])


def test_history_added_and_joint_have_different_declared_meanings():
    ordered = all_ordered_pairs(('a', 'b', 'c', 'd'))
    assert len(ordered) == 12 and len(set(ordered)) == 12
    costs = dict(a=2., b=5., c=7., d=11.)
    added = history_score_offsets(ordered, costs, mode='added')
    joint = history_score_offsets(ordered, costs, mode='joint')
    assert added['a__to__b'] == 5. and joint['a__to__b'] == 2.
    q = 13.
    assert (q-joint['a__to__b'])-(q-added['a__to__b']) == costs['b']-costs['a']


@pytest.mark.parametrize('a,b,scale', [(0.,0.,1.), (.4,.1,1.), (3.,3.,.5)])
def test_offsets_shift_or_move_true_continuous_optimum(a,b,scale):
    pairs = [direct('a', [0.], [[1.]]), direct('b', [1.], [[-1.]])]
    out = exchange_bilinear_scores(pairs, [0.], [1.], offsets={'a': a, 'b': b},
        score_scale=scale, initial_u=[.5], segments=4, max_iterations=8,
        absolute_gap=1e-7, relative_gap=0.)
    ustar = (1+a-b)/2
    optimum = scale*(ustar**2-a)
    if a == b:
        assert out['completed'], out['status']
        assert out['gap'] <= out['target_gap']+1e-8
    else:
        # The optimum is between chord knots. A remaining chord gap is
        # retained even though this symmetric example returns the exact u.
        assert out['unfinished'] and out['status']=='repeated_control_with_gap'
        assert 0 < out['gap'] <= 1/(4*32**2)+1e-8
    assert out['objective_lower'] <= optimum+1e-8 <= out['objective_upper']+1e-8
    assert out['u'][0] == pytest.approx(ustar, abs=1e-6)
    assert out['objective_lower'] == pytest.approx(optimum, abs=1e-7)
    if a == b == 3:
        assert out['objective_lower'] < 0 and out['objective_upper'] < 0
    assert out['full_curve_enumeration'] is False


def test_all_required_pairs_evaluated_and_difficult_pair_retained():
    pairs = [direct(a+'__to__'+b, [0. if (a,b)==('d','c') else 1.], [[0.]])
             for a,b in all_ordered_pairs(('a','b','c','d'))]
    offsets = {p.name: .1 for p in pairs}
    out = exchange_bilinear_scores(pairs, [0.], [1.], offsets=offsets, score_scale=.5, initial_u=[.2])
    assert out['completed']
    assert out['objective_lower'] == pytest.approx(-.05)
    assert len(out['required_pairs']) == 12
    assert all(len(e['qp_results']) == 12 for e in out['evaluations'])
    assert out['qp_calls'] == 12*len(out['evaluations'])


def test_qr_preserves_gram_value_and_constant_for_many_residual_rows():
    rng = np.random.default_rng(145901)
    m,n,t = 200,5,4
    p = BilinearPair('many', rng.normal(size=m), rng.normal(size=(m,t)),
        rng.normal(size=(m,n)), rng.normal(size=(t,m,n)), [-1]*n, [2]*n)
    z = rng.uniform(-1,2,n); low = np.full(t,-2.); high = np.full(t,3.)
    q, audit = witness_score_quadratic(p,z,7.,.5,low,high)
    assert audit['original_rows'] == m and audit['compressed_rows'] == t+1
    assert audit['constant_column_retained']
    assert audit['gram_max_abs_error'] < 1e-10
    for u in rng.uniform(low,high,(30,t)):
        exact = .5*(float(p.residual(u,z)@p.residual(u,z))-7.)
        assert q.value(u) >= exact-1e-9
        assert q.value(u)-exact <= 2*audit['box_numerical_upward_guard']+1e-9


def test_qr_rank_deficiency_keeps_nonzero_constant():
    p = direct('constant', [3.,4.,5.], [[0.,0.],[0.,0.],[0.,0.]])
    q,a = witness_score_quadratic(p, [], 7., .5, [0.,0.], [1.,1.])
    assert q.value([0.,0.]) == pytest.approx(.5*(50.-7.),abs=1e-10)
    assert q.value([1.,1.]) == pytest.approx(q.value([0.,0.]),abs=1e-10)
    assert a['compressed_rows'] == 3


def test_competing_bilinear_refit_with_signed_offset_bounds():
    h=.6
    p=BilinearPair('anchored',[0.,0.],[[1.],[0.]],[[0.],[h]],[[[-1.],[0.]]],[0.],[2.])
    out=exchange_bilinear_scores([p],[0.],[2.],offsets={'anchored': 2.},score_scale=.5,
        initial_u=[0.],absolute_gap=1e-7,relative_gap=1e-6,max_iterations=10)
    truth=.5*(4*h*h/(4+h*h)-2.)
    assert out['completed'],out['status']
    assert out['objective_lower']<=truth+1e-8<=out['objective_upper']+1e-8
    assert out['objective_lower']==pytest.approx(truth,abs=1e-7)
    assert len(out['witnesses'][0])>=2
    uppers=[out['initial_fixed_witness_upper']]+[r['global_upper'] for r in out['iterations']]
    assert np.all(np.diff(uppers)<=1e-10)
    assert all(a['compressed_rows']<=2 for r in out['iterations'] for a in r['compressions'])


def test_interval_for_variable_self_cost_uses_opposite_endpoints():
    interval=difference_interval(8.,10.,3.,6.,scale=.5)
    assert interval=={'lower':1.,'upper':3.5}
    for b in np.linspace(8,10,5):
        for a in np.linspace(3,6,7):
            assert interval['lower'] <= .5*(b-a) <= interval['upper']
    # Lower-lower subtraction would be 2.5, exceeding the attainable 1.0.
    assert .5*(8.-3.) > interval['lower']


def test_original_action_box_audit_preserves_its_difference_from_outer_relaxation():
    p=BilinearPair('shift',[0.],[[1.]],[[1.]],[[[0.]]],[-1.],[.8])
    relaxed=evaluate_qp(p,[1.])
    audit=evaluate_action_box(p,[1.],[.2],[.8],[[-1.]],[[-1.]])
    assert audit['original_box_inside_declared_outer']
    assert relaxed['q_primal_upper']==pytest.approx(0.,abs=1e-12)
    assert audit['q_primal_upper']==pytest.approx(.04,abs=1e-12)
    assert audit['q_tangent_lower']<=.04<=audit['q_primal_upper']+1e-12


@pytest.mark.parametrize('site', [0,1])
def test_pure_charge_domain_matches_battery_service_with_driving_and_access(site):
    c=EVServiceConstraints(terminal_energy=1.3425,p_charge_max=.3,battery_capacity=3.,
        initial_soc=.4,travel_slots=[[0,1],[1,0]],travel_energy=[[0,.01],[.01,0]])
    d=charging_service_domain(site,4,2,c,power_scale=20.)
    assert d['success']
    u=d['constant_u']; lc=d['constraints']
    assert np.all(u>=d['lower']) and np.all(u<=d['upper']+1e-10)
    assert np.all(lc.A@u>=lc.lb-1e-10) and np.all(lc.A@u<=lc.ub+1e-10)
    p=canonical_service_plan(site,u/20.,2,c)
    assert check_service_plan(p,c)['feasible']
    if site==1:
        assert d['upper'][0]==0 and u[0]==0
        assert np.sum(u)*.25 > 3.  # same terminal battery task compensates driving
    else:
        assert u==pytest.approx([3.,3.,3.,3.])


def test_failure_and_invalid_offsets_do_not_silently_drop_pairs():
    p=direct('a',[0.],[[1.]])
    with pytest.raises(ValueError,match='every required pair'):
        exchange_bilinear_scores([p],[0.],[1.],offsets={})
    with pytest.raises(ValueError,match='mode'):
        history_score_offsets([('a','b')],{'a':0,'b':0},mode='probability')
    with pytest.raises(ValueError,match='ordered'):
        difference_interval(3.,2.,0.,1.)
    out=exchange_bilinear_scores([p],[0.],[1.],LinearConstraint([[1.]],2.,3.))
    assert out['status']=='operations_infeasible' and out['objective_lower'] is None
    assert out['unfinished']


def test_uniform_curve_failure_does_not_reject_a_feasible_variable_curve():
    c=EVServiceConstraints(terminal_energy=1.3425,p_charge_max=[[.1],[.3],[.3],[.3]],
        battery_capacity=3.,initial_soc=.4)
    d=charging_service_domain(0,4,1,c,power_scale=20.)
    assert d['success'] and d['constant_u'] is None
    u=np.array([2.,4.,4.,2.])
    assert np.all(u<=d['upper'])
    assert np.all(d['constraints'].A@u>=d['constraints'].lb-1e-10)
    assert np.all(d['constraints'].A@u<=d['constraints'].ub+1e-10)
    assert check_service_plan(canonical_service_plan(0,u/20.,1,c),c)['feasible']
