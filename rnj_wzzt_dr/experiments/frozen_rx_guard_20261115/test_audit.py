"""Independent algebra and evidence-scope tests; no optimizer or AC calls."""
import numpy as np
import pytest

from . import audit


def test_design_reproduces_full_frozen_response_including_common_block():
    rng = np.random.default_rng(14)
    p, q = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    supports = [[0], [1], [2], [0, 1], [0, 1, 2]]
    r, x = rng.uniform(size=(2, 5))
    R, X = np.zeros((3, 3)), np.zeros((3, 3))
    for group, rr, xx in zip(supports, r, x):
        R[np.ix_(group, group)] += rr
        X[np.ix_(group, group)] += xx
    got = np.einsum('tnk,k->tn', audit.support_design(p, q, supports), np.r_[r, x])
    np.testing.assert_allclose(got, p@R.T+q@X.T, atol=1e-14)


def test_sandwich_matches_explicit_time_block_covariance():
    rng = np.random.default_rng(23)
    p, q = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
    covariance = np.array([[.07, .03], [.03, .11]])
    result = audit.parameter_sandwich(p, q, [[0], [1], [0, 1]], covariance)
    design = result['design'].reshape(20, 6)
    inverse = np.linalg.inv(design.T@design)
    expected = inverse@design.T@np.kron(np.eye(10), covariance)@design@inverse
    np.testing.assert_allclose(result['covariance'], expected, atol=2e-14)
    assert result['rank'] == 6


def test_rank_deficiency_is_rejected_instead_of_silent_pseudoinverse():
    p = np.arange(1, 13, dtype=float).reshape(6, 2)
    with pytest.raises(ValueError, match='not full column rank'):
        audit.parameter_sandwich(p, .2*p, [[0], [1]], np.eye(2))


def test_increment_W_uses_increment_only_and_hour_repeat_cancels():
    rng = np.random.default_rng(8)
    p, q = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
    affine = audit.affine_history(p, q)
    cov = audit.parameter_sandwich(p, q, [[0], [1]], np.eye(2)*.01)
    rows = {4: dict(supports=[[0], [1]], covariance=cov['covariance'])}
    pmap, qmap = np.array([[.01, 0], [0, 0]]), np.array([[0, 0], [0, .01]])
    action = rng.normal(size=(24, 2))
    result = audit.increment_values(action, pmap, qmap, affine, rows, .1)
    repeated = audit.increment_values(np.repeat(action, 4, axis=0), pmap, qmap, affine, rows, .1)
    assert result == pytest.approx(repeated)
    zero = audit.increment_values(np.zeros_like(action), pmap, qmap, affine, rows, .1)
    assert zero == {'pq': 0., 'response_native4': 0.}


def test_original_budget_feasible_can_lie_outside_numerical_buffer():
    result = audit.quadratic_value(np.array([[1.]]), np.array([[1.]]), 1.)
    assert result['passed'] and not result['encoded_budget_passed']


def test_zero_fallback_is_not_an_upper_bound_for_failed_new_quote():
    result = audit.cost_certificate(dict(feasible=False, zero_fallback=True,
        new_guard_and_probability_feasible=False, cost_yuan=None, lower_bound_yuan=0.),
        0., False, False)
    assert result['feasible_upper_bound_yuan'] is None
    with pytest.raises(AssertionError, match='Fallback'):
        audit.cost_certificate(dict(feasible=True, zero_fallback=True), 0., True, True)


def test_conflicting_raw_bound_is_retained_but_effective_bound_must_be_withdrawn():
    quote = dict(feasible=True, cost_yuan=1., lower_bound_yuan=0., relative_gap=1.,
        lower_bound_withdrawn=True, native_attributes={'ObjBound': 4.2})
    result = audit.cost_certificate(quote, 1., True, True)
    assert result['raw_native_attributes']['ObjBound'] == 4.2
    with pytest.raises(AssertionError, match='exceeds'):
        audit.cost_certificate(dict(quote, lower_bound_yuan=4.2), 1., True, True)


def test_unbuffered_feasible_point_cannot_use_buffered_domain_lower_bound():
    with pytest.raises(AssertionError, match='Buffered-domain'):
        audit.cost_certificate(dict(feasible=True, cost_yuan=1., lower_bound_yuan=.2), 1., True, False)


def test_full_native_margin_includes_a_nonrepresentative_rival():
    ctx=dict(n=3,truth={(0,1)},active=[{(0,1)},{(1,2)},{(1,2)}],
             H=np.zeros((3,3,6)),history=np.array([1.,2.,3.]),sigma2=.5)
    result=audit.full_native_diagnostic(ctx,dict(clean_sse=np.array([2.,5.,0.]),
                                                observed_sse=np.array([4.,5.,0.])))
    assert result['true_history_fixed_target']==0
    assert result['observed_strongest_foreign_native']==2
    assert result['observed_worst_true_logodds']==-2
    assert not result['observed_true_fixed_point_beats_all_foreign']


def test_nodewise_PQ_budget_does_not_substitute_pooled_scale():
    affine=dict(slope=np.zeros(2),residual_rms=np.array([1.,10.]))
    actual=audit.increment_values(np.ones((24,1)),np.zeros((2,1)),np.array([[1.],[0.]]),affine,{},1.)
    assert actual['pq']==pytest.approx(.5)
    assert actual['pq']!=pytest.approx(1/101)
