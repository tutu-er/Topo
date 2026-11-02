"""Independent synthetic checks of the conditional-DR wrapper.

No case packet, true topology, observed trajectory, fit or main-batch runner is
opened.  The full-Gaussian replay below is independent of the interval kernel.
Zero-conditioning-factor guards were repaired after the bounded main batch;
that batch's immutable executed source is retained separately.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import norm

from problem import ConditionalProblem, integration_direction


REPO = Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')


def _packet():
    return dict(
        shape=(2, 2),
        mu_C=np.array([.4, -.2, .5, .3]),
        mu_F=np.array([.25, .6, -.1, .4]),
        L_C=np.array([[.5, .15], [-.25, .4], [.2, -.3], [.35, .1]]),
        L_F=np.array([[.3, -.1], [.2, .2], [-.1, .3], [.1, .25]]),
        direction=np.array([1., 0.]),
        complement=np.array([[0.], [1.]]),
        effect_C=np.array([[.8, -.25], [.3, .6]]),
        effect_F=np.array([[.2, .15], [-.4, .1]]),
        cv=np.full((2, 2), .2),
        response_factor=None,
        kappa=1.03,
        threshold=.05,
        dt=.25,
        elasticities=np.array([[.5, 1.], [2., .25]]),
    )


def _full_gaussian_probability(p, u, n=150000, seed=903422):
    """Direct all-factor MC, without conditioning or the interval kernel."""
    rng = np.random.default_rng(seed)
    zbase = rng.standard_normal((n, p['L_C'].shape[1]))
    zexec = rng.standard_normal((n, int(np.prod(p['shape']))))
    if p.get('response_factor') is not None:
        zexec = zexec @ p['response_factor'].T
    delivered = np.asarray(u)[None] * (1 + p['cv'] * zexec.reshape((n,) + p['shape']))
    rc = p['mu_C'][None] + zbase @ p['L_C'].T
    rf = p['mu_F'][None] + zbase @ p['L_F'].T
    rc += np.einsum('nr,srt->stn', p['effect_C'], delivered).reshape(n, -1)
    rf += np.einsum('nr,srt->stn', p['effect_F'], delivered).reshape(n, -1)
    event = np.abs(rc).mean(1) - p['kappa'] * np.abs(rf).mean(1) > p['threshold']
    probability = float(event.mean())
    return probability, np.sqrt(probability * (1-probability) / n)


@pytest.mark.parametrize('a0_equals_f', [False, True])
def test_conditional_wrapper_matches_independent_full_gaussian(a0_equals_f):
    p = _packet()
    if a0_equals_f:
        p['effect_F'] = np.zeros_like(p['effect_F'])
    u = np.array([[.25, -.35], [.3, .1]])
    result = ConditionalProblem(p, 1536, 903420).evaluate(u.ravel())
    direct, direct_se = _full_gaussian_probability(p, u)
    # Independent conditional-outer and full-Gaussian draws: account for both
    # finite sampling errors rather than demanding spurious decimal equality.
    combined_se = np.hypot(result['conditional_mc_se'], direct_se)
    assert abs(result['probability'] - direct) < 5*combined_se + .002


def test_reference_and_competitor_share_the_same_base_gaussian_realization():
    p = _packet()
    p['mu_F'] = p['mu_C'].copy()
    p['L_F'] = p['L_C'].copy()
    p['effect_F'] = p['effect_C'].copy()
    p['threshold'] = .02
    q = ConditionalProblem(p, 64, 903430)
    assert np.array_equal(q.a0[:, :q.m], q.a0[:, q.m:])
    assert np.array_equal(q.b[:q.m], q.b[q.m:])
    result = q.evaluate(np.array([.3, -.2, .1, .4]))
    # Identical residuals give (1-kappa)*mean(abs(r)) <= 0 on every path.
    assert result['probability'] == 0.
    assert np.all(result['conditional_values'] == 0.)


def test_fixed_conditional_approximation_chain_rule_with_nonzero_f_effect():
    p = _packet()
    q = ConditionalProblem(p, 96, 903440)
    u = np.array([.25, -.35, .3, .1])
    result = q.evaluate(u)
    assert not result['invalid_indices']
    assert result['gradient'] is not None
    assert 'finite fixed integration approximation' in result['gradient_scope']
    eps = 2.e-6
    for j in range(u.size):
        move = np.eye(u.size)[j] * eps
        finite = (q.evaluate(u + move)['probability'] - q.evaluate(u - move)['probability'])/(2*eps)
        assert result['gradient'][j] == pytest.approx(finite, rel=2.e-4, abs=2.e-6)


def test_signed_correlated_multipliers_cancel_opposite_commands_exactly():
    p = dict(shape=(2, 1), mu_C=np.array([.5]), mu_F=np.array([.3]),
             L_C=np.array([[.2]]), L_F=np.array([[.1]]),
             direction=np.array([1.]), complement=np.empty((1, 0)),
             effect_C=np.array([[1., 1.]]), effect_F=np.array([[0., 0.]]),
             cv=np.full((2, 1), .3),
             response_factor=np.array([[1., 0.], [1., 0.]]),
             kappa=1.03, threshold=.03, dt=.25, elasticities=np.full((2, 1), .5))
    q = ConditionalProblem(p, 96, 903450)
    u = np.array([[1.], [-1.]])
    delivered = q.multiplier * u[None]
    assert np.array_equal(delivered.sum(axis=1), np.zeros((q.samples, 1)))
    # abs(u) would not cancel the correlated shocks. This checks the joint law,
    # which cannot be diagnosed by per-node marginal means/variances alone.
    wrong_abs_delivery = u[None] + np.abs(u)[None] * (q.multiplier-1)
    assert np.any(np.abs(wrong_abs_delivery.sum(axis=1)) > 1.e-6)
    moved = q.evaluate(u.ravel())['probability']
    zero = q.evaluate(np.zeros(2))['probability']
    assert moved == pytest.approx(zero, abs=2.e-15)


def test_strict_event_retains_positive_probability_threshold_platform():
    p = dict(shape=(1, 1), mu_C=np.array([1.]), mu_F=np.array([1.]),
             L_C=np.array([[.2]]), L_F=np.array([[.2]]),
             direction=np.array([1.]), complement=np.empty((1, 0)),
             effect_C=np.array([[0.]]), effect_F=np.array([[0.]]),
             cv=np.zeros((1, 1)), response_factor=None,
             kappa=1., threshold=0., dt=.25, elasticities=np.full((1, 1), .5))
    result = ConditionalProblem(p, 4, 903460).evaluate([0.])
    assert result['probability'] == 0.  # M == threshold everywhere is NOT >.
    assert result['gradient'] is None
    assert len(result['invalid_indices']) == 4


def test_bidirectional_cost_gradient_is_positive_and_uses_each_elasticity():
    q = ConditionalProblem(_packet(), 1, 903470)
    u = np.array([.3, -.2, .1, -.4])
    assert q.cost(u) > 0
    assert q.cost(u) == pytest.approx(q.cost(-u))
    for j in range(u.size):
        d = np.eye(u.size)[j] * 1.e-6
        finite = (q.cost(u+d)-q.cost(u-d))/2.e-6
        assert q.cost_gradient(u)[j] == pytest.approx(finite, rel=1.e-8, abs=1.e-9)


def _small_frozen_pool():
    """Two algebra-only predictors; no historical or physical file is read."""
    labels = [101, 102]
    points = dict(coarse=dict(supports=[[0], [1]], r=[.4, .4], x=[0., 0.]),
                  refined=dict(supports=[[0], [1], [0, 1]], r=[.2, .2, .2], x=[0., 0., 0.]))
    matrices = dict(coarse=np.eye(2)*.4, refined=np.array([[.4, .2], [.2, .4]]))
    losses = dict(coarse=1.01e-4, refined=1.e-4)
    models = []
    for identity in ('coarse', 'refined'):
        point = points[identity]
        models.append(dict(candidate_id=identity, labels=labels, point=deepcopy(point),
            R=matrices[identity].tolist(), X=np.zeros((2, 2)).tolist(),
            historical_validation_abs_sum=288*2*losses[identity],
            historical_validation_mae=losses[identity],
            frozen_output_supports=deepcopy(point['supports']),
            frozen_output_support_labels=[[labels[i] for i in s] for s in point['supports']]))
    return dict(status='frozen', labels=labels, models=models, selection_order=['coarse', 'refined'],
                historical_validation_cells=288*2, validation_rows=288,
                calibration=dict(a=1., b=1., median_root_depth=.1))


@pytest.mark.parametrize('position', [0., .5, 1.])
def test_w2_tau3_margin_equals_existing_selector_on_both_sides(position):
    sys.path.insert(0, str(REPO))
    from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score
    pool = _small_frozen_pool()
    p = np.full((96, 2), .001)
    pred_c = p @ np.asarray(pool['models'][0]['R']).T
    pred_f = p @ np.asarray(pool['models'][1]['R']).T
    y = pred_c + position*(pred_f-pred_c)
    result = score(pool, dict(P_terminal=p, Q_terminal=np.zeros_like(p), drop_target=y,
                              labels=pool['labels']), history_gate=True)
    rule = next(r for r in result['rules'] if r['new_day_weight'] == 2 and r['new_day_fraction'] == .03)
    h_c, h_f = [m['historical_validation_mae']*1.e6 for m in pool['models']]
    d_c = np.mean(np.abs(y-pred_c))*1.e6
    d_f = np.mean(np.abs(y-pred_f))*1.e6
    threshold = 1.5*(1.03*h_f-h_c) + 2.5e-6
    event = d_c - 1.03*d_f > threshold
    assert event == ('coarse' not in rule['eligible_candidate_ids'])
    assert event == (rule['selected_candidate_id'] == 'refined')
    assert rule['historical_eligible_candidate_ids'] == ['coarse', 'refined']


def test_nominal_upper_mean_does_not_clip_unbounded_gaussian_delivery():
    p = _packet()
    p['shape'] = (1, 1)
    p['mu_C'], p['mu_F'] = np.array([.5]), np.array([.2])
    p['L_C'], p['L_F'] = np.array([[.2]]), np.array([[.1]])
    p['direction'], p['complement'] = np.array([1.]), np.empty((1, 0))
    p['effect_C'], p['effect_F'] = np.array([[1.]]), np.array([[.2]])
    p['cv'], p['elasticities'] = np.full((1, 1), .2), np.full((1, 1), .5)
    q = ConditionalProblem(p, 4096, 903480)
    delivered = q.multiplier[:, 0, 0]  # command u=+1, nominal upper command.
    assert np.any(delivered > 1.)
    assert abs(float(np.mean(delivered > 1.))-.5) < .025
    assert np.all(np.isfinite(delivered))


def test_one_sided_cp_endpoints_and_nominal_success_are_not_certification():
    # Import defines utilities only; no prepare/solve/evaluate command runs.
    from runner import lower_cp
    assert lower_cp(0, 100) == 0.
    assert lower_cp(100, 100) == pytest.approx(.05**(1/100), rel=1.e-13)
    assert lower_cp(95, 100) < .95
    assert lower_cp(1900, 2000) < .95
    assert lower_cp(100, 100, .05/4) < lower_cp(100, 100, .05)


def test_no_historical_gaussian_factor_must_be_rejected():
    with pytest.raises(ValueError):
        integration_direction(np.array([1.]), np.array([.5]),
                              np.zeros((1, 3)), np.zeros((1, 3)), 1.03)


def test_ineffective_conditioner_returns_no_optimization_gradient():
    p = dict(shape=(1, 1), mu_C=np.array([0.]), mu_F=np.array([0.]),
             L_C=np.array([[1., 0.]]), L_F=np.array([[0., 0.]]),
             direction=np.array([0., 1.]), complement=np.array([[1.], [0.]]),
             effect_C=np.array([[0.]]), effect_F=np.array([[1.]]),
             cv=np.zeros((1, 1)), response_factor=None,
             kappa=1.03, threshold=0., dt=.25, elasticities=np.full((1, 1), .5))
    try:
        q = ConditionalProblem(p, 256, 903490)
    except ValueError:
        return
    result = q.evaluate([.7])
    # Exact population derivative = -2*kappa*phi(kappa*u), not zero.
    assert abs(-2*p['kappa']*norm.pdf(p['kappa']*.7)) > .5
    assert result['gradient'] is None


def test_fixed_zero_reference_residual_preserves_restricted_gradient():
    p=_packet()
    p['mu_F']=np.zeros_like(p['mu_F']);p['L_F']=np.zeros_like(p['L_F'])
    p['effect_F']=np.zeros_like(p['effect_F'])
    model=ConditionalProblem(p,96,991)
    u=np.array([.2,-.1,.15,.1]);result=model.evaluate(u)
    assert result['gradient'] is not None
    direction=np.array([1.,-2.,3.,1.]);direction/=np.linalg.norm(direction)
    eps=1e-5
    fd=(model.evaluate(u+eps*direction)['probability']-model.evaluate(u-eps*direction)['probability'])/(2*eps)
    assert result['gradient']@direction==pytest.approx(fd,abs=2e-6)


@pytest.mark.parametrize('key,value',[('cv',-1.),('elasticities',0.),('dt',0.)])
def test_invalid_contract_inputs_rejected(key,value):
    p=_packet();p[key]=value
    with pytest.raises(ValueError):ConditionalProblem(p,32,1)


def test_response_factor_unit_variance_checked_without_jitter():
    p=_packet();p['response_factor']=np.ones((4,4))
    with pytest.raises(ValueError):ConditionalProblem(p,32,1)
    p['response_factor']=np.tile(np.array([1.,0.,0.,0.]),(4,1))
    model=ConditionalProblem(p,32,1)
    assert np.array_equal(model.multiplier[:,0,0],model.multiplier[:,1,1])
