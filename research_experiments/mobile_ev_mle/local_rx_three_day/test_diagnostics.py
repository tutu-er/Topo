import copy

import numpy as np
import pytest

from .. import likelihood as lk
from .diagnostics import diagnose, derivative_blocks, canonical_trunk_check


def fixture(count=24):
    candidate = lk.RadialCandidate.from_supports('star', 3)
    r = np.array([.035, .021, .018, .026])
    x = np.array([.025, .014, .012, .017])
    rng = np.random.default_rng(161)
    p = rng.uniform(.015, .06, (count, 3))
    q = rng.uniform(.004, .025, (count, 3))
    root = 1.+.003*np.sin(np.arange(count)/5)
    physical = lk.ac_predict(candidate, r, x, p, q, root_v=root)
    # The diagnostic receives observed data and an attained point, not truth.
    obs = dict(p=p+.0001, q=q-.00003, v=physical['v']+.00004)
    sigma = dict(p=np.full(3, .0003), q=np.full(3, .0001), v=np.full(3, .0005))
    residual = np.c_[(p-obs['p'])/sigma['p'], (q-obs['q'])/sigma['q'],
                     (physical['v']-obs['v'])/sigma['v']]
    fit = dict(r=r.tolist(), x=x.tolist(), p=p.tolist(), q=q.tolist(),
               root_v=root.tolist(), loss=float(np.sum(residual**2)))
    domain = dict(z_min=.005, z_max=.12, ratio_min=.25, ratio_max=1.5,
                  power_caps=[.2]*3, root_min=.97, root_max=1.03,
                  v_min=.9, v_max=1.1)
    return candidate, obs, sigma, domain, fit, root, physical


@pytest.mark.parametrize('mode,rank', [('U', 6), ('R', 8), ('K', 8)])
def test_raw_residual_gradient_and_boundary_information(mode, rank):
    candidate, obs, sigma, domain, fit, root, _ = fixture()
    kwargs = {}
    if mode == 'K':
        kwargs['root_known'] = root
    elif mode == 'R':
        kwargs['root_meter'] = (root+.0002, .0005)
        fit['loss'] += len(root)*(.0002/.0005)**2
    result = diagnose(candidate, obs, sigma, domain, fit, **kwargs)
    assert not result['truth_used']
    assert result['extra_likelihood_terms'] == 0
    assert result['loss_absolute_difference'] < 1e-9
    assert result['profile_information']['rank'] == rank
    assert result['profile_information']['downstream_after_trunk_profile']['rank'] == 6
    assert max(d['residual_relative_error'] for d in result['finite_direction_checks']) < 1e-6
    assert result['local_star_identity']['branch_max_abs'] < 1e-12
    assert result['local_star_identity']['contrast_max_abs'] < 1e-12
    if mode == 'R':
        assert result['profile_information']['without_added_root_meter_at_same_point']['rank'] == 6


def test_exact_local_drop_derivatives_and_contrast_covariance():
    candidate, obs, sigma, domain, fit, root, physical = fixture(4)
    value = diagnose(candidate, obs, sigma, domain, fit)
    local = value['local_star_identity']
    contrast = np.asarray(local['contrast'])
    np.testing.assert_allclose(contrast @ np.ones(3), 0., atol=1e-15)
    np.testing.assert_allclose(contrast @ contrast.T, np.eye(2), atol=1e-15)
    p, q = np.asarray(fit['p']), np.asarray(fit['q'])
    w = physical['v']**2
    r, x = np.asarray(fit['r'])[1:], np.asarray(fit['x'])[1:]
    def drop(rr, xx):
        return 2*(rr*p+xx*q)+(rr*rr+xx*xx)*(p*p+q*q)/w
    step = 1e-6
    np.testing.assert_allclose((drop(r+step, x)-drop(r-step, x))/(2*step), local['dL_dr'], atol=1e-11)
    np.testing.assert_allclose((drop(r, x+step)-drop(r, x-step))/(2*step), local['dL_dx'], atol=1e-11)
    # Even orthogonal contrasts have correlated noise when channel variances differ.
    covariance = contrast @ np.diag([1., 2., 4.]) @ contrast.T
    assert abs(covariance[0, 1]) > .1


def test_canonical_anchor_checks_domain_and_does_not_mutate_fit():
    candidate, obs, sigma, domain, fit, root, physical = fixture()
    before = copy.deepcopy(fit)
    anchor = complex(.038, .022)
    result = diagnose(candidate, obs, sigma, domain, fit, canonical_anchor=anchor)
    assert fit == before
    assert result['canonical_trunk']['legal']
    assert result['canonical_trunk']['terminal_voltage_max_abs_change'] < 1e-12
    tight = dict(domain, root_min=float(root.min()), root_max=float(root.max()))
    result = canonical_trunk_check(candidate, physical, tight,
        np.asarray(fit['r']), np.asarray(fit['x']), np.asarray(fit['p']), np.asarray(fit['q']), .09+.04j)
    assert not result['legal']
    assert result['reason'] == 'mapped_root_outside_original_domain'
    assert not diagnose(candidate, obs, sigma, domain, fit, root_known=root,
                        canonical_anchor=anchor)['canonical_trunk']['legal']


def test_unsupported_refinement_retained_and_inputs_validated():
    candidate, obs, sigma, domain, fit, root, physical = fixture()
    pair = lk.RadialCandidate.from_supports('pair01', 3, ((0, 1),))
    rr = np.r_[.035, .01, .021, .018, .026]
    xx = np.r_[.025, .008, .014, .012, .017]
    pp, qq = np.asarray(fit['p']), np.asarray(fit['q'])
    physical = lk.ac_predict(pair, rr, xx, pp, qq, root_v=root)
    fit.update(r=rr.tolist(), x=xx.tolist())
    result = diagnose(pair, obs, sigma, domain, fit)
    assert not result['local_star_identity']['supported']
    with pytest.raises(ValueError, match='distinct'):
        derivative_blocks(candidate, obs, sigma, fit, root_known=root, root_meter=(root, .0005))
    bad = dict(sigma, v=0.)
    with pytest.raises(ValueError, match='positive'):
        diagnose(pair, obs, bad, domain, fit)


def test_excitation_proportional_q_and_constant_channel_are_explicit():
    candidate, obs, sigma, domain, fit, root, physical = fixture()
    obs['q'] = .3*obs['p']
    result = diagnose(candidate, obs, sigma, domain, fit)
    for row in result['observed_pq_excitation']:
        assert row['q_orthogonal_fraction'] < 1e-14
        assert row['centered_correlation'] == pytest.approx(1.)
    obs['p'][:, 0] = .03
    obs['q'][:, 0] = .01
    result = diagnose(candidate, obs, sigma, domain, fit)
    assert result['observed_pq_excitation'][0]['centered_correlation'] is None
