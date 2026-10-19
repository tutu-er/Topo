import numpy as np
import pytest
from ... import likelihood as lk
from ..scoring import (
    contrast_matrix, prepare, response_matrix, profile_at, fixed_covariance,
    score_fixed, convex_first_order_bound, fit_qp, fit_profile,
    _qp_geometry,
)


def fixture(count=36, *, noisy=True):
    candidate = lk.RadialCandidate.from_supports('pair01', 3, ((0, 1),))
    r = np.array([.035, .017, .021, .018, .026])
    x = np.array([.025, .011, .014, .012, .017])
    rng = np.random.default_rng(1921)
    p = rng.uniform(.015, .065, (count, 3))
    q = rng.uniform(.004, .028, (count, 3))
    R = (candidate.M*r) @ candidate.M.T
    X = (candidate.M*x) @ candidate.M.T
    w = 1.-2*p @ R.T-2*q @ X.T
    v = np.sqrt(w)
    sigma = dict(p=np.full(3, .0003), q=np.full(3, .00013), v=np.full(3, .0005))
    obs = dict(p=p.copy(), q=q.copy(), v=v.copy())
    if noisy:
        for key in ('p', 'q', 'v'):
            obs[key] += rng.normal(size=obs[key].shape)*sigma[key]
    domain = dict(z_min=.005, z_max=.12, ratio_min=.25, ratio_max=1.5,
                  v_min=.9, v_max=1.1)
    return candidate, r, x, obs, sigma, domain


def test_orthonormal_contrast_and_tree_response():
    candidate, r, x, _, _, _ = fixture()
    d = contrast_matrix(3)
    np.testing.assert_allclose(d @ np.ones(3), 0, atol=1e-15)
    np.testing.assert_allclose(d @ d.T, np.eye(2), atol=1e-15)
    R, X = (candidate.M*r) @ candidate.M.T, (candidate.M*x) @ candidate.M.T
    np.testing.assert_allclose(response_matrix(candidate, r, x), np.c_[-2*d @ R, -2*d @ X], atol=1e-16)
    r2, x2 = r.copy(), x.copy()
    r2[0] *= 2; x2[0] *= .5
    np.testing.assert_allclose(response_matrix(candidate, r, x), response_matrix(candidate, r2, x2))


def test_profile_is_direct_latent_weighted_least_squares():
    candidate, r, x, obs, sigma, _ = fixture(5)
    data = prepare(obs, sigma)
    result = profile_at(candidate, data, r, x)
    h = result['H']
    for t in range(len(obs['p'])):
        sx = data['sigma_x'][t]
        chol = np.linalg.cholesky(data['cov_y'][t])
        design = np.r_[np.diag(1/sx), np.linalg.solve(chol, h)]
        target = np.r_[data['x'][t]/sx, np.linalg.solve(chol, data['y'][t])]
        latent = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target-design @ latent
        np.testing.assert_allclose(latent, result['latent_x'][t], atol=2e-15, rtol=1e-12)
        expected = result['residual'][t] @ np.linalg.solve(result['covariance'][t], result['residual'][t])
        np.testing.assert_allclose(residual @ residual, expected, rtol=1e-10, atol=1e-12)
    sx2 = data['sigma_x']**2
    correction = (result['latent_x']-data['x'])/np.sqrt(sx2)
    y_residual = data['y']-result['latent_x'] @ h.T
    direct = np.sum(correction**2)+np.sum(y_residual*np.linalg.solve(data['cov_y'], y_residual[..., None])[..., 0])
    np.testing.assert_allclose(direct, result['loss'], rtol=1e-11)


def test_profile_gradient_includes_covariance_derivative():
    candidate, r, x, obs, sigma, _ = fixture()
    # Enlarge input noise so omission of covariance derivative fails clearly.
    sigma['p'] *= 8; sigma['q'] *= 8
    data = prepare(obs, sigma)
    result = profile_at(candidate, data, r, x)
    theta = np.r_[r, x]
    numeric = np.empty_like(theta)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += 2e-7; minus[j] -= 2e-7
        numeric[j] = (profile_at(candidate, data, plus[:candidate.m], plus[candidate.m:])['loss']-
                      profile_at(candidate, data, minus[:candidate.m], minus[candidate.m:])['loss'])/4e-7
    np.testing.assert_allclose(numeric, result['gradient_rx'], rtol=2e-6, atol=2e-6)


def test_covariance_upper_dominates_all_sampled_physical_models():
    candidate, _, _, obs, sigma, domain = fixture()
    data = prepare(obs, sigma)
    upper = fixed_covariance(candidate, data, domain, weight='upper')
    rng = np.random.default_rng(42)
    for _ in range(50):
        radius = rng.uniform(domain['z_min'], domain['z_max'], candidate.m)
        phi = rng.uniform(np.arctan(domain['ratio_min']), np.arctan(domain['ratio_max']), candidate.m)
        actual = profile_at(candidate, data, radius*np.cos(phi), radius*np.sin(phi), derivatives=False)['covariance']
        assert np.linalg.eigvalsh(upper-actual).min() >= -1e-16


def test_fixed_holdout_does_not_refit_training_lines():
    candidate, r, x, obs, sigma, domain = fixture()
    before = (r.copy(), x.copy())
    one = score_fixed(candidate, obs, sigma, r, x)
    two = score_fixed(candidate, obs, sigma, r, x, weight='upper', domain=domain)
    assert one['fixed_training_lines'] and two['fixed_training_lines']
    np.testing.assert_array_equal(r, before[0]); np.testing.assert_array_equal(x, before[1])
    assert two['loss'] <= one['loss']+1e-12


def test_convex_dual_bound_at_nonoptimal_point():
    # f(x)=(x0-.8)^2+(x1-.9)^2 over [0,1]^2, x0+x1<=1.
    point = np.array([.1, .2])
    center = np.array([.8, .9])
    result = convex_first_order_bound(point, float(np.sum((point-center)**2)),
                                     2*(point-center), [0, 0], [1, 1], [[1, 1]], [1])
    exact_optimum = .7**2/2
    assert result['lower_bound'] <= exact_optimum+1e-12
    assert min(result['multiplier']) >= 0
    g = 2*(point-center)
    adjusted = g+np.array([[1, 1]]).T @ result['multiplier']
    expected_dual = np.minimum(adjusted, 0).sum()-sum(result['multiplier'])
    np.testing.assert_allclose(result['dual_linear'], expected_dual)


@pytest.mark.parametrize('weight', ['upper', 'nominal'])
def test_qp_returns_small_verified_gap_and_physical_outer_domain(weight):
    candidate, _, _, obs, sigma, domain = fixture(60)
    result = fit_qp(candidate, obs, sigma, domain, weight=weight)
    assert result['qualified'], result
    assert result['constraint_violation'] < 1e-9
    assert 0 <= result['lower_bound'] <= result['loss']+1e-9
    assert result['absolute_gap'] < 1e-4
    assert result['unidentified_edges'] == [0]
    assert result['lower_bound_for_profile_model'] == (weight == 'upper')
    assert not result['AC_lower_bound']
    r, x = np.array(result['r']), np.array(result['x'])
    assert np.min(x/r) >= domain['ratio_min']-1e-8
    assert np.max(x/r) <= domain['ratio_max']+1e-8
    fixed = score_fixed(candidate, obs, sigma, r, x, weight=weight, domain=domain)
    np.testing.assert_allclose(fixed['loss'], result['loss'], atol=1e-10)
    theta = np.r_[r[result['active_edges']], x[result['active_edges']]]
    hessian, linear = np.array(result['quadratic_hessian']), np.array(result['quadratic_linear'])
    np.testing.assert_allclose(.5*theta @ hessian @ theta+linear @ theta+result['quadratic_constant'],
                               result['loss'], atol=1e-9)


def test_profile_fit_in_original_polar_domain_and_speed_metadata():
    candidate, _, _, obs, sigma, domain = fixture(80)
    result = fit_profile(candidate, obs, sigma, domain, multistart=2, max_iter=300, seed=9)
    assert result['qualified'], result
    radius = np.hypot(result['r'], result['x'])
    ratio = np.array(result['x'])/result['r']
    assert radius.min() >= domain['z_min']-1e-10
    assert radius.max() <= domain['z_max']+1e-10
    assert ratio.min() >= domain['ratio_min']-1e-10
    assert ratio.max() <= domain['ratio_max']+1e-10
    assert result['seconds'] >= result['initializer_seconds'] > 0
    assert result['response_calls'] >= 1
    assert not result['truth_used'] and not result['global_optimality_certified']
    recomputed = score_fixed(candidate, obs, sigma, result['r'], result['x'])
    np.testing.assert_allclose(recomputed['loss'], result['loss'], rtol=1e-11)


def test_convex_outer_qp_bounds_original_domain_profile_fit():
    candidate, _, _, obs, sigma, domain = fixture(60)
    profile = fit_profile(candidate, obs, sigma, domain, multistart=2, seed=7)
    qp = fit_qp(candidate, obs, sigma, domain, weight='upper')
    assert qp['lower_bound'] <= profile['loss']+1e-9


def test_sector_chord_and_tangent_outer_hull_contains_entire_polar_domain():
    candidate, _, _, obs, sigma, domain = fixture()
    b, active, _, lower, upper, a = _qp_geometry(candidate, prepare(obs, sigma), domain)
    rng = np.random.default_rng(104)
    for _ in range(100):
        radius = rng.uniform(domain['z_min'], domain['z_max'], len(active))
        angle = rng.uniform(np.arctan(domain['ratio_min']), np.arctan(domain['ratio_max']), len(active))
        point = np.r_[radius*np.cos(angle), radius*np.sin(angle)]
        assert np.max(a @ point-b['linear_rhs']) < 1e-14
        assert np.min(point-lower) >= -1e-14
        assert np.min(upper-point) >= -1e-14
    # The independent component-box lower corner is cut off by the chord.
    assert np.max(a @ lower-b['linear_rhs']) > 0


def test_noiseless_linear_response_recovers_response_not_common_trunk():
    candidate, r, x, obs, sigma, domain = fixture(100, noisy=False)
    result = fit_profile(candidate, obs, sigma, domain, multistart=2, seed=5)
    assert result['qualified'], result
    assert result['loss'] < 1e-9
    np.testing.assert_allclose(result['H'], response_matrix(candidate, r, x), atol=3e-7)
    assert result['unidentified_edges'] == [0]


@pytest.mark.parametrize('bad', [0., -1., np.nan])
def test_invalid_meter_scales_fail_explicitly(bad):
    _, _, _, obs, sigma, _ = fixture()
    sigma['p'][0] = bad
    with pytest.raises(ValueError):
        prepare(obs, sigma)
