import numpy as np

from .. import likelihood as lk
from .fitting import evaluate, eliminated_step, fit, profile_information, polar_lines


def fixture(t=24):
    candidate = lk.RadialCandidate.from_supports('star', 3)
    r = np.array([.035, .021, .018, .026])
    x = np.array([.025, .014, .012, .017])
    theta = np.r_[np.log(np.hypot(r, x)), np.arctan2(x, r)]
    rng = np.random.default_rng(73)
    p = rng.uniform(.015, .06, (t, 3))
    q = rng.uniform(.004, .025, (t, 3))
    root = 1.+.003*np.sin(np.arange(t)/5)
    physical = lk.ac_predict(candidate, r, x, p, q, 0., 0., root)
    obs = dict(p=p, q=q, v=physical['v'])
    sigma = dict(p=np.full(3, .0003), q=np.full(3, .0001), v=np.full(3, .0005))
    domain = dict(z_min=.005, z_max=.12, ratio_min=.25, ratio_max=1.5,
                  power_caps=[.2]*3, root_min=.97, root_max=1.03,
                  v_min=.9, v_max=1.1)
    return candidate, theta, obs, sigma, domain, root, physical


def test_physical_implicit_and_root_derivatives():
    candidate, theta, obs, sigma, domain, root, _ = fixture(5)
    local = np.c_[np.zeros((5, 6)), (root-1)/.01]
    result = evaluate(candidate, obs, sigma, theta, local)
    step = 1e-5
    for j in (0, 3, 4, 7):
        a, b = theta.copy(), theta.copy()
        a[j] += step; b[j] -= step
        diff = (evaluate(candidate, obs, sigma, a, local, derivatives=False)['residual']-
                evaluate(candidate, obs, sigma, b, local, derivatives=False)['residual'])/(2*step)
        np.testing.assert_allclose(diff, result['shared'][:, :, j], atol=2e-7, rtol=2e-6)
    for j in (0, 4, 6):
        a, b = local.copy(), local.copy()
        a[:, j] += step; b[:, j] -= step
        diff = (evaluate(candidate, obs, sigma, theta, a, derivatives=False)['residual']-
                evaluate(candidate, obs, sigma, theta, b, derivatives=False)['residual'])/(2*step)
        np.testing.assert_allclose(diff, result['state'][:, :, j], atol=2e-7, rtol=2e-6)


def test_elimination_matches_direct_damped_least_squares():
    rng = np.random.default_rng(17)
    a, c, r = rng.normal(size=(3, 9, 4)), rng.normal(size=(3, 9, 6)), rng.normal(size=(3, 9))
    da, dc = eliminated_step(a, c, r, .13)
    matrix = np.zeros((27, 22))
    for t in range(3):
        matrix[9*t:9*t+9, :4] = a[t]
        matrix[9*t:9*t+9, 4+6*t:10+6*t] = c[t]
    direct = np.linalg.solve(matrix.T@matrix+.13*np.eye(22), -matrix.T@r.ravel())
    np.testing.assert_allclose(np.r_[da, dc.ravel()], direct, atol=1e-12)


def test_unknown_boundary_exact_common_line_ambiguity():
    candidate, theta, obs, sigma, domain, root, physical = fixture()
    r, x = polar_lines(theta)
    # Preserve downstream phasors up to a common rotation; change only prefix Z.
    rr, xx = r.copy(), x.copy()
    rr[0] *= 1.2; xx[0] *= .8
    new_root = abs(physical['voltage'][:, 0]+(rr[0]+1j*xx[0])*physical['branch_current'][:, 0])
    other = lk.ac_predict(candidate, rr, xx, obs['p'], obs['q'], 0., 0., new_root)
    np.testing.assert_allclose(other['v'], obs['v'], atol=2e-12)
    local = np.c_[np.zeros((len(root), 6)), (root-1)/.01]
    value = evaluate(candidate, obs, sigma, theta, local)
    info = profile_information(value['shared'], value['state'], theta)
    assert info['rank'] == 6
    assert np.isinf(info['physical_standard_errors'][0])
    assert np.isinf(info['physical_standard_errors'][4])


def test_noiseless_observed_only_recovery_known_boundary():
    candidate, theta, obs, sigma, domain, root, physical = fixture(48)
    result = fit(candidate, obs, sigma, domain, root_known=root, max_iter=120)
    assert result['qualified'], result['attempts']
    assert result['loss'] < 1e-10
    r, x = polar_lines(theta)
    np.testing.assert_allclose(result['r'], r, atol=2e-6)
    np.testing.assert_allclose(result['x'], x, atol=2e-6)
    assert result['information']['rank'] == 8


def test_competitor_positive_edge_boundary_is_optimized():
    candidate, theta, obs, sigma, domain, root, physical = fixture(48)
    competitor = lk.RadialCandidate.from_supports('pair01',3,((0,1),))
    result = fit(competitor,obs,sigma,domain,root_known=root,max_iter=160)
    assert result['qualified'],result['attempts']
    assert result['projected_gradient'] < 1e-4
    assert result['information']['active_line_coordinate_count'] > 0
    assert not result['information']['interior_CRLB_interpretation_applicable']
