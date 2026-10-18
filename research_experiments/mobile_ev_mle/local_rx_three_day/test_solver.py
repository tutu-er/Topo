import numpy as np
import pytest

from .. import likelihood as lk
from ..local_rx_identifiability.fitting import polar_lines, profile_information
from .solver import evaluate, fit, _step


def fixture(count=32):
    candidate = lk.RadialCandidate.from_supports('star', 3)
    r, x = np.array([.035, .021, .018, .026]), np.array([.025, .014, .012, .017])
    theta = np.r_[np.log(np.hypot(r, x)), np.arctan2(x, r)]
    rng = np.random.default_rng(73)
    p, q = rng.uniform(.015, .06, (count, 3)), rng.uniform(.004, .025, (count, 3))
    root = 1.+.003*np.sin(np.arange(count)/5)
    response = lk.ac_predict(candidate, r, x, p, q, root_v=root)
    obs = dict(p=p, q=q, v=response['v'])
    sigma = dict(p=np.full(3, .0003), q=np.full(3, .0001), v=np.full(3, .0005))
    domain = dict(z_min=.005, z_max=.12, ratio_min=.25, ratio_max=1.5,
                  power_caps=[.2]*3, root_min=.97, root_max=1.03, v_min=.9, v_max=1.1)
    local = np.c_[np.zeros((count, 6)), (root-1)/.01]
    return candidate, theta, obs, sigma, domain, root, local


def test_noisy_boundary_residual_and_exact_derivative():
    candidate, theta, obs, sigma, _, root, local = fixture(5)
    measured, sd = root+.0002, .0004
    result = evaluate(candidate, obs, sigma, theta, local, root_meter=(measured, sd))
    assert result['residual'].shape == (5, 10)
    np.testing.assert_allclose(result['residual'][:, -1], -.5)
    for column in (0, 4, 6):
        a, b = local.copy(), local.copy()
        a[:, column] += 1e-5; b[:, column] -= 1e-5
        numerical = (evaluate(candidate, obs, sigma, theta, a, root_meter=(measured, sd),
                              derivatives=False)['residual']-
                     evaluate(candidate, obs, sigma, theta, b, root_meter=(measured, sd),
                              derivatives=False)['residual'])/2e-5
        np.testing.assert_allclose(numerical, result['state'][:, :, column], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(result['shared'][:, -1], 0.)


def test_boundary_measurement_adds_information_without_freezing_boundary():
    candidate, theta, obs, sigma, _, root, local = fixture()
    unknown = evaluate(candidate, obs, sigma, theta, local)
    measured = evaluate(candidate, obs, sigma, theta, local, root_meter=(root, .0005))
    assert profile_information(unknown['shared'], unknown['state'], theta)['rank'] == 6
    assert profile_information(measured['shared'], measured['state'], theta)['rank'] == 8
    assert measured['state'].shape[2] == 7
    # Infinite meter variance tends to the unmeasured observation model.
    weak = evaluate(candidate, obs, sigma, theta, local, root_meter=(root+.001, 1e12))
    np.testing.assert_allclose(weak['residual'][:, :-1], unknown['residual'])
    assert abs(weak['residual'][:, -1]).max() < 1e-12


def test_complete_ac_steps_decrease_original_eiv_loss():
    candidate, theta, obs, sigma, domain, root, local = fixture(48)
    rng = np.random.default_rng(172)
    noisy = {name: values+rng.normal(size=values.shape)*sigma[name] for name, values in obs.items()}
    meter = root+rng.normal(size=len(root))*.0005
    result = fit(candidate, noisy, sigma, domain, root_meter=(meter, .0005),
                 max_iter=160, multistart=2)
    assert result['qualified'], result['attempts']
    for attempt in result['attempts']:
        assert np.all(np.diff(attempt['trace']) < 0.)
    assert result['trace'][-1] == result['loss']
    assert result['residual'] and len(result['residual'][0]) == 10
    assert result['ac_equation_residual'] < 2e-10
    assert not result['global_optimality_certified']
    # It fitted a latent boundary, instead of substituting noisy meter readings.
    assert np.max(abs(np.asarray(result['root_v'])-meter)) > 1e-7


def test_accurate_boundary_meter_limit_matches_known_boundary():
    candidate, theta, obs, sigma, domain, root, local = fixture(48)
    known = fit(candidate, obs, sigma, domain, root_known=root, multistart=1)
    metered = fit(candidate, obs, sigma, domain, root_meter=(root, 1e-6), multistart=1)
    assert known['qualified'] and metered['qualified'], (known['attempts'], metered['attempts'])
    np.testing.assert_allclose(metered['root_v'], root, atol=1e-8)
    np.testing.assert_allclose(metered['r'], known['r'], atol=2e-6)
    np.testing.assert_allclose(metered['x'], known['x'], atol=2e-6)


def test_observation_loss_is_exact_sum_of_declared_channels():
    candidate, theta, obs, sigma, _, root, local = fixture(4)
    local[:, 0] = .3; local[:, 4] = -.2; local[:, 6] += .07
    value = evaluate(candidate, obs, sigma, theta, local, root_meter=(root, .0005))
    expected = np.sum(((value['p']-obs['p'])/sigma['p'])**2)
    expected += np.sum(((value['q']-obs['q'])/sigma['q'])**2)
    expected += np.sum(((value['physical']['v']-obs['v'])/sigma['v'])**2)
    expected += np.sum(((value['root_v']-root)/.0005)**2)
    np.testing.assert_allclose(np.sum(value['residual']**2), expected, rtol=1e-13)


def test_boundary_model_inputs_are_unambiguous():
    candidate, theta, obs, sigma, domain, root, local = fixture(4)
    with pytest.raises(ValueError):
        fit(candidate, obs, sigma, domain, root_known=root, root_meter=(root, .001))
    with pytest.raises(ValueError):
        evaluate(candidate, obs, sigma, theta, local, root_meter=(root, 0.))


def test_optional_preconditioned_block_step_matches_dense_scaled_problem():
    rng = np.random.default_rng(19)
    a, c, residual = rng.normal(size=(3, 10, 4)), rng.normal(size=(3, 10, 7)), rng.normal(size=(3, 10))
    actual_a, actual_c = _step(a, c, residual, .07, True)
    sa = 1./np.linalg.norm(a, axis=(0, 1))
    sc = 1./np.linalg.norm(c, axis=1)
    matrix = np.zeros((30, 25))
    for time in range(3):
        matrix[10*time:10*(time+1), :4] = a[time]*sa
        matrix[10*time:10*(time+1), 4+7*time:11+7*time] = c[time]*sc[time]
    direct = np.linalg.solve(matrix.T@matrix+.07*np.eye(25), -matrix.T@residual.ravel())
    np.testing.assert_allclose(actual_a, sa*direct[:4], atol=1e-12)
    np.testing.assert_allclose(actual_c, sc*direct[4:].reshape(3, 7), atol=1e-12)
