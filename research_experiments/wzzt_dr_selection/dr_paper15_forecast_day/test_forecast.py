import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_paper15_forecast_day.forecast import (
    condition_packet, publish_forecast, training_scale,
)


def _history():
    rng = np.random.default_rng(9214)
    loads = np.zeros((3, 96, 30))
    loads[:, :, [1, 17]] = rng.normal(0.05, 0.002, (3, 96, 2))
    residuals = rng.normal(20., 1., (2, 3, 1440))
    packet = {'threshold': 9., 'effect_C': np.array([[2.]]), 'effect_F': np.array([[5.]])}
    for rows, key in zip(residuals, ('C', 'F')):
        packet['mu_' + key] = rows.mean(axis=0)
        packet['L_' + key] = (rows - rows.mean(axis=0)).T / np.sqrt(2)
    return loads, residuals, packet


def test_publication_matches_stationary_separable_law_and_masks_unseen_channels():
    loads, _, _ = _history()
    rms = training_scale(loads)
    natural = np.full((96, 30), .1)
    published = publish_forecast(natural, rms, np.random.default_rng(82))

    innovations = np.random.default_rng(82).standard_normal((96, 30))
    time = np.arange(96)
    active = np.flatnonzero(rms)
    temporal = np.linalg.cholesky(0.8 ** np.abs(time[:, None] - time))
    spatial = np.linalg.cholesky(.25 * np.ones((30, 30)) + .75 * np.eye(30))
    raw = temporal @ innovations @ spatial.T
    peak = (time >= 68) & (time < 88)
    expected = natural + rms * (np.where(peak, -.05, 0.)[:, None] +
                                np.where(peak, .08, .04)[:, None] * raw)
    np.testing.assert_allclose(published[:, active], expected[:, active])
    assert np.all(published[:, rms == 0] == 0.)
    assert np.all(natural == .1)


def test_condition_matches_dense_joint_gaussian_without_using_contrast_or_whitener():
    loads, residuals, packet = _history()
    rms = training_scale(loads)
    forecast = publish_forecast(loads[0], rms, np.random.default_rng(32))
    frozen_forecast, frozen_training = forecast.copy(), loads.copy()
    old = {k: v.copy() for k, v in packet.items() if isinstance(v, np.ndarray)}
    conditional = condition_packet(packet, loads, forecast)
    active = np.flatnonzero(rms)
    t = np.arange(96)
    peak = (t >= 68) & (t < 88)
    sd = (np.where(peak, .08, .04)[:, None] * rms[active]).ravel()
    corr = np.kron(0.8 ** np.abs(t[:, None] - t), [[1, .25], [.25, 1]])
    # Condition joint historical residual rows and P/Q directly in the
    # 192-observation space. Inspect cross-candidate as well as within-model covariances.
    powers = loads[:, :, active].reshape(3, -1)
    forecast_mean = powers.mean(0) + (np.where(peak, -.05, 0.)[:, None] * rms[active]).ravel()
    power_cov = np.cov(powers, rowvar=False)
    forecast_cov = power_cov + sd[:, None] * corr * sd[None, :]
    indices = np.array([0, 377, 1439])
    rows = np.concatenate((residuals[0][:, indices], residuals[1][:, indices]), axis=1)
    cross_cov = (rows - rows.mean(0)).T @ (powers - powers.mean(0)) / 2
    gain = np.linalg.solve(forecast_cov, cross_cov.T).T
    expected_mean = rows.mean(0) + gain @ (forecast[:, active].ravel() - forecast_mean)
    expected_cov = np.cov(rows, rowvar=False) - gain @ cross_cov.T
    joint_mean = np.r_[conditional['mu_C'][indices], conditional['mu_F'][indices]]
    joint_factor = np.vstack([conditional['L_C'][indices], conditional['L_F'][indices]])
    np.testing.assert_allclose(joint_mean, expected_mean, atol=5e-12)
    np.testing.assert_allclose(joint_factor @ joint_factor.T, expected_cov, atol=5e-12)
    assert conditional['forecast_conditioning']['active_channels'] == [1, 17]
    diff = forecast[:, active].ravel() - forecast_mean
    distance = (diff @ np.linalg.solve(forecast_cov, diff)) / diff.size
    assert conditional['forecast_conditioning']['forecast_distance_per_entry'] == pytest.approx(distance)
    np.testing.assert_array_equal(forecast, frozen_forecast)
    np.testing.assert_array_equal(loads, frozen_training)
    for key, value in old.items():
        np.testing.assert_array_equal(value, packet[key])
    assert 'forecast_conditioning' not in packet
    assert conditional['effect_C'] is packet['effect_C'] and conditional['threshold'] == packet['threshold']


def test_no_active_channel_leaves_prior_and_rejects_unaligned_columns():
    loads, _, packet = _history()
    conditional = condition_packet(packet, 0 * loads, np.ones((96, 30)))
    np.testing.assert_array_equal(conditional['forecast_conditioning']['latent_covariance'], np.eye(2))
    assert conditional['forecast_conditioning']['forecast_distance_per_entry'] is None
    for key in ('C', 'F'):
        np.testing.assert_array_equal(conditional['mu_' + key], packet['mu_' + key])
        old, new = packet['L_' + key][:7], conditional['L_' + key][:7]
        np.testing.assert_allclose(new @ new.T, old @ old.T)
    with pytest.raises(ValueError, match='centered'):
        condition_packet(dict(packet, L_C=packet['L_C'] + 1), loads, loads[0])
    with pytest.raises(ValueError, match='shape'):
        condition_packet(packet, loads[:, :, :15], loads[0])
    with pytest.raises(ValueError, match='RMS'):
        publish_forecast(loads[0], -np.ones(30), np.random.default_rng(2))
