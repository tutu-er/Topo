"""Synthetic day-ahead publication and forecast-conditioned paper15 residuals.

All powers are p.u.; each (96, 30) trajectory has 15 P followed by 15 Q
values per time slot, in the frozen terminal order. Only publish_forecast
accepts a hidden test trajectory. The two residual models share one
posterior, including their historical power/reference-voltage dependence.
"""
import numpy as np


SLOTS, TERMINALS, DAYS = 96, 15, 3
CHANNELS = 2 * TERMINALS
AR_COEFFICIENT = 0.8
SPATIAL_CORRELATION = 0.25
PEAK = (np.arange(SLOTS) >= 17 * 4) & (np.arange(SLOTS) < 22 * 4)
BIAS = np.where(PEAK, -0.05, 0.0)
RELATIVE_SD = np.where(PEAK, 0.08, 0.04)

# Orthonormal contrasts in the same day order as the three columns of
# make_packet's centered L_C and L_F. The omitted all-ones direction is null.
DAY_CONTRAST = np.array([[1., 1.], [-1., 1.], [0., -2.]]) / np.sqrt([2., 6.])


def _array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} requires finite values and shape {shape}')
    return value


def training_scale(training_pq):
    """Per-channel RMS across all three full training days and within-day slots."""
    training = _array(training_pq, (DAYS, SLOTS, CHANNELS), 'training_pq')
    return np.sqrt(np.mean(training ** 2, axis=(0, 1)))


def _spatial_cholesky(channels):
    correlation = np.full((channels, channels), SPATIAL_CORRELATION)
    np.fill_diagonal(correlation, 1.)
    return np.linalg.cholesky(correlation)


def publish_forecast(natural_pq, training_rms, rng):
    """Evaluator publishes only the curve, with no private truth/error/seed fields."""
    truth = _array(natural_pq, (SLOTS, CHANNELS), 'natural_pq')
    scale = _array(training_rms, (CHANNELS,), 'training_rms')
    if np.any(scale < 0) or not isinstance(rng, np.random.Generator):
        raise ValueError('requires nonnegative training RMS and a numpy Generator')

    innovations = rng.standard_normal((SLOTS, CHANNELS)) @ _spatial_cholesky(CHANNELS).T
    colored = innovations.copy()
    for t in range(1, SLOTS):
        colored[t] = AR_COEFFICIENT * colored[t - 1] + np.sqrt(1 - AR_COEFFICIENT**2) * innovations[t]
    forecast = truth + scale * (BIAS[:, None] + RELATIVE_SD[:, None] * colored)
    # Publication also masks inactive channels; zero measurement noise must
    # not reveal an unexpected nonzero held-out load in an unmodeled channel.
    forecast[:, scale == 0] = 0.
    return forecast


def _whiten(columns, scale):
    """Apply the inverse forecast-noise factor to a time/channel/column array."""
    standardized = columns / (RELATIVE_SD[:, None, None] * scale[None, :, None])
    innovations = standardized.copy()
    innovations[1:] = ((standardized[1:] - AR_COEFFICIENT * standardized[:-1])
                       / np.sqrt(1 - AR_COEFFICIENT**2))
    channels = scale.size
    rhs = innovations.transpose(1, 0, 2).reshape(channels, -1)
    solved = np.linalg.solve(_spatial_cholesky(channels), rhs)
    return solved.reshape(channels, SLOTS, -1).transpose(1, 0, 2).reshape(-1, columns.shape[-1])


def condition_packet(packet, training_pq, forecast):
    """Condition the shared two-dimensional training-day factor on the published curve.

    training_pq must follow the same three days and terminal order used by the
    original make_packet. Returns a new dict; unchanged frozen objects are
    shared read-only. This function has no access to the test generator.
    """
    training = _array(training_pq, (DAYS, SLOTS, CHANNELS), 'training_pq')
    published = _array(forecast, (SLOTS, CHANNELS), 'forecast')
    scale = training_scale(training)
    active = scale > 0
    precision = np.eye(2)
    information = np.zeros(2)
    if np.any(active):
        mean = training.mean(axis=0)
        day_columns = (training - mean).transpose(1, 2, 0) / np.sqrt(DAYS - 1)
        power_factor = day_columns @ DAY_CONTRAST
        centered_forecast = published - mean - BIAS[:, None] * scale
        columns = np.concatenate((power_factor, centered_forecast[:, :, None]), axis=2)
        white = _whiten(columns[:, active], scale[active])
        design, obs = white[:, :2], white[:, 2]
        precision += design.T @ design
        information = design.T @ obs

    latent_mean = np.linalg.solve(precision, information)
    # Marginal predictive squared distance, using the two-dimensional solve.
    # Under the specified forecast law its expected value per active entry is one.
    distance_per_entry = None
    if np.any(active):
        whitened_residual = obs - design @ latent_mean
        distance_per_entry = float((whitened_residual @ whitened_residual
                                    + latent_mean @ latent_mean) / obs.size)
    # For a precision LL', L^{-T} is a covariance factor.
    covariance_factor = np.linalg.solve(np.linalg.cholesky(precision).T, np.eye(2))
    output = dict(packet)
    cells = SLOTS * TERMINALS
    for candidate in ('C', 'F'):
        mean = _array(packet['mu_' + candidate], (cells,), 'residual mean')
        historical = _array(packet['L_' + candidate], (cells, DAYS), 'three residual day columns')
        tolerance = 128 * np.finfo(float).eps * max(1., np.max(np.abs(mean)), np.max(np.abs(historical)))
        if np.max(np.abs(historical.sum(axis=1))) > tolerance:
            raise ValueError('expected centered make_packet day columns')
        residual_factor = historical @ DAY_CONTRAST
        output['mu_' + candidate] = mean + residual_factor @ latent_mean
        output['L_' + candidate] = residual_factor @ covariance_factor
    output['forecast_conditioning'] = dict(
        active_channels=np.flatnonzero(active).tolist(),
        forecast_distance_per_entry=distance_per_entry,
        latent_mean=latent_mean.tolist(),
        latent_covariance=(covariance_factor @ covariance_factor.T).tolist(),
    )
    return output
