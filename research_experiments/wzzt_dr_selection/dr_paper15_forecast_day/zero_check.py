"""Zero-contract loss means and analytic lower bounds using the published forecast only."""
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

if __package__:
    from .forecast import condition_packet
else:
    from forecast import condition_packet


def expected_absolute_loss(mean, factor):
    """Average folded-normal expectations; row-to-row independence is unnecessary."""
    deviation = np.linalg.norm(factor, axis=1)
    expected = np.abs(mean)
    random = deviation > 0
    standardized = expected[random] / deviation[random]
    erf = np.fromiter((math.erf(z / math.sqrt(2)) for z in standardized),
                      dtype=float, count=standardized.size)
    expected[random] = (deviation[random] * math.sqrt(2 / math.pi) * np.exp(-0.5 * standardized**2)
                        + np.abs(mean[random]) * erf)
    return float(expected.mean())


def zero_check(public):
    # The experiment's own trusted historical pickle and published forecast are
    # the only inputs. No test truth, meter, execution, or AC file is opened.
    with (public / 'history.pkl').open('rb') as source:
        history = pickle.load(source)
    forecast = np.load(public / 'forecast.npy', allow_pickle=False)
    packet = condition_packet(history['packet'], history['training'], forecast)
    coarse, fine = packet['mu_C'], packet['mu_F']
    coarse_factor, fine_factor = packet['L_C'], packet['L_F']
    if coarse_factor.shape[1] != 2 or fine_factor.shape != coarse_factor.shape:
        raise ValueError('Disk probability formula requires a shared two-dimensional standard normal')
    ratio = 1.03
    threshold = float(packet['threshold'])
    loss_c = expected_absolute_loss(coarse, coarse_factor)
    loss_f = expected_absolute_loss(fine, fine_factor)
    center = float(np.abs(coarse).mean() - ratio * np.abs(fine).mean())
    slope = (np.sign(coarse) @ coarse_factor - ratio * np.sign(fine) @ fine_factor) / coarse.size
    slope_norm = float(np.linalg.norm(slope))
    fine_norms = np.linalg.norm(fine_factor, axis=1)
    bounds = []
    for alpha in (0.05, 0.01):
        # P(||Z||<=r)=1-exp(-r²/2). The coarse absolute-value subgradient
        # gives a lower tangent; the fine sign-flip remainder on the disk is
        # at most 2 (r ||b_i|| - |mu_i|)_+ for each row.
        radius = math.sqrt(-2 * math.log(alpha))
        penalty = 2 * ratio * np.maximum(0., radius * fine_norms - np.abs(fine)).mean()
        lower = float(center - radius * slope_norm - penalty)
        bounds.append(dict(background_mass=1-alpha, radius=radius, fine_sign_flip_penalty=float(penalty),
                           loss_gap_lower=lower, exceeds_threshold=bool(lower > threshold)))
    return dict(conditional_on_published_forecast=True, units='1e-6 p.u.^2', threshold=threshold,
                expected_loss_coarse=loss_c, expected_loss_fine=loss_f,
                expected_loss_gap=loss_c - ratio * loss_f, loss_gap_at_center=center,
                slope_norm=slope_norm, disk_bounds=bounds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True, help='trusted prepared/public directory')
    print(json.dumps(zero_check(parser.parse_args().public), indent=2, allow_nan=False))
