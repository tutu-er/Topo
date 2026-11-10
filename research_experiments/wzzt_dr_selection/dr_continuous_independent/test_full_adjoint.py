"""Independent finite differences and existing-direction checks for coefficient adjoints."""
import json
from pathlib import Path

import numpy as np

from research_experiments.wzzt_dr_selection.dr_continuous_independent.continuous import IndependentResiduals
from research_experiments.wzzt_dr_selection.dr_continuous_independent.full_adjoint import probability_with_adjoint
from research_experiments.wzzt_dr_selection.dr_continuous_independent.gauss_frequency import gauss_probability
from research_experiments.wzzt_dr_selection.dr_heterogeneous_pilot.devices import Devices
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles


def test_coefficient_adjoint_matches_existing_directions_and_finite_difference():
    c, f, sigma = np.array([.5, .8, -.2]), np.array([0., .2, .1]), np.array([1., .7, .5])
    model = IndependentResiduals(c, f, sigma)
    normalization = model.sum_mean + .17, model.sum_sd
    settings = dict(cutoff=64., order=32, refinements=0, normalization=normalization)
    result = probability_with_adjoint(model, .03, **settings)
    direct = gauss_probability(model, .03, daC=np.eye(3), daF=np.zeros((3, 3)), **settings)
    assert abs(result['probability'] - direct['probability']) < 2e-14
    assert np.max(np.abs(result['gradient_coarse'] - direct['gradient'])) < 2e-13
    for key, values in [('gradient_coarse', c), ('gradient_fine', f)]:
        for coordinate in range(3):
            perturbation = np.eye(3)[coordinate] * 1e-6
            positive_c, negative_c = (values + perturbation, values - perturbation) if key == 'gradient_coarse' else (c, c)
            positive_f, negative_f = (values + perturbation, values - perturbation) if key == 'gradient_fine' else (f, f)
            positive = gauss_probability(IndependentResiduals(positive_c, positive_f, sigma), .03, **settings)['probability']
            negative = gauss_probability(IndependentResiduals(negative_c, negative_f, sigma), .03, **settings)['probability']
            assert abs((positive - negative) / 2e-6 - result[key][coordinate]) < 2e-7


def test_mixed_zero_noise_and_deterministic_strict_event():
    model = IndependentResiduals([1., .5], [.2, .1], [0., .6])
    settings = dict(cutoff=64., order=32, refinements=0, normalization=(model.sum_mean, model.sum_sd))
    result = probability_with_adjoint(model, .3, **settings)
    direct = gauss_probability(model, .3, daC=np.eye(2), daF=np.zeros((2, 2)), **settings)
    assert np.max(np.abs(result['gradient_coarse'] - direct['gradient'])) < 2e-13
    fixed = IndependentResiduals([1.], [0.], [0.])
    assert probability_with_adjoint(fixed, 1.)['probability'] == 0.
    assert not probability_with_adjoint(fixed, 1.)['gradient_valid']
    assert probability_with_adjoint(fixed, .9)['gradient_valid']


def test_device_hourly_vjp_chain_matches_probability_direction():
    folder = Path(__file__).resolve().parent.parent
    cfg = json.loads((folder / 'dr_heterogeneous_resources/library.json').read_text())
    node = dict(node_label=101, shiftable_kw=1., curtailable_kw=.5,
                ev_charger_kw=1., battery_power_kw=1., ev_grid_task_kwh=2.,
                cost_factor=dict(user=1., ev=1., battery=1.),
                sd_factor=dict(user=1., ev=1., battery=1.), user_q_over_p=.35)
    device = Devices(dict(nodes=[node]), cfg, profiles)
    rng = np.random.default_rng(2026111301)
    x = .02 * rng.normal(size=(4, 24))
    x[1] = -.03 - .01 * np.abs(x[1])
    effect_c = .03 * rng.normal(size=(2, 4))
    effect_f = .03 * rng.normal(size=(2, 4))
    baseline_c = np.full((96, 2), .4)
    baseline_f = np.full((96, 2), .1)
    sigma = np.full(192, .3)
    delivered, cache = device.nominal(x, cache=True)
    c = (baseline_c + delivered.T @ effect_c.T).ravel()
    f = (baseline_f + delivered.T @ effect_f.T).ravel()
    model = IndependentResiduals(c, f, sigma)
    settings = dict(cutoff=64., order=32, refinements=0, normalization=(model.sum_mean, model.sum_sd))
    threshold = model.moments()['mean']
    result = probability_with_adjoint(model, threshold, **settings)
    adjoint_u = (result['gradient_coarse'].reshape(96, 2) @ effect_c
                 + result['gradient_fine'].reshape(96, 2) @ effect_f).T
    adjoint_x = device.hourly_vjp(adjoint_u, cache)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    probabilities = []
    for sign in (1., -1.):
        u = device.nominal(x + sign * 1e-6 * direction)
        case = IndependentResiduals((baseline_c + u.T @ effect_c.T).ravel(),
                                    (baseline_f + u.T @ effect_f.T).ravel(), sigma)
        probabilities.append(gauss_probability(case, threshold, **settings)['probability'])
    difference = (probabilities[0] - probabilities[1]) / 2e-6
    assert abs(difference - np.sum(adjoint_x * direction)) < 2e-7
