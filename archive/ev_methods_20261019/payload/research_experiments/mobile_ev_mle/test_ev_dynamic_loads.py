"""Independent identities, forecasts and non-anticipation for ordinary loads."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_dynamic_loads as loads


def _fixture(count=96, seed=271):
    path = loads.generate_background(seed, count + 10)
    bp, bq = np.array([-.00015, 0., .00015]), np.array([.0001, 0., -.0001])
    training = SimpleNamespace(p=path['p_kw'][:count] / 20. + bp,
        q=path['q_kvar'][:count] / 20. + bq, v=np.ones((count, 3)), root_v=np.ones(count))
    meter = SimpleNamespace(bias_p=bp, bias_q=bq)
    fits = {'a': SimpleNamespace(source_p=path['source_p_kw'][:count] / 20.,
        source_q=path['source_q_kvar'][:count] / 20., success=True)}
    return path, training, meter, fits


def test_generator_reproducible_nested_and_physical_pf_relation():
    short, full = loads.generate_background(27, 96), loads.generate_background(27, 106)
    repeated = loads.generate_background(27, 106)
    for key in ('p_kw', 'q_kvar', 'pf', 'source_p_kw', 'source_q_kvar', 'root_v'):
        assert np.array_equal(full[key], repeated[key])
        assert np.array_equal(short[key], full[key][:96])
    assert np.all(full['p_kw'] > 0.) and np.all(full['q_kvar'] > 0.)
    assert np.all((full['pf'] >= .94) & (full['pf'] <= .99))
    assert full['q_kvar'] == pytest.approx(full['p_kw'] * np.tan(np.arccos(full['pf'])))
    assert np.array_equal(full['root_v'], np.ones(106))
    assert full['metadata']['config']['field_data'] is False
    assert full['metadata']['source_interface'] == 1
    assert len(full['metadata']['child_seeds']) == 5


def test_generator_is_smooth_but_individual_changes_are_not_synchronous():
    path = loads.generate_background(425, 4096)
    p = path['p_kw']
    assert p.mean(axis=0) == pytest.approx([2.4, 2.8, 2.1], abs=.09)
    for j in range(3):
        assert .65 < np.corrcoef(p[:-1, j], p[1:, j])[0, 1] < .96
    cross = np.corrcoef(p, rowvar=False)[np.triu_indices(3, 1)]
    assert np.all(np.abs(cross) < .6)
    assert np.std(p[:, 0] - p[:, 1]) > .25
    assert np.corrcoef(path['source_p_kw'][:-1], path['source_p_kw'][1:])[0, 1] > .75


def test_mean_forecast_calibrates_broadcast_bias_and_preserves_pq_means():
    _, training, meter, fits = _fixture()
    # Meter offsets can be supplied as full time-by-terminal arrays.
    meter.bias_p = np.broadcast_to(meter.bias_p, training.p.shape).copy()
    meter.bias_q = np.broadcast_to(meter.bias_q, training.q.shape).copy()
    forecast = loads.fit_forecast(training, meter, fits, kind='mean')
    assert forecast['p'] == pytest.approx(np.tile((training.p - meter.bias_p).mean(axis=0), (6, 1)))
    assert forecast['q'] == pytest.approx(np.tile((training.q - meter.bias_q).mean(axis=0), (6, 1)))
    assert forecast['source_by_candidate']['a']['source_p'] == pytest.approx(np.mean(fits['a'].source_p))
    assert forecast['root'] == pytest.approx(np.ones(6))
    assert np.array_equal(forecast['root'], forecast['root_v'])


def test_ar1_gap_matches_a_hand_calculated_conditional_mean():
    count, phi, mean = 24, .8, 2.4
    p_kw = mean + .5 * phi ** np.arange(count)
    pf = .97
    p = np.tile(p_kw[:, None], (1, 3)) / 20.
    training = SimpleNamespace(p=p, q=p * np.tan(np.arccos(pf)))
    sp = .1 + .02 * phi ** np.arange(count)
    fits = {'a': dict(source_p=sp, source_q=np.full(count, .035))}
    forecast = loads.fit_forecast(training, {}, fits, horizon=6, gap=4)
    steps = np.arange(5, 11)
    expected = (mean + (p_kw[-1] - mean) * phi ** steps) / 20.
    assert forecast['p'][:, 0] == pytest.approx(expected, abs=1e-12)
    assert forecast['q'] == pytest.approx(forecast['p'] * np.tan(np.arccos(pf)), abs=1e-12)
    assert forecast['source_by_candidate']['a']['source_p'] == pytest.approx(.1 + (sp[-1] - .1) * phi ** steps, abs=1e-12)
    assert forecast['metadata']['forecast_steps'] == [5, 6, 7, 8, 9, 10]
    assert forecast['metadata']['ar1']['p']['phi'] == pytest.approx([phi] * 3)
    assert forecast['metadata']['ar1']['pf']['constant_columns'] == [True] * 3


def test_forecast_does_not_read_validation_truth_future_noise_or_generator(monkeypatch):
    path, training, meter, fits = _fixture()
    before = loads.fit_forecast(training, meter, fits)
    training.validation = object()
    training.truth = {'p': np.full((6, 3), 999.)}
    training.future_noise = np.full((6, 3), 1e12)
    path['p_kw'][96:] += 999.
    path['source_p_kw'][96:] *= 100.
    monkeypatch.setattr(loads, 'generate_background', lambda *a, **k: pytest.fail('forecast called simulator'))
    monkeypatch.setitem(loads.CONFIG, 'common_phi', -.99)
    after = loads.fit_forecast(training, meter, fits)
    for key in ('p', 'q', 'root', 'root_v'):
        assert np.array_equal(before[key], after[key])
    assert before['metadata'] == after['metadata']
    for key in ('uses_validation', 'uses_truth', 'uses_future_noise', 'generator_config_used_by_forecast'):
        assert after['metadata'][key] is False
    assert after['metadata']['training_only'] is True


def test_source_forecasts_are_candidate_specific_and_only_need_training_arrays():
    _, training, meter, fits = _fixture()
    fits['b'] = dict(source_p=np.full(96, .15), source_q=np.full(96, .05), success=True)
    forecast = loads.fit_forecast(training, meter, fits, kind='mean')
    assert forecast['source_by_candidate']['b']['source_p'] == pytest.approx(.15)
    assert not np.array_equal(forecast['source_by_candidate']['a']['source_p'], forecast['source_by_candidate']['b']['source_p'])
    changed = deepcopy(fits)
    changed['b']['source_p'][0] += .01
    after = loads.fit_forecast(training, meter, changed, kind='mean')
    assert forecast['metadata']['training_sha256'] == after['metadata']['training_sha256']
    assert forecast['metadata']['training_source_fits_sha256'] != after['metadata']['training_source_fits_sha256']


def test_ar_coefficient_and_physical_clips_are_recorded():
    p = np.tile((.01 * 1.2 ** np.arange(10))[:, None], (1, 3))
    training = SimpleNamespace(p=p, q=np.zeros_like(p))
    fits = {'a': dict(source_p=np.full(10, -.7), source_q=np.full(10, .7))}
    forecast = loads.fit_forecast(training, {}, fits, horizon=6, gap=4)
    estimate = forecast['metadata']['ar1']['p']
    assert estimate['phi_raw'] == pytest.approx([1.2] * 3)
    assert estimate['phi'] == pytest.approx([.98] * 3)
    assert estimate['phi_clipped'] == [True] * 3
    assert forecast['metadata']['clips']['training_implied_pf']['count'] == p.size
    assert forecast['source_by_candidate']['a']['source_p'] == pytest.approx(-.5)
    assert forecast['source_by_candidate']['a']['source_q'] == pytest.approx(.5)
    assert forecast['metadata']['clips']['source_by_candidate']['a']['source_p']['count'] == 6
    assert np.all(forecast['p'] >= .05 / 20.) and np.all(forecast['p'] <= 12. / 20.)


def test_negative_training_source_explanations_are_preserved_within_the_original_box():
    _, training, meter, _ = _fixture()
    source_p = np.linspace(-.12, -.08, 96)
    source_q = np.full(96, -.025)
    fits = {'wrong_candidate': dict(source_p=source_p, source_q=source_q)}
    mean = loads.fit_forecast(training, meter, fits, kind='mean')
    assert mean['source_by_candidate']['wrong_candidate']['source_p'] == pytest.approx(source_p.mean())
    assert mean['source_by_candidate']['wrong_candidate']['source_q'] == pytest.approx(source_q.mean())
    ar1 = loads.fit_forecast(training, meter, fits, kind='ar1')
    assert np.all(ar1['source_by_candidate']['wrong_candidate']['source_p'] < 0.)
    assert ar1['source_by_candidate']['wrong_candidate']['source_q'] == pytest.approx(-.025)
    for forecast in (mean, ar1):
        for channel in ('source_p', 'source_q'):
            assert forecast['metadata']['clips']['source_by_candidate']['wrong_candidate'][channel]['count'] == 0
    assert ar1['metadata']['config']['source_bounds_pu'] == [-.5, .5]


def test_inputs_are_not_mutated_and_invalid_inputs_are_rejected():
    _, training, meter, fits = _fixture()
    before = deepcopy((training, meter, fits))
    loads.fit_forecast(training, meter, fits)
    assert np.array_equal(training.p, before[0].p)
    assert np.array_equal(meter.bias_p, before[1].bias_p)
    assert np.array_equal(fits['a'].source_p, before[2]['a'].source_p)
    for kwargs in (dict(gap=-1), dict(horizon=0), dict(kind='oracle'), dict(base_kw=0.)):
        with pytest.raises(ValueError):
            loads.fit_forecast(training, meter, fits, **kwargs)
    with pytest.raises(ValueError, match='per training time'):
        loads.fit_forecast(training, meter, {'a': dict(source_p=np.ones(6), source_q=np.ones(6))})
    with pytest.raises(ValueError, match='successful'):
        loads.fit_forecast(training, meter, {'a': dict(success=False)})
    with pytest.raises(ValueError, match='finite'):
        loads.fit_forecast(training, dict(bias_p=np.nan), fits)
    for args in ((0, 0), (-1, 3), (0, 3, 0)):
        with pytest.raises(ValueError):
            loads.generate_background(*args)
