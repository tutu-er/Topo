"""Independent information-boundary and pairing checks for broad EV cases.

These tests generate physical cases and forecasts only.  They never select an
action, fit parameters or access an evaluation output directory.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_systematic_cases as cases


def _assert_same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.mark.parametrize('case_id', ['ordinary96', 'train8', 'case33local12', 'legacy_n5_12'])
def test_forecast_needs_only_public_config_training_and_training_fit(case_id):
    case = cases.make_case(case_id, 996091, 1)
    H = case['train'].t
    fits = {k.name: SimpleNamespace(source_p=np.linspace(.03, .07, H),
                                   source_q=np.linspace(.01, .02, H)) for k in case['bank']}
    # This view has no validation, simulator, future inputs, actual noise or
    # true topology.  Its forecast must exactly equal the complete container's.
    clean = cases.forecast_for(case['train'], case['tm'], fits, case['config'])
    poisoned = deepcopy(case)
    poisoned['truth'] = {'r': np.array([123.]), 'x': np.array([456.]), 'candidate': 'oracle'}
    for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
        poisoned[key][100:] = 999.
    for value in poisoned['white'].values():
        value[:] = 888.
    poisoned['validation'] = replace(case['validation'], v=case['validation'].v + 1.)
    _assert_same(clean, cases.forecast_for(poisoned['train'], poisoned['tm'], fits, poisoned['config']))


def test_training_lengths_share_all_meter_channels_validation_and_future():
    full = cases.make_case('ordinary96', 996092, 2)
    for case_id in ('train8', 'train32'):
        short = cases.make_case(case_id, 996092, 2)
        H = short['train'].t
        for key in ('p', 'q', 'v', 'master_p', 'master_q', 'root_v'):
            np.testing.assert_array_equal(getattr(short['train'], key), getattr(full['train'], key)[-H:])
            np.testing.assert_array_equal(getattr(short['validation'], key), getattr(full['validation'], key))
            np.testing.assert_array_equal(getattr(cases.observe_future(short, np.zeros((6, 3))), key),
                                          getattr(cases.observe_future(full, np.zeros((6, 3))), key))
        for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
            np.testing.assert_array_equal(short[key], full[key])
        for key in full['white']:
            np.testing.assert_array_equal(short['white'][key], full['white'][key])


@pytest.mark.parametrize('case_id', ['ordinary96', 'alternating96', 'long_horizon96', 'legacy_n5_12'])
def test_each_future_channel_keeps_the_declared_sampling_budget(case_id):
    case = cases.make_case(case_id, 996093, 1)
    T, n = case['config']['horizon'], case['train'].n
    meter, future = case['tm'], case['fm']
    # All single-read sigmas are constant in these synthetic models.  The sum
    # of inverse variance multipliers is 30 per terminal/channel, independently
    # of the number of future time slots and the chosen EV action.
    for channel in ('p', 'q', 'v'):
        sigma = float(np.asarray(getattr(meter, channel + '_sigma')).ravel()[0])
        after = np.broadcast_to(getattr(future, channel + '_sigma'), (T, n))
        np.testing.assert_allclose(np.sum((sigma / after) ** 2, axis=0), np.full(n, 30.), atol=1e-12)
        known_bias = np.broadcast_to(getattr(meter, 'bias_' + channel), (case['train'].t, n))
        np.testing.assert_array_equal(known_bias, np.tile(known_bias[0], (case['train'].t, 1)))
        np.testing.assert_array_equal(np.broadcast_to(getattr(future, 'bias_' + channel), (T, n)),
                                      np.tile(known_bias[0], (T, 1)))
    for channel in ('master_p', 'master_q'):
        sigma = float(np.asarray(getattr(meter, channel + '_sigma')).ravel()[0])
        after = np.broadcast_to(getattr(future, channel + '_sigma'), (T,))
        assert np.sum((sigma / after) ** 2) == pytest.approx(30., abs=1e-12)


@pytest.mark.parametrize('case_id', ['ordinary96', 'case33local12', 'legacy_n5_12'])
def test_EV_is_added_once_in_physical_units_and_noise_is_paired(case_id):
    case = cases.make_case(case_id, 996094, 1)
    T, n = case['config']['horizon'], case['train'].n
    zero = np.zeros((T, n))
    power = zero.copy()
    power[:, 0] = np.linspace(-.5, 1., T)
    off = cases.observe_future(case, zero)
    on = cases.observe_future(case, power)
    np.testing.assert_allclose(on.p - off.p, power / case['base_kw'], atol=1e-15)
    np.testing.assert_array_equal(on.q, off.q)
    np.testing.assert_array_equal(on.root_v, off.root_v)
    # Removing meter noise changes the same output error for both actions.
    clean_off = cases.observe_future(case, zero, clean=True)
    clean_on = cases.observe_future(case, power, clean=True)
    for key in ('p', 'q', 'v', 'master_p', 'master_q'):
        np.testing.assert_allclose(getattr(on, key) - getattr(clean_on, key),
                                   getattr(off, key) - getattr(clean_off, key), atol=3e-16)


def test_standard_and_n5_true_family_are_read_from_supports_not_row_label():
    for case_id, expected in [('case33local12', ((0, 2),)),
                              ('legacy_n5_12', ((0, 1), (0, 1, 2)))]:
        actual = []
        for family_index in range(4):
            case = cases.make_case(case_id, 996095 + family_index, family_index)
            candidate = case['truth']['candidate']
            reduced = tuple(s for s in candidate.supports if 1 < len(s) < candidate.n)
            assert set(reduced) == set(expected)
            matching = [c.name for c in case['bank'] if c.supports == candidate.supports]
            assert len(matching) == 1
            actual.append(matching[0])
        assert len(set(actual)) == 1


def test_missing_truth_stress_is_predeclared_and_all_missing():
    declared = [c.supports for c in cases.candidate_bank('candidate_miss96')]
    for family_index in range(4):
        case = cases.make_case('candidate_miss96', 996101 + family_index, family_index)
        assert [c.supports for c in case['bank']] == declared
        assert case['truth']['candidate'].supports not in declared
        assert case['config']['stress']
    assert not set(cases.STRESS) & set(cases.FORMAL_CASES)


def test_driver_planning_boundary_is_invariant_to_hidden_future_and_truth(monkeypatch):
    from . import ev_systematic_study as study
    original = cases.make_case('ordinary96', 996109, 1)
    hidden_changed = deepcopy(original)
    for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
        hidden_changed[key][100:] = 999.
    hidden_changed['truth'] = dict(candidate=original['truth']['candidate'],
                                   r=original['truth']['r'] * 2., x=original['truth']['x'] * 2.)
    hidden_changed['validation'] = replace(original['validation'], v=original['validation'].v + .5)
    for values in hidden_changed['white'].values():
        values[100:] = -777.
    captured = []

    def trained(bank, observations, meter, cfg, seed):
        fits = {candidate.name: SimpleNamespace(
            r=np.full(candidate.m, .003), x=np.full(candidate.m, .001),
            source_p=np.full(observations.t, .05), source_q=np.full(observations.t, .02),
            loss=1., success=True) for candidate in bank}
        return fits, dict(all_complete=True, fits=[], failures={})

    def planning(training, bank, fits, tm, fm, forecast, **kwargs):
        captured.append(dict(training=study.io.observations_record(training),
            bank=[(k.name, k.supports) for k in bank], fits={k:vars(v) for k, v in fits.items()},
            tm=study.meter_to_record(tm), fm=study.meter_to_record(fm),
            forecast=forecast, public=kwargs))
        raise StopIteration('stop before simulated execution and inference')

    monkeypatch.setattr(study.rx, 'train_bank', trained)
    monkeypatch.setattr(study, 'select_plans', planning)
    spec=dict(phase='development', index=0, case_id='ordinary96', seed=996109, family_index=1)
    for container in (original, hidden_changed):
        monkeypatch.setattr(study.cases, 'make_case', lambda *args, _container=container: _container)
        with pytest.raises(StopIteration, match='stop before simulated'):
            study._run(spec)
    _assert_same(captured[0], captured[1])


def test_formal_sampling_is_disjoint_iid_regime_truth_plan_and_dev_units_are_deduplicated():
    from . import ev_systematic_study as study
    development = [study.specification('development', i) for i in range(64)]
    calibration = [study.specification('calibration', i) for i in range(108)]
    evaluation = [study.specification('evaluation', i) for i in range(72)]
    seeds = [set(row['seed'] for row in phase) for phase in (development, calibration, evaluation)]
    assert len(seeds[0]) == 52  # 8/32/96 and weak numeric/asset share paired seeds.
    assert len(seeds[1]) == 108 and len(seeds[2]) == 72
    assert not (seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2])
    for phase in (calibration, evaluation):
        for row in phase:
            assert row['case_id'] in cases.FORMAL_CASES
            assert row['case_id'] not in cases.STRESS
            assert row['family_index'] in range(4)
            # Rebuild the predeclared selection stream, separately from the
            # physical generation streams and before seeing any data.
            rng = np.random.default_rng(np.random.SeedSequence([row['seed'], 87231]))
            expected = cases.FORMAL_CASES[int(rng.integers(len(cases.FORMAL_CASES)))]
            assert row['case_id'] == expected
            assert row['family_index'] == int(rng.integers(4))


def test_freeze_binds_new_scalar_and_standard_parameter_implementation():
    from hashlib import sha256
    from . import ev_systematic_study as study
    values = study.source_hashes()
    for name in ('ev_scalar_distance_milp.py', 'ev_physical_cases.py', 'ev_systematic_cases.py',
                 'ev_systematic_refit.py', 'ev_systematic_design.py', 'ev_systematic_study.py'):
        assert name in values, 'missing actual computational dependency: ' + name
        assert values[name] == sha256((study.HERE / name).read_bytes()).hexdigest()
