"""Signed causal forecasts, strict data boundaries, and selector integration."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from .. import likelihood as lk
from ..full_pipeline.forecast import ForecastConfig
from ..g_precision.coefficients import score_tasks, QuadraticTask
from . import selector


def fixture():
    bank = [lk.RadialCandidate.from_supports('candidate_a', 3, [(0, 1)]),
            lk.RadialCandidate.from_supports('candidate_b', 3, [(1, 2)])]
    p = np.tile(np.array([.03, .04, .02]), (12, 1))
    q = -.1*p
    states = {c.name:SimpleNamespace(r=np.full(c.m, .004), x=np.full(c.m, .002),
        source_p=np.full(len(p), -.003), source_q=np.full(len(p), -.001)) for c in bank}
    phy = lk.ac_predict(bank[0], states[bank[0].name].r, states[bank[0].name].x, p, q, -.003, -.001)
    D = lk.Observations(p, q, phy['v'], master_p=phy['master_p'], master_q=phy['master_q'])
    meter = lk.MeasurementModel(.0001, .0001, .0001, master_p_sigma=.0001, master_q_sigma=.0001)
    timestamps = [t.isoformat() for t in pd.date_range('2020-01-01', periods=len(p), freq='15min', tz='UTC')]
    observed = dict(history=D, history_meter=meter, future_meter=meter, timestamps=timestamps,
                    horizon=12, base_kw=100., public_parameters=dict(base_kv=.4, base_kw=100.))
    options = dict(r_bounds=(0., .05), x_bounds=(0., .03), ratio_bounds=(.1, 1.),
        source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1), voltage_bounds=(.8, 1.2))
    historical = SimpleNamespace(complete=True, fits=states, options=options, failures={}, seconds=.0)
    historical.record = lambda:dict(complete=historical.complete, failures=historical.failures)
    config = dict(T=12, dt_hours=.25, passive=dict(max_candidates=12, bootstrap_replicates=0),
        iterations=1, segments=2, max_segments=2, solver_seconds=.05,
        total_seconds_per_domain=.12, relative_gap=.1, absolute_gap=1e-5, backend='gurobi')
    fc = dict(p=asdict(ForecastConfig(method='persistence')), q=asdict(ForecastConfig(method='persistence')))
    return bank, observed, options, historical, config, fc


def patch_history(monkeypatch, bank, history):
    def propose(D, meter, config, seed):
        return bank, dict(seconds=0., failures=[], config=config)
    monkeypatch.setattr(selector.passive, 'propose', propose)
    monkeypatch.setattr(selector.inference, 'fit_history', lambda *args, **kwargs:history)


def test_signed_forecasts_preserve_negative_terminal_and_source_values():
    bank, observed, options, history, config, fc = fixture()
    states = {name:vars(state) for name,state in history.fits.items()}
    f = selector.signed_forecast(observed['history'], observed['history_meter'], states,
        observed['timestamps'], horizon=12, base_kw=100., forecast_configs=fc,
        source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1))
    assert np.all(f['q'] < 0)
    assert np.all(f['source_by_candidate']['candidate_a']['source_p'] < 0)
    assert np.all(f['source_by_candidate']['candidate_a']['source_q'] < 0)
    assert f['metadata']['P']['nonnegative_clips'] == 0
    assert f['metadata']['Q']['nonnegative_clips'] == 0
    assert not f['metadata']['future_used'] and not f['metadata']['truth_used']


def test_distinct_p_q_configs_and_source_clipping_are_recorded():
    _, observed, options, history, _, fc = fixture()
    fc['q'] = asdict(ForecastConfig(method='ridge', ridge_alpha=100.))
    states = {name:vars(state) for name,state in history.fits.items()}
    f = selector.signed_forecast(observed['history'], observed['history_meter'], states,
        observed['timestamps'], horizon=12, base_kw=100., forecast_configs=fc,
        source_p_bounds=(-.002, .002), source_q_bounds=(-.1, .1))
    assert f['metadata']['forecast_configs']['P']['method'] == 'persistence'
    assert f['metadata']['forecast_configs']['Q']['ridge_alpha'] == 100.
    assert f['metadata']['source_domain_clips']['candidate_a']['source_p'] == 12
    assert np.all(f['source_by_candidate']['candidate_a']['source_p'] == -.002)


@pytest.mark.parametrize('extra', [dict(truth={}), dict(future_p_kw=np.zeros((12, 3))),
                                  dict(network=dict(branches=[]))])
def test_selector_rejects_simulator_information(extra):
    _, observed, _, _, config, _ = fixture()
    observed.update(extra)
    with pytest.raises(ValueError):
        selector._validate_observed(observed, config)


def test_operations_are_optional_mobility_and_match_the_public_service():
    template = selector.operations_config(dict(T=12, dt_hours=.25), 6)
    assert template.p_charge_max_kw == 11. and template.p_discharge_max_kw == 11.
    assert template.terminal_energy_kwh-template.initial_soc*template.battery_capacity_kwh == 6.
    assert template.mobile and template.max_moves == 1
    assert template.initial_site == 0 and template.initial_connected
    assert template.driver_budget == 30.
    assert template.max_throughput_kwh == 15.


def test_operations_read_current_ev_field_and_retain_agreeing_alias():
    config = dict(T=12, dt_hours=.25, ev=dict(p_charge_max_kw=7., max_moves=0),
                  ev_operations=dict(p_charge_max_kw=7., p_discharge_max_kw=3.))
    template = selector.operations_config(config, 3)
    assert template.p_charge_max_kw == 7.
    assert template.p_discharge_max_kw == 3.
    assert template.max_moves == 0


def test_conflicting_ev_configuration_is_rejected():
    config = dict(T=12, dt_hours=.25, ev=dict(p_charge_max_kw=7.),
                  ev_operations=dict(p_charge_max_kw=11.))
    with pytest.raises(ValueError, match='conflicting ev'):
        selector.operations_config(config, 3)


def test_current_protocol_ev_values_equal_the_preceding_pilot_defaults():
    from .config import CONFIG
    actual = selector.operations_config(CONFIG, 3)
    preceding = selector.operations_config({key:value for key,value in CONFIG.items() if key != 'ev'}, 3)
    assert asdict(actual) == asdict(preceding)


def test_selector_complete_history_plans_all_methods_and_nested_domains(monkeypatch):
    bank, observed, options, history, config, fc = fixture()
    patch_history(monkeypatch, bank, history)
    planning, selected_bank, fitted, domains = selector.select(observed, options, config, fc, 71)
    assert selected_bank is bank and fitted is history
    assert set(domains) == {'fixed_charge', 'fixed_bidir', 'mobile_charge', 'mobile_bidir'}
    assert set(planning['plans']) == {'OBS', 'CC', 'HL', 'RAND', 'P-SQ', 'P-SQ/F±',
                                      'G-SQ', 'G-SQ/F±', 'G-AQ', 'G-AQ/F±'}
    assert not planning['failures']
    assert planning['passive_pre']['public_coefficient_bound_value'] == .1
    assert not planning['selector_uses_true_future_load']
    assert set(planning['families']) == {'P-SQ', 'G-SQ', 'G-AQ'}
    assert len(planning['nominal_AC']) == 10
    assert not planning['context']['forecast']['metadata']['future_used']
    for method, plan in planning['plans'].items():
        assert np.asarray(plan['power']).shape == (12, 3)
        if method != 'OBS':
            assert plan['physicalcheck']['feasible']
    for method in ('G-SQ', 'G-AQ'):
        tasks = [QuadraticTask(**row) for row in planning['tasks'][method]]
        for suffix in ('', '/F±'):
            # This tiny physical fixture has legal proposals, so the inherited
            # same-domain square plan gives an independent objective floor.
            assert score_tasks(tasks, planning['plans'][method+suffix]['power'])+1e-6 >= score_tasks(
                tasks, planning['plans']['P-SQ'+suffix]['power'])


def test_required_history_failure_falls_back_without_dropping_a_competitor(monkeypatch):
    bank, observed, options, history, config, fc = fixture()
    history.complete = False
    history.failures = {'candidate_b':'retained test optimization failure'}
    patch_history(monkeypatch, bank, history)
    planning, selected_bank, _, _ = selector.select(observed, options, config, fc, 72)
    assert len(selected_bank) == 2
    assert 'response_or_G_planning' in planning['failures']
    assert not planning['history']['complete']
    for method in ('G-SQ', 'G-AQ'):
        for suffix in ('', '/F±'):
            assert np.array_equal(planning['plans'][method+suffix]['power'],
                                  planning['plans']['P-SQ'+suffix]['power'])
            assert 'fallback_reason' in planning['plans'][method+suffix]
