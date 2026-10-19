"""History-source reuse, full-block scoring, costs, and retained failures."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from .. import likelihood as lk
from .. import ev_systematic_refit as legacy
from . import inference


@pytest.fixture
def case():
    rng = np.random.default_rng(1901)
    candidate = lk.RadialCandidate.from_supports('local_cache_case', 3, [(0, 1)])
    r = np.linspace(.004, .008, candidate.m); x = .5*r
    p = rng.uniform(.02, .08, (16, 3)); q = p*rng.uniform(.1, .6, p.shape)
    sp = rng.uniform(-.01, .01, len(p)); sq = rng.uniform(-.004, .004, len(p))
    phy = lk.ac_predict(candidate, r, x, p, q, sp, sq)
    def block(rows):
        return lk.Observations(p[rows], q[rows], phy['v'][rows],
                               master_p=phy['master_p'][rows], master_q=phy['master_q'][rows])
    D, E = block(slice(0, 12)), block(slice(12, 16))
    meter = lk.MeasurementModel(.0001, .0001, .0001, master_p_sigma=.0001, master_q_sigma=.0001)
    options = dict(r_bounds=(0., .05), x_bounds=(0., .03), ratio_bounds=(.1, 1.),
        source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1), voltage_bounds=(.8, 1.2),
        multistart=1, max_nfev=300, tol=1e-9)
    history = inference.fit_history([candidate], D, meter, options, seed=19)
    assert history.complete
    return candidate, D, E, meter, options, history


def spy_fits(monkeypatch):
    original = inference.fit_rx_inputs; calls = []
    def fit(candidate, observations, model, **kwargs):
        calls.append(dict(rows=observations.t, fixed=kwargs.get('fixed_r') is not None))
        return original(candidate, observations, model, **kwargs)
    monkeypatch.setattr(inference, 'fit_rx_inputs', fit)
    return calls


def test_cached_history_exact_lines_skips_only_redundant_h_fit(case, monkeypatch):
    c, D, E, meter, options, history = case
    fitted = history.fits[c.name]; counter = {}; calls = spy_fits(monkeypatch)
    state, result = inference._source_witness(c, D, meter, E, meter, fitted.r, fitted.x,
        options, 10, counter, cached_history=fitted)
    assert result['historical_source_state_reused']
    assert result['history_source_cache_reason'] == 'same_candidate_exact_lines_successful_H_source'
    assert calls == [dict(rows=E.t, fixed=True)]
    assert counter['history_source_cache_hits'] == 1
    assert counter['fit_calls'] == 1
    assert np.array_equal(state['source_p'][:D.t], fitted.source_p)
    assert np.array_equal(state['source_q'][:D.t], fitted.source_q)
    assert len(state['source_p']) == D.t+E.t


def test_nearby_but_not_exact_rx_does_not_reuse(case, monkeypatch):
    c, D, E, meter, options, history = case
    fitted = history.fits[c.name]; r = fitted.r.copy()
    r[0] = np.nextafter(r[0], np.inf)
    counter = {}; calls = spy_fits(monkeypatch)
    _, result = inference._source_witness(c, D, meter, E, meter, r, fitted.x,
        options, 11, counter, cached_history=fitted)
    assert not result['historical_source_state_reused']
    assert result['history_source_cache_reason'] == 'different_exact_line_vector'
    assert [row['rows'] for row in calls] == [D.t, E.t]
    assert counter.get('history_source_cache_hits', 0) == 0
    assert counter['fit_calls'] == 2


@pytest.mark.parametrize('corruption,reason', [
    ('candidate', 'different_candidate'),
    ('failed', 'historical_optimizer_unsuccessful'),
    ('source_length', 'invalid_historical_source_shape_or_value'),
])
def test_invalid_historical_state_is_not_reused(case, monkeypatch, corruption, reason):
    c, D, E, meter, options, history = case
    cached = deepcopy(history.fits[c.name]); calls = spy_fits(monkeypatch)
    if corruption == 'candidate':
        cached.candidate = 'different_candidate'
    elif corruption == 'failed':
        cached.success = False
    else:
        cached.source_q = cached.source_q[:-1]
    _, result = inference._source_witness(c, D, meter, E, meter, cached.r, cached.x,
        options, 12, {}, cached_history=cached)
    assert not result['historical_source_state_reused']
    assert result['history_source_cache_reason'] == reason
    assert [row['rows'] for row in calls] == [D.t, E.t]


def test_evaluate_recomputes_h_plus_e_scores_and_counts_actual_fits(case, monkeypatch):
    c, D, E, meter, options, history = case
    calls = spy_fits(monkeypatch)
    evaluated = inference.evaluate_refits([c], D, meter, E, meter, history, options, seed=20)
    assert evaluated['complete']
    assert calls == [dict(rows=E.t, fixed=True), dict(rows=D.t+E.t, fixed=False)]
    assert evaluated['counters']['history_source_cache_hits'] == 1
    assert evaluated['counters']['fit_calls'] == 2
    assert evaluated['history_cost_included'] is False
    row = evaluated['comparisons'][c.name]
    all_data = lk.concat_observations(D, E)
    all_meter = legacy.concat_models((D, E), (meter, meter))
    for route in ('fixed', 'joint'):
        score = row[route]
        rebuilt = legacy._score(c, all_data, all_meter,
            {k:np.asarray(v) for k,v in score['state'].items()}, inference._options(options),
            (D.t, E.t), ('history', 'future'), {})
        assert score['loss'] == pytest.approx(rebuilt['loss'], abs=1e-10)
        assert score['loss'] == pytest.approx(sum(b['loss'] for b in score['blocks'].values()), abs=1e-10)
        assert score['blocks']['history']['time_count'] == D.t
        assert score['blocks']['future']['time_count'] == E.t
        assert score['residual_count'] == (D.t+E.t)*(D.n+2)
    assert row['joint_no_worse_than_fixed']


def test_cache_rejects_changed_history_meter_domain_or_candidate(case):
    c, D, E, meter, options, history = case
    changed = lk.Observations(D.p, D.q, D.v+1e-8,
                              master_p=D.master_p, master_q=D.master_q)
    with pytest.raises(ValueError, match='cache differs'):
        inference.evaluate_refits([c], changed, meter, E, meter, history, options)
    altered_options = dict(options, source_q_bounds=(-.2, .2))
    with pytest.raises(ValueError, match='cache differs'):
        inference.evaluate_refits([c], D, meter, E, meter, history, altered_options)
    changed_meter = replace(meter, v_sigma=.0002)
    with pytest.raises(ValueError, match='cache differs'):
        inference.evaluate_refits([c], D, changed_meter, E, meter, history, options)
    other = lk.RadialCandidate.from_supports('different_candidate', 3, [(1, 2)])
    with pytest.raises(ValueError, match='cache differs'):
        inference.evaluate_refits([other], D, meter, E, meter, history, options)


def test_cached_history_does_not_hide_a_new_future_voltage_mismatch(case):
    c, D, E, meter, options, history = case
    voltage = E.v.copy(); voltage[:, 0] += .001
    changed = lk.Observations(E.p, E.q, voltage, master_p=E.master_p, master_q=E.master_q)
    result = inference.evaluate_refits([c], D, meter, changed, meter, history, options, seed=21)
    fixed = result['comparisons'][c.name]['fixed']
    assert fixed['blocks']['future']['loss'] > 1.
    assert fixed['blocks']['history']['loss'] < 1e-4
    assert fixed['loss'] == pytest.approx(fixed['blocks']['history']['loss']+
                                        fixed['blocks']['future']['loss'], abs=1e-9)


def test_future_source_failure_remains_a_failure_with_cached_h(case, monkeypatch):
    c, D, E, meter, options, history = case
    def fail(*args, **kwargs):
        raise RuntimeError('deliberate future source numerical failure')
    monkeypatch.setattr(inference, 'fit_rx_inputs', fail)
    result = inference.evaluate_refits([c], D, meter, E, meter, history, options)
    assert not result['complete'] and result['keep_all_on_failure']
    assert result['retained_on_failure'] == [c.name]
    assert result['counters']['history_source_cache_hits'] == 1
    assert c.name in result['failures']
    assert 'deliberate future source' in result['comparisons'][c.name]['starts'][0]['error']
    assert not result['comparisons'][c.name]['success']
