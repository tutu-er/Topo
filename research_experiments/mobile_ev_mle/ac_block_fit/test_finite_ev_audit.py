"""Independent finite differences at nonzero AC action, not affine outputs."""
from copy import deepcopy
import numpy as np
import pytest

from .test_fixed_source import fixture
from .finite_ev_audit import finite_response, pair_distance, summarize


@pytest.mark.parametrize('power', [1., 3.])
def test_nonzero_full_ac_response_derivative_and_fixed_history(power):
    candidate, data, meter, _, state = fixture(periods=5)
    saved = deepcopy(state); saved_p = data.p.copy(); saved_q = data.q.copy()
    count = {}
    actual = finite_response(candidate, data, meter, state, base_kw=100., power_kw=power, counter=count)
    eps = 1e-3
    plus = finite_response(candidate, data, meter, state, base_kw=100., power_kw=power+eps)
    minus = finite_response(candidate, data, meter, state, base_kw=100., power_kw=power-eps)
    numeric = (plus['mean']-minus['mean'])/(2*eps)
    np.testing.assert_allclose(actual['active_derivative_per_kw'], numeric, rtol=2e-6, atol=1e-10)
    assert count == dict(full_AC_replays=1, exact_derivative_evaluations=1)
    assert np.array_equal(data.p, saved_p) and np.array_equal(data.q, saved_q)
    assert all(np.array_equal(state[key], saved[key]) for key in state)
    assert actual['working_history_row'] == 4
    assert actual['source_p'][0] == state['source_p'][-1]
    assert actual['source_q'][0] == state['source_q'][-1]
    assert actual['Q_input'][0, 0] == data.q[-1, 0]
    assert abs(actual['P_input'][0, 0]-data.p[-1, 0]-power/100.) < 1e-15


@pytest.mark.parametrize('power', [1., 3.])
def test_pair_gradient_matches_independent_nonzero_power_ac_difference(power):
    candidate, data, meter, _, state = fixture(periods=5)
    changed = dict(state, r=state['r']*1.1, x=state['x']*.9)
    def value(u):
        responses = [finite_response(candidate, data, meter, theta, base_kw=100., power_kw=u)
                     for theta in (state, changed)]
        return pair_distance(*responses)
    actual = value(power); eps = 1e-3
    numeric = (value(power+eps)['distance']-value(power-eps)['distance'])/(2*eps)
    assert actual['distance'] > 0
    np.testing.assert_allclose(actual['active_derivative_per_kw'], numeric, rtol=2e-6, atol=1e-7)


def test_pair_does_not_compare_different_actions_or_meter_sigmas():
    candidate, data, meter, _, state = fixture(periods=3)
    first = finite_response(candidate, data, meter, state, base_kw=100., power_kw=1.)
    second = deepcopy(first); second['power_kw'] = 3.
    with pytest.raises(ValueError, match='actions differ'):
        pair_distance(first, second)
    second = deepcopy(first); second['sigma'] *= 2
    with pytest.raises(ValueError, match='precisions differ'):
        pair_distance(first, second)


def test_failed_response_retains_declared_candidate_pair_denominator():
    plan = dict(endpoint_manifest=[dict(network='n',candidate=c) for c in ('a','b')],
                powers_kw=[1.],relative_tolerance=1e-3,relative_denominator_floor=1e-12)
    rows = [dict(network='n',candidate=c,algorithm=m,power_kw=1.,
        common_qualified=True,available=False,failure='declared test AC failure')
        for c in ('a','b') for m in ('ref_jac','block_jac_gap')]
    out = summarize(rows, plan)
    assert out['endpoint_failures'] == 4
    assert len(out['candidate_comparisons']) == 2 and len(out['pair_comparisons']) == 1
    assert out['primary_pair_active_derivative'] == dict(denominator=1,available=0,
        passed=0,max_relative=None,max_absolute=None,near_zero_references=0)
