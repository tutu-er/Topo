"""Check observation boundaries, full-bank selection and delivery invariants."""
import numpy as np
import json
import pytest

from ...observations import Observation
from ..customer_flexibility_20261113.model import CustomerCase
from .config import grid, resource_configs
from .execution import native_selection, fixed_rotation
from .simulation import sample_history, forecast, response
from .fitting import native_json_record


def day(offset=0):
    p = np.repeat((np.arange(96)+offset)[:, None], 2, axis=1)
    return Observation(p, p*.2, p*.01, (101, 102), np.ones_like(p), np.ones(96))


def test_full_day_cadence_and_explicit_validation_boundaries():
    for n in (48, 96, 192):
        all_o, train, validation = sample_history([day(), day(96)] if n == 192 else [day()], n)
        assert all_o.points == n and train.points == 2*n//3
        assert validation.points == n//3
        assert np.array_equal(all_o.p, np.concatenate((train.p, validation.p)))
        if n == 48:
            assert np.array_equal(all_o.p[:, 0], np.arange(0, 96, 2))
        else:
            assert np.array_equal(all_o.p[:, 0], np.arange(n))


def test_prelocked_grid_and_higher_adoption_have_declared_size():
    assert len(grid(0)) == len(grid(1)) == 36
    assert not set(r['id'] for r in grid(0)) & set(r['id'] for r in grid(1))
    configs = resource_configs()
    old, new = CustomerCase(configs['current']), CustomerCase(configs['higher'])
    assert (old.R, new.R) == (10, 12)
    assert len(set(new.node)) == 9
    assert new.audit(np.zeros(new.shape))['passed']
    pattern, _ = fixed_rotation(new)
    for replicate in (100000, 200000):
        eta = response(new.R, replicate)
        assert np.all((eta >= .9) & (eta <= 1))
        assert new.audit(pattern*eta[None])['passed']
        assert new.audit(.17*pattern*eta[None])['passed']
        assert np.isclose(new.fee(.17*pattern), .17*new.fee(pattern))


def test_forecast_is_noisier_than_meter_and_is_paired_by_seed():
    b = dict(p=np.ones((96, 15)), q=np.ones((96, 15))*.3)
    a, same, independent = forecast(b, 123), forecast(b, 123), forecast(b, 124)
    true = np.column_stack((b['p'], b['q']))
    assert np.array_equal(a, same) and not np.array_equal(a, independent)
    assert .04 < np.std(a/true-1) < .06


def test_native_selection_uses_all_raw_points_and_original_tie_rule():
    # The duplicate representative is deliberately absent from this raw-bank
    # test: native strict MSE must still inspect every actual output parameter.
    o = Observation(np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), (101, 102))
    s = dict(cells=4, reference_mse=1., validation_tie=1e-12, native_bank=[
        dict(H=np.zeros((2, 4)), history_sse=4., internal=1, topology_id='wrong', active_clades=[]),
        dict(H=np.array([[1., 0, 0, 0], [0, 1., 0, 0]]), history_sse=0., internal=2,
             topology_id='true', active_clades=[[101, 102]])])
    result = native_selection(s, o, dict(truth_id='true', truth_clades=[[101, 102]]))
    assert result['native_map_correct'] and result['native_selected_index'] == 1
    assert result['native_missing'] == result['native_extra'] == 0


def test_unbounded_native_diagnostics_are_recorded_without_mutating_coefficients():
    original = dict(events=[dict(solver=dict(bound=float('-inf'), gap=float('inf'), objective=float('nan')))],
                    frozen_candidates=[dict(R=np.array([[.12, .23]]), X=np.array([[.34, .45]]))])
    record = native_json_record(original)
    parsed = json.loads(json.dumps(record, allow_nan=False))
    assert parsed['events'][0]['solver'] == dict(bound=None, gap=None, objective=None)
    assert [r['value'] for r in parsed['_nonfinite_native_fields']] == ['-inf', 'inf', 'nan']
    assert parsed['_nonfinite_native_fields'][0]['path'] == 'native.events[0].solver.bound'
    assert np.isneginf(original['events'][0]['solver']['bound'])
    assert np.array_equal(record['frozen_candidates'][0]['R'], original['frozen_candidates'][0]['R'])


def test_nonfinite_fitted_coefficients_remain_an_error():
    with pytest.raises(ValueError, match='Frozen'):
        native_json_record(dict(frozen_candidates=[dict(R=[[float('inf')]], X=[[.1]])]))
