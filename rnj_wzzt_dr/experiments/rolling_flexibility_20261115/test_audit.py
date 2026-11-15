"""Small counterexamples for the independent rolling audit arithmetic."""
import numpy as np
import pytest

from . import audit


def test_family_min_after_accumulation_not_sum_of_per_step_minima():
    # Each parameter fits one row; neither can fit both. A false time-varying
    # parameter would report zero family loss instead of the true loss one.
    H = np.array([[[0., 0.]], [[1., 0.]], [[.5, 0.]]])
    result = audit.cumulative_scores(H, [0, 0, 0], [[1], [1]], [[0], [0]], [[0], [1]],
        internal=[0, 1, 1], families=['A', 'A', 'B'], initial_cells=2,
        sigma2=1., reference_mse=1., tie=1e-12)
    np.testing.assert_array_equal(result['snapshots'][-1]['sse'], [1, 1, .5])
    np.testing.assert_array_equal(result['snapshots'][-1]['profile']['sse'], [1, .5])
    assert result['snapshots'][-1]['native_target'] == 2


def test_duplicate_native_point_does_not_add_a_family_prior():
    a = audit.profile_scores([2., 4.], ['A', 'B'], 1.)
    b = audit.profile_scores([2., 2., 4.], ['A', 'A', 'B'], 1.)
    np.testing.assert_array_equal(a['weights'], b['weights'])


def test_all_96_prefixes_preserve_initial_history_exactly_once():
    H = np.zeros((2, 1, 2))
    result = audit.cumulative_scores(H, [3., 4.], np.zeros((96, 1)), np.zeros((96, 1)),
        np.ones((96, 1)), internal=[2, 1], families=['A', 'B'], initial_cells=96,
        sigma2=1., reference_mse=1., tie=0.)
    assert len(result['snapshots']) == 97
    for k, row in enumerate(result['snapshots']):
        np.testing.assert_array_equal(row['sse'], [3.+k, 4.+k])
        assert row['cells'] == 96+k


def test_same_family_may_retain_different_zero_weight_support_counts():
    selected = audit.strict_winner([1., 1.], [3, 2], cells=2, reference_mse=1., tie=0.)
    assert selected == 1
    assert audit.profile_scores([1., 1.], ['A', 'A'], 1.)['points'] == [0]


def test_firm_service_is_not_multiplied_and_pq_share_daily_gain():
    resources = [dict(physical_device_id='battery', action_unit='kW'),
                 dict(physical_device_id='battery', action_unit='kvar')]
    result = audit.reconstruct_delivery([[2., 0.], [-1., 0.]], [[1., 2.], [0., -2.]],
                                        {'battery': .9}, resources)
    np.testing.assert_allclose(result['actual_total'], [[2.9, 1.8], [-1., -1.8]], atol=0)
    with pytest.raises(ValueError, match='share'):
        audit.reconstruct_delivery([[0., 0.]], [[1., 1.]], [.9, 1.], resources)


def test_capacity_is_daily_peak_once_and_service_compensation_is_paid():
    requested = np.array([[1., 0.], [2., 1.], [-1., -3.], [0., 2.]])
    actual = requested.copy()  # Includes a physical-service-only third step.
    r = audit.cumulative_fees(requested, actual, np.ones_like(actual), np.ones_like(actual), [.5, .5])
    assert r['cumulative_availability'][-1] == 2.5
    assert r['cumulative_activation'][-1] == .25*10.
    np.testing.assert_array_equal(r['availability_by_step'], [.5, 1., 1., 0.])
    assert r['cumulative_total'][-1] == 5.


def test_net_total_change_not_separate_service_plus_extra_absolute_cost():
    # The agreed charge is relative to original net device baseline; opposite
    # service/additional adjustments must be combined before the absolute value.
    total = np.array([[2.]])+np.array([[-1.]])
    r = audit.cumulative_fees(total, total, [[1.]], [[1.]], [0.])
    assert r['cumulative_activation'][-1] == .25


def test_thermal_quarter_hour_matches_original_hour_not_four_hourly_steps():
    d = dict(hourly_temperature_retention=.98, heat_capacity_kwh_per_C=2.,
             heat_demand_kw=[.4]*4, ambient_temperature_C=20.,
             initial_temperature_C=45., cop=1.)
    states = audit.thermal_states(d, [.8]*4)
    expected = 20+.98*(45-20)+(.8-.4)/2
    assert states[-1] == pytest.approx(expected, abs=2e-14)


def test_battery_prefix_keeps_realized_energy_before_recovery():
    d = dict(initial_soc=.5, energy_kwh=10., charging_efficiency=.95, discharging_efficiency=.95)
    soc = audit.battery_states(d, [2., -2*.95**2])
    assert soc[1] == pytest.approx(.5475)
    assert soc[-1] == pytest.approx(.5)


def test_reserved_tail_capacity_is_paid_even_if_later_cancelled():
    # First request commits a future 4kW peak, although actual dispatch remains
    # 1kW and the next plan is zero; this contract does not refund capacity.
    req = np.array([[1.], [0.]])
    r = audit.cumulative_fees(req, req, np.ones_like(req), np.ones_like(req), [.5],
                             reserved_tail_peaks=[[4.], [0.]])
    np.testing.assert_array_equal(r['availability_by_step'], [2., 0.])
    assert r['cumulative_total'][-1] == 2.25


def test_monitor_union_does_not_fill_the_gap_between_separate_intervals():
    H = np.array([[[1., 0.]], [[0., 0.]]])
    means = np.array([[[0., 0.]], [[10., 0.]]])
    v = audit.monitor_values(H, means, 1., np.zeros((1, 1)), (0, 1), [1., 0.], [5.])
    np.testing.assert_array_equal(v['mean_under_each_native'], [-.5, 9.5])
    assert v['actual_new_logodds'] == 4.5
    assert v['outside_union']


def test_two_hour_monitor_requires_eight_quarters_and_eight_for_recovery():
    state = dict(state='normal', outside_streak=0, recovery_streak=0)
    outside = dict(outside_union=True, target_mean_explains=False)
    inside = dict(outside_union=False, target_mean_explains=True)
    for k in range(8):
        state = audit.feedback_transition(state, outside, changed=False)
        assert state['state'] == ('caution' if k < 7 else 'pause')
    for k in range(8):
        state = audit.feedback_transition(state, inside, changed=False)
        assert state['state'] == ('pause' if k < 7 else 'caution')


def test_losing_ev_energy_near_departure_is_not_a_feasible_zero_fallback():
    clock = (8+(np.arange(96)+.5)*.25) % 24
    baseline = np.where((clock >= 22) | (clock < 4), 1., 0.)
    d = dict(id='ev', kind='ev', node_label=1, power_kw=2., baseline_p_kw=baseline,
             baseline_q_kvar=np.zeros(96), connected_clock_hours=[18, 7],
             charging_efficiency=.95, required_battery_energy_kwh=5.7)
    resources = [dict(physical_device_id='ev', action_unit='kW', node_label=1)]
    prefix = np.zeros((92, 1)); prefix[56, 0] = -.5
    earlier = audit.device_prefix({'devices':[d]}, resources, prefix[:57])
    after_departure = audit.device_prefix({'devices':[d]}, resources, prefix)
    assert earlier['passed']
    assert not after_departure['passed']
    assert after_departure['max_box_violation'] == 0.
    assert after_departure['max_reachability_violation'] == pytest.approx(.11875)


def test_progress_epoch_reset_can_precede_later_first_feasible_plan():
    assert audit.epoch_transition_allowed(0, 1, {1:71}, 72)
    assert not audit.epoch_transition_allowed(0, 1, {}, 72)
    assert not audit.epoch_transition_allowed(0, 1, {1:73}, 72)
    assert audit.epoch_transition_allowed(None, 0, {}, 62)


def test_negative_quadratic_box_certificate_encloses_all_corner_values():
    q = np.array([[[-.2, .1], [.1, -.1]]])
    a = np.array([[.1, -.1]])
    lo, hi = np.array([[-1., -2.]]), np.array([[1., 2.]])
    upper = audit.independent_quadratic_box_upper(q, a, -10., lo, hi)
    for x in np.linspace(-1, 1, 11):
        for y in np.linspace(-2, 2, 11):
            u = np.array([x,y])
            assert u@q[0]@u+2*a[0]@u-10 <= upper
    assert upper < 0
