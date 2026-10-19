"""Independent analytic checks, with no frozen AC selector dependency."""
from dataclasses import replace
import numpy as np
import pytest
from .ev_flexibility_analytic import (CurveTask, profile_signal,
    local_rx_information, battery_accounting, solve_rank_one_curve,
    one_site_one_way_lp_audit, run_checks)


def test_history_profile_formula_matches_independent_residual_and_stationarity():
    a = np.array([.2,.7,-.1])
    h = 2.
    answer = profile_signal(a,h)
    b = answer['offset']
    assert abs(np.sum(b-a)+h*b) < 1e-14
    assert answer['distance_squared'] == pytest.approx(np.sum((a-b)**2)+h*b*b)
    assert answer['distance_squared'] == pytest.approx(answer['closed_form'])
    for shift in (-1.,-.05,.05,1.):
        assert np.sum((a-(b+shift))**2)+h*(b+shift)**2 > answer['distance_squared']


def test_constant_has_zero_signal_only_without_history_anchor():
    u = [.25,.25]
    assert profile_signal(u,0.)['distance_squared'] == 0.
    assert profile_signal(u,1.)['distance_squared'] == pytest.approx(1/24)
    assert profile_signal(u,4.)['distance_squared'] == pytest.approx(1/12)
    for h in (0.,1.,4.):
        gain = profile_signal([.5,0.],h)['distance_squared']-profile_signal(u,h)['distance_squared']
        assert gain == pytest.approx(.125)


def test_charge_only_milp_matches_hand_bound_independent_lp_and_small_grid():
    task = CurveTask()
    result = solve_rank_one_curve(task)
    assert result['success'] and result['includes_complete_curve_library'] is False
    np.testing.assert_allclose(sorted(result['net_power_kw']),[0.,.5],atol=2e-7)
    assert result['distance_squared'] == pytest.approx(.125)
    assert result['distance_best_bound'] == pytest.approx(.125)
    assert result['absolute_amplitude_gap'] == pytest.approx(0.,abs=2e-7)
    audits = one_site_one_way_lp_audit(task)
    assert all(row['success'] for row in audits)
    assert max(row['amplitude'] for row in audits) == pytest.approx(.5)
    grid = np.linspace(0.,.5,51)
    assert max(.5*(p-(.5-p))**2 for p in grid) == pytest.approx(result['distance_squared'])


def test_v2g_same_terminal_service_reports_net_throughput_and_conversion_losses():
    task = replace(CurveTask(),p_discharge_max=1.)
    result = solve_rank_one_curve(task)
    assert result['success']
    np.testing.assert_allclose(sorted(result['net_power_kw']),[-.405,1.],atol=2e-7)
    assert result['amplitude_best_bound'] == pytest.approx(1.405)
    assert result['distance_squared'] == pytest.approx(.9870125)
    account = result['battery']
    assert account['battery_gain_kwh'] == pytest.approx(.45)
    assert account['grid_net_kwh'] == pytest.approx(.595)
    assert account['grid_throughput_kwh'] == pytest.approx(1.405)
    assert account['conversion_loss_kwh'] == pytest.approx(.145)
    assert abs(account['balance_residual_kwh']) < 1e-14


def test_same_grid_energy_v2g_cycle_does_not_meet_same_battery_service():
    task = replace(CurveTask(),p_discharge_max=1.)
    account = battery_accounting([1.,-.5],task)
    assert account['grid_net_kwh'] == pytest.approx(.5)
    assert account['battery_gain_kwh'] == pytest.approx(.9-.5/.9)
    assert account['battery_gain_kwh'] < .45
    # Equality of both the one-way grid task and terminal battery task gives
    # (.9-1/.9)*D=0, so D=0 for these efficiencies.
    coefficient = task.eta_charge-1/task.eta_discharge
    rhs = .45-task.eta_charge*.5
    assert coefficient < 0 and rhs == 0.


def test_equal_post_driving_battery_domains_give_equal_ev_ess_electrical_curves():
    # EV and ESS have different initial/transport bookkeeping. Align their
    # physical battery state at connection, then compare the electrical task.
    ev = replace(CurveTask(),p_discharge_max=1.,energy_min=1.9,driving_energy=.1)
    ess = replace(ev,initial_energy=1.9,driving_energy=0.)
    ev_result,ess_result = solve_rank_one_curve(ev),solve_rank_one_curve(ess)
    assert ev_result['success'] and ess_result['success']
    np.testing.assert_allclose(ev_result['net_power_kw'],[1.,-.315],atol=2e-7)
    np.testing.assert_allclose(ev_result['net_power_kw'],ess_result['net_power_kw'])
    np.testing.assert_allclose(ev_result['battery']['battery_energy_kwh'],
                               ess_result['battery']['battery_energy_kwh'])
    assert ev_result['distance_squared'] == pytest.approx(ess_result['distance_squared'])
    assert ev_result['battery']['grid_net_kwh'] == pytest.approx(ess_result['battery']['grid_net_kwh'])
    assert ev_result['battery']['driving_energy_kwh'] == .1
    assert ess_result['battery']['driving_energy_kwh'] == 0.
    assert ev_result['battery']['battery_gain_kwh'] == pytest.approx(.45)
    assert ess_result['battery']['battery_gain_kwh'] == pytest.approx(.55)


def test_staff_added_window_preserves_task_and_improves_attained_global_toy_score():
    task = replace(CurveTask(),eta_charge=1.,eta_discharge=1.,terminal_energy=2.5)
    old = solve_rank_one_curve(replace(task,availability=[[True],[False]]),[[1.],[3.]])
    new = solve_rank_one_curve(task,[[1.],[3.]])
    assert old['success'] and new['success']
    assert old['distance_squared'] == pytest.approx(.125)
    assert new['distance_squared'] == pytest.approx(1.125)
    assert old['battery']['grid_net_kwh'] == pytest.approx(new['battery']['grid_net_kwh'])
    np.testing.assert_allclose(new['net_power_kw'],[0.,.5],atol=2e-7)


def test_forced_staff_window_shift_loses_old_plan_and_can_reduce_score():
    task = replace(CurveTask(),eta_charge=1.,eta_discharge=1.,terminal_energy=2.5)
    old = solve_rank_one_curve(replace(task,availability=[[True],[False]]),[[3.],[1.]])
    shifted = solve_rank_one_curve(replace(task,availability=[[False],[True]]),[[3.],[1.]])
    assert old['success'] and shifted['success']
    assert old['distance_squared'] == pytest.approx(1.125)
    assert shifted['distance_squared'] == pytest.approx(.125)


def test_joint_site_milp_can_leave_declared_local_blind_spot_with_driving_cost_recorded():
    task = replace(CurveTask(),n_sites=2,eta_charge=1.,eta_discharge=1.,
        terminal_energy=2.5,travel_slots=[0,1],driving_energy=[0.,.1])
    result = solve_rank_one_curve(task,[[0.,1.],[0.,1.]])
    assert result['success'] and result['site'] == 1
    np.testing.assert_allclose(result['net_power_kw'],[0.,.6],atol=2e-7)
    assert result['distance_squared'] == pytest.approx(.18)
    assert result['battery']['driving_energy_kwh'] == .1
    assert result['battery']['grid_net_kwh'] == pytest.approx(.6)
    assert result['battery']['battery_gain_kwh'] == pytest.approx(.5)
    assert result['includes_joint_site_choice'] and not result['includes_complete_curve_library']


def test_travel_exhausted_high_sensitivity_site_is_not_selected_and_all_infeasible_retained():
    task = replace(CurveTask(),n_sites=2,eta_charge=1.,eta_discharge=1.,
        terminal_energy=2.5,travel_slots=[0,2],driving_energy=[0.,.1])
    result = solve_rank_one_curve(task,[[1.,10.],[1.,10.]])
    assert result['success'] and result['site'] == 0
    assert all(attempt['same_constraint_matrix'] and attempt['same_objective']
               for attempt in result['solver_attempts'])
    if result['solver_attempts'][0]['status'] == 4:
        assert len(result['solver_attempts']) == 2
        assert result['solver_attempts'][1]['presolve'] is False
    assert result['distance_squared'] == pytest.approx(.125)
    failed = solve_rank_one_curve(replace(task,travel_slots=[2,2]))
    assert failed['success'] is False and failed['status'] == 2
    assert failed['distance_squared'] is None and failed['net_power_kw'] is None


def test_v2g_adds_no_direction_when_terminal_task_uses_all_charging_headroom():
    result = solve_rank_one_curve(replace(CurveTask(),p_discharge_max=1.,terminal_energy=3.8))
    assert result['success']
    np.testing.assert_allclose(result['net_power_kw'],[1.,1.],atol=2e-7)
    assert result['distance_squared'] == pytest.approx(0.,abs=2e-14)
    assert result['battery']['grid_export_kwh'] == pytest.approx(0.)


def test_lin_distflow_pure_p_and_fixed_power_factor_leave_rx_rank_deficient():
    pure = local_rx_information([0.,1.],[0.,0.],history_precision=1.)
    fixed = local_rx_information([0.,1.],[0.,.4],history_precision=1.)
    independent = local_rx_information([0.,1.,0.],[0.,0.,1.],history_precision=0.)
    assert pure[0,0] > 0 and pure[1,1] == 0.
    assert np.linalg.matrix_rank(pure,tol=1e-12) == 1
    assert np.linalg.matrix_rank(fixed,tol=1e-12) == 1
    assert np.linalg.eigvalsh(independent).min() > .3
    assert np.linalg.matrix_rank(independent,tol=1e-12) == 2


def test_zero_price_at_constant_can_stall_despite_nonzero_feasible_curvature_gain():
    u = np.array([.25,.25])
    projected = u-u.mean()
    np.testing.assert_array_equal(2*projected,[0.,0.])
    assert profile_signal(u,0.)['distance_squared'] == 0.
    assert profile_signal([.5,0.],0.)['distance_squared'] == .125
    epsilon = 1e-5
    assert profile_signal(u+np.array([epsilon,-epsilon]),0.)['distance_squared'] > 0.


def test_independent_per_slot_offset_can_compensate_any_curve_exactly():
    for signal in ([.25,.25],[.5,0.],[1.,-.405]):
        a = np.asarray(signal)
        b = np.linalg.lstsq(np.eye(2),a,rcond=None)[0]
        np.testing.assert_array_equal(a-b,[0.,0.])
        assert float(np.sum((a-b)**2)) == 0.


def test_soc_prefix_is_enforced_not_only_terminal_energy():
    task = replace(CurveTask(),p_discharge_max=1.,energy_min=2.)
    result = solve_rank_one_curve(task)
    assert result['success']
    np.testing.assert_allclose(result['net_power_kw'],[1.,-.405],atol=2e-7)
    assert min(result['battery']['battery_energy_kwh']) >= 2.-2e-7
    forbidden = battery_accounting([-.405,1.],task)
    assert forbidden['battery_energy_kwh'][-1] == pytest.approx(task.terminal_energy)
    assert min(forbidden['battery_energy_kwh']) < task.energy_min
