"""Rolling data isolation, physical carry and full-trace failure retention."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from . import adaptive_rolling_study as rolling
    from .likelihood import Observations
except ImportError:
    import adaptive_rolling_study as rolling
    from research_experiments.mobile_ev_mle.likelihood import Observations


def _manual_plan(config, arcs, u):
    """Independent energy recurrence; no MILP rows or rolling interpreter."""
    u = np.asarray(u, float)
    pp, pm = np.maximum(u, 0.), np.maximum(-u, 0.)
    movement = np.zeros(config.n_slots)
    for arc in arcs:
        if arc.is_travel:
            movement[arc.departure] += arc.travel_energy_kwh
    energy = [config.initial_soc*config.battery_capacity_kwh]
    for t in range(config.n_slots):
        energy.append(energy[-1]+config.dt_hours*(config.eta_charge*pp[t].sum()
            -pm[t].sum()/config.eta_discharge)-movement[t])
    plan = rolling.EVPlan(tuple(arcs), np.ones(len(arcs)), pp, pm, u, np.asarray(energy))
    assert rolling.check_plan(plan, config)["feasible"]
    return plan


def _staying_plan(config):
    arcs = [rolling.Arc(t, t+1, config.initial_site, config.initial_site) for t in range(config.n_slots)]
    u = np.zeros((config.n_slots, config.n_sites))
    u[:, config.initial_site] = config.net_grid_energy_kwh/(config.n_slots*config.dt_hours)
    return _manual_plan(config, arcs, u)


def _movement_fixture(travel=1):
    access = np.ones((8, 3), bool); access[:2, 2] = False
    config = rolling.EVConfig(access=access, travel_slots=travel)
    arcs = [rolling.Arc(0, 1, 0, 0), rolling.Arc(1, 1+travel, 0, 1, .2)]
    arcs += [rolling.Arc(t, t+1, 1, 1) for t in range(1+travel, 8)]
    u = np.zeros((8, 3)); u[0, 0] = 4.
    u[1+travel:, 1] = 5./((7-travel)*config.dt_hours)
    return config, _manual_plan(config, arcs, u)


def _history(count=12):
    return Observations(p=np.full((count, 3), .12), q=np.full((count, 3), .04),
        v=np.full((count, 3), .99), master_p=np.full(count, .46), master_q=np.full(count, .16))


def test_prefix_carries_travel_soc_location_and_remaining_grid_task_once():
    config, plan = _movement_fixture()
    prefix = rolling.execute_prefix(plan, config, 2)
    assert prefix.sites == (0, -1)
    assert prefix.next_site == 1 and prefix.in_transit_destination is None
    assert prefix.grid_energy_kwh == pytest.approx(1.)
    assert prefix.driving_energy_kwh == pytest.approx(.2)
    assert prefix.energy_kwh[-1] == pytest.approx(24.+.95-.2)
    remaining = rolling.remaining_config(config, prefix)
    assert remaining.n_slots == 6 and remaining.initial_site == 1
    assert remaining.net_grid_energy_kwh == pytest.approx(5.)
    assert remaining.initial_soc*60. == pytest.approx(24.75)
    np.testing.assert_array_equal(remaining.access, config.access[2:])
    realized, check = rolling.combine_executed_plans(plan, _staying_plan(remaining), config, prefix)
    assert check["feasible"] and check["travel_energy_kwh"] == pytest.approx(.2)
    assert check["grid_energy_kwh"] == pytest.approx(6.)
    assert realized.energy_kwh[-1] == pytest.approx(24.+.95*6.-.2)


def test_transit_is_recorded_and_never_relabelled_as_arrived():
    config, plan = _movement_fixture(travel=2)
    prefix = rolling.execute_prefix(plan, config, 2)
    assert prefix.next_site is None and prefix.in_transit_destination == 1
    assert prefix.arrival_remaining_slots == 1
    assert prefix.driving_energy_kwh == pytest.approx(.2)
    with pytest.raises(ValueError, match="pending transit"):
        rolling.remaining_config(config, prefix)
    with pytest.raises(ValueError, match="one-slot"):
        rolling.plan_round(_history(), rolling.model(3), config)


def test_remaining_throughput_and_absolute_terminal_requirements_survive():
    config, plan = _movement_fixture()
    config = replace(config, max_throughput_kwh=6., terminal_site=1, terminal_energy_kwh=29.5)
    assert rolling.check_plan(plan, config)["feasible"]
    prefix = rolling.execute_prefix(plan, config, 2)
    remaining = rolling.remaining_config(config, prefix)
    assert remaining.max_throughput_kwh == pytest.approx(5.)
    assert remaining.terminal_site == 1 and remaining.terminal_energy_kwh == 29.5
    _, check = rolling.combine_executed_plans(plan, _staying_plan(remaining), config, prefix)
    assert check["feasible"]


def test_zero_remaining_grid_energy_has_a_valid_operational_seed():
    config = rolling.EVConfig(net_grid_energy_kwh=2.)
    u = np.zeros((8, 3)); u[:2, 0] = 4.
    plan = _manual_plan(config, [rolling.Arc(t, t+1, 0, 0) for t in range(8)], u)
    remaining = rolling.remaining_config(config, rolling.execute_prefix(plan, config, 2))
    assert remaining.net_grid_energy_kwh == 0.
    seed, record = rolling._feasible_seed(remaining, .2)
    assert record["operational"]["feasible"]
    assert np.max(abs(seed.u_kw)) < 1e-9


def test_prefix_interpreter_uses_signed_grid_energy_and_charge_discharge_efficiencies():
    config = rolling.EVConfig(n_slots=4, allow_discharge=True, net_grid_energy_kwh=-.5)
    u = np.zeros((4, 3)); u[:, 0] = [4., -6., 2., -2.]
    plan = _manual_plan(config, [rolling.Arc(t, t+1, 0, 0) for t in range(4)], u)
    prefix = rolling.execute_prefix(plan, config, 2)
    assert prefix.grid_energy_kwh == pytest.approx(-.5)
    assert prefix.throughput_kwh == pytest.approx(2.5)
    assert prefix.energy_kwh[-1] == pytest.approx(24.+.95-1.5/.95)


def test_history_appends_only_executed_total_meters_and_known_actual_ev_once():
    config, plan = _movement_fixture()
    prefix = rolling.execute_prefix(plan, config, 2)
    history = _history()
    measured = Observations(p=.121+prefix.u_kw/20., q=np.full((2, 3), .04), v=np.full((2, 3), .98),
        master_p=np.full(2, .51), master_q=np.full(2, .16))
    augmented, known = rolling.append_executed_history(history, measured, prefix)
    assert augmented.t == 14 and known.shape == (14, 3)
    np.testing.assert_array_equal(augmented.p[:12], history.p)
    np.testing.assert_array_equal(augmented.p[12:], measured.p)
    np.testing.assert_array_equal(known[:12], np.zeros((12, 3)))
    np.testing.assert_array_equal(known[12:], prefix.u_kw)
    np.testing.assert_allclose(augmented.p[12:]-known[12:]/20., .121)
    with pytest.raises(ValueError, match="must match"):
        rolling.append_executed_history(history, _history(3), prefix)


def test_controller_accepts_only_history_public_config_and_keeps_all_four(monkeypatch):
    captured = []
    def fake_context(history, candidates, meter, **kwargs):
        captured.append((history, candidates, kwargs))
        return dict(status="history_fit_failure", summary={"test": "recorded failure"},
            centres={c.name:[] for c in candidates})
    def forbidden(*args, **kwargs):
        raise AssertionError("controller called truth generator/evaluator")
    monkeypatch.setattr(rolling, "build_history_context", fake_context)
    monkeypatch.setattr(rolling, "generate", forbidden)
    monkeypatch.setattr(rolling, "observe", forbidden)
    monkeypatch.setattr(rolling, "_evaluation_path", forbidden)
    history = _history(); config = rolling.EVConfig()
    decision = rolling.plan_round(history, rolling.model(3), config,
        policy={"run_wzzt": False, "solver_seconds": .2}, known_history_ev_kw=np.zeros((12, 3)))
    assert decision.record["status"] == "selection_failure_operational_fallback"
    assert decision.record["operational"]["feasible"]
    assert tuple(rolling.normalize_family(c.supports, 3) for c in decision.candidates) == rolling.FAMILIES
    assert not decision.record["gates_used_for_pruning"]
    assert captured[0][0] is history and captured[0][2]["horizon"] == 8
    assert captured[0][2]["known_history_ev_kw"].shape == (12, 3)
    params = set(inspect.signature(rolling.plan_round).parameters)
    assert not params.intersection({"path", "truth", "future", "noise", "true_topology"})


def _mock_evaluation(monkeypatch, failed_first_round=False):
    captured = []
    def controller(history, meter, config, **kwargs):
        captured.append(dict(history=history, config=config, known=kwargs["known_history_ev_kw"].copy(),
            warm=kwargs["historical_fits"]))
        candidates = tuple(rolling.RadialCandidate.from_supports("star" if not f else
            "clade_"+"".join(map(str, f[0])), 3, f) for f in rolling.FAMILIES)
        return rolling.RollingDecision(_staying_plan(config), config, candidates,
            dict(centres={c.name:[] for c in candidates}), dict(status="complete"))
    def fitter(candidates, observations, meter, **kwargs):
        fits, failures = {}, {}
        for index, c in enumerate(candidates):
            success = not (failed_first_round and observations.t == 14 and index == 1)
            fits[c.name] = SimpleNamespace(success=success, loss=float(observations.t+index), loglik=-1.,
                r=np.full(c.m, .01), x=np.full(c.m, .005), p=observations.p.copy(), q=observations.q.copy(),
                source_p=np.full(observations.t, .1), source_q=np.full(observations.t, .03),
                diagnostics={"mock_fit_for_data_flow_test": True})
            if not success: failures[c.name] = "deliberate first-round optimizer failure"
        return fits, failures
    monkeypatch.setattr(rolling, "plan_round", controller)
    monkeypatch.setattr(rolling, "fit_bank", fitter)
    return captured


def test_evaluator_observes_two_then_six_and_passes_only_augmented_history(monkeypatch, tmp_path):
    captured = _mock_evaluation(monkeypatch)
    result = rolling.rolling_trial(8140, output=tmp_path)
    assert [row["observed_history_points"] for row in result["rounds"]] == [14, 20]
    assert [row["executed_slots"] for row in result["rounds"]] == [2, 6]
    assert [row["history"].t for row in captured] == [12, 14]
    assert [row["config"].n_slots for row in captured] == [8, 6]
    assert captured[1]["config"].net_grid_energy_kwh == pytest.approx(4.5)
    assert captured[1]["config"].initial_soc == pytest.approx((24.+.95*1.5)/60.)
    assert len(captured[1]["warm"]) == 4
    np.testing.assert_array_equal(captured[1]["known"][:12], np.zeros((12, 3)))
    np.testing.assert_array_equal(captured[1]["known"][12:], np.tile([3., 0., 0.], (2, 1)))
    assert result["score_record"]["complete"] and result["score_record"]["fit_count"] == 8
    truth_index = result["true_index"]
    assert result["score_record"]["relative_gap"] == pytest.approx(float(truth_index))
    assert result["score_record"]["absolute_loss"] == pytest.approx(20.+truth_index)
    assert result["requires_independent_rolling_calibration"] and not result["may_reuse_one_round_gates"]
    archive = np.load(tmp_path/"rolling_8140_observations.npz")
    assert archive["round_1_joint_p"].shape == (14, 3)
    assert archive["round_2_joint_p"].shape == (20, 3)
    assert archive["round_2__clade_01__fit_source_p"].shape == (20,)
    assert result["operational"]["feasible"]


def test_first_round_fit_failure_remains_infinite_even_if_second_round_recovers(monkeypatch):
    captured = _mock_evaluation(monkeypatch, failed_first_round=True)
    result = rolling.rolling_trial(8141)
    assert not result["rounds"][0]["complete"] and result["rounds"][1]["complete"]
    assert not result["score_record"]["complete"]
    assert result["score_record"]["compute_failure"]
    assert np.isinf(result["score_record"]["relative_gap"])
    assert result["score_record"]["failed_fit_count"] == 1
    assert len(captured[1]["warm"]) == 3  # failed initializer is not relabelled successful


def test_rolling_policy_id_changes_with_execution_law_and_task():
    p = rolling.rolling_policy_spec()
    first = rolling.policy_identifiers(p, rolling.EVConfig())
    changed = rolling.policy_identifiers(p, rolling.EVConfig(net_grid_energy_kwh=5.))
    assert first["policy_id"] != changed["policy_id"]
    stressed = rolling.policy_identifiers(p, rolling.EVConfig(), stress="source_drift")
    assert first["policy_id"] == stressed["policy_id"]  # stress is not a controller input
    assert first["population_id"] != stressed["population_id"]
    assert first["may_reuse_one_round_gates"] is False
    with pytest.raises(ValueError, match="tracking=1"):
        rolling.rolling_policy_spec({"tracking_gain": .9})
    with pytest.raises(ValueError, match="no gate-based pruning"):
        rolling.rolling_policy_spec({"gates_influence_shadow_policy": True})
