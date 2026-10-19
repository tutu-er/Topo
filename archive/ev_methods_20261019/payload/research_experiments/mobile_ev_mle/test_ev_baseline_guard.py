"""Baseline-guard tests; tiny MILPs exercise the existing operation builder."""
from copy import deepcopy
import inspect
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

try:
    from . import ev_baseline_guard as guard
except ImportError:
    import ev_baseline_guard as guard

from research_experiments.mobile_ev_mle.ev_operations_milp import EVConfig, check_plan, LinearMILP
from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair


def config():
    return EVConfig(n_slots=2, n_sites=1, net_grid_energy_kwh=1.5)


def pair(B, *, name="pair"):
    B = np.atleast_2d(np.asarray(B, float))
    return FrozenPair(name, np.zeros(len(B)), B, np.zeros((len(B), 0)),
                      np.zeros((0, 0)), np.zeros(0), np.zeros(0),
                      metadata={"nested": {"label": "unchanged"}})


def search_result(plan, *, success=True, failures=()):
    return dict(success=success, status="fixture", plan=plan, failures=list(failures),
                arbitrary_full_record={"raw": [1, -2, 3]}, proxy_lower=-77,
                proxy_upper=123, solver_records=[{"message": "retained"}])


def test_real_two_slot_milp_finds_gain_without_action_enumeration():
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]],
        search_options=dict(max_iterations=3, solver_seconds=3, total_seconds=12,
                            absolute_gap=1e-6, relative_gap=1e-6))
    assert result["success"] and result["guard_passed"]
    assert result["selected_origin"] == "milp_search"
    assert result["proxy_gain_interval"]["lower"] == pytest.approx(36, abs=1e-6)
    assert check_plan(result["plan"], config())["feasible"]
    assert result["search"]["enumerated_complete_actions"] == 0
    assert result["baseline"]["requested_u_kw"] == [[3.0], [3.0]]
    assert result["baseline"]["actual_u_kw"] == pytest.approx(np.array([[3.], [3.]]))


def test_fixed_baseline_movement_soc_and_raw_returned_controls():
    cfg = EVConfig(n_slots=3, n_sites=2, net_grid_energy_kwh=1.5)
    requested = np.array([[0., 0.], [0., 3.], [0., 3.]])
    result = guard.realize_baseline(cfg, requested)
    assert result["success"]
    audit = result["operation_audit"]
    assert audit["travel_energy_kwh"] == pytest.approx(.2)
    assert audit["soc"][-1] == pytest.approx((24 + .95*1.5 - .2)/60)
    assert audit["route"][0]["origin"] == 0
    assert audit["route"][0]["destination"] == 1
    assert result["requested_u_kw"] is not result["actual_u_kw"]
    assert np.array_equal(requested, [[0, 0], [0, 3], [0, 3]])


def test_soc_below_limit_during_move_is_infeasible_even_if_later_charging_recovers():
    cfg = EVConfig(n_slots=3, n_sites=2, net_grid_energy_kwh=.5,
        battery_capacity_kwh=2, initial_soc=.1, soc_min=.1, travel_energy_kwh=.2)
    result = guard.realize_baseline(cfg, [[0, 0], [0, 1], [0, 1]])
    assert not result["success"]
    assert result["plan"] is None
    assert result["status"] == "baseline_infeasible_or_solver_failed"


@pytest.mark.parametrize("requested", [None, [[3]], [[np.nan], [3]]])
def test_unknown_or_invalid_baseline_cannot_become_an_executable_fallback(requested):
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], requested)
    assert not result["success"] and result["plan"] is None
    assert result["search"] is None
    assert result["status"] == "baseline_unavailable"
    assert result["failures"]


def test_baseline_energy_infeasibility_retained_and_search_not_run(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("search must not run without an executable baseline")
    monkeypatch.setattr(guard, "adaptive_search", forbidden)
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[6], [6]])
    assert result["status"] == "baseline_unavailable"
    assert result["baseline"]["solver"]["status"] == 2
    assert result["plan"] is None


def test_baseline_solver_without_incumbent_is_explicit_failure(monkeypatch):
    monkeypatch.setattr(LinearMILP, "solve", lambda *a, **kw:
        OptimizeResult(x=None, status=1, success=False, message="test budget exhausted"))
    result = guard.realize_baseline(config(), [[3], [3]])
    assert not result["success"]
    assert result["failures"] == ["test budget exhausted"]
    assert not result["solver"]["incumbent_available"]


def test_negative_growth_returns_baseline_and_keeps_entire_search_output(monkeypatch):
    plan = guard.realize_baseline(config(), [[6], [0]])["plan"]
    raw_search = search_result(plan)
    monkeypatch.setattr(guard, "adaptive_search", lambda *a, **kw: raw_search)
    result = guard.guarded_adaptive_search(config(), [pair([0, 1])], [[3], [3]])
    assert result["fallback"] and not result["guard_passed"]
    assert result["selected_origin"] == "baseline"
    assert result["proxy_gain_interval"]["upper"] < 0
    assert result["search"] is raw_search
    assert result["search"]["proxy_lower"] == -77


def test_open_gain_interval_keeps_baseline(monkeypatch):
    plan = guard.realize_baseline(config(), [[6], [0]])["plan"]
    monkeypatch.setattr(guard, "adaptive_search", lambda *a, **kw: search_result(plan))
    def approximate_qp(p, u):
        lo, hi = (8., 12.) if np.asarray(u).ravel()[0] < 4 else (11., 15.)
        return dict(success=True, q_lower=lo, q_upper=hi, status=1, message="open numerical gap")
    monkeypatch.setattr(guard, "solve_pair", approximate_qp)
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]])
    interval = result["proxy_gain_interval"]
    assert interval["raw_lower"] == -1 and interval["raw_upper"] == 7
    assert interval["lower"] < 0 < interval["upper"]
    assert result["status"] == "baseline_proxy_gain_unproven"


def test_failed_search_with_good_plan_still_keeps_baseline(monkeypatch):
    plan = guard.realize_baseline(config(), [[6], [0]])["plan"]
    raw = search_result(plan, success=False, failures=["inner competitor failure"])
    monkeypatch.setattr(guard, "adaptive_search", lambda *a, **kw: raw)
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]])
    assert result["proxy_gain_interval"]["lower"] > 35
    assert result["selected_origin"] == "baseline"
    assert result["status"] == "baseline_search_failed"
    assert "search: inner competitor failure" in result["failures"]
    assert result["search"] is raw


def test_search_exception_and_missing_plan_keep_audited_baseline(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("solver exception")
    monkeypatch.setattr(guard, "adaptive_search", failed)
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]])
    assert result["success"] and result["fallback"]
    assert result["status"] == "baseline_search_exception"
    assert "search: solver exception" in result["failures"]
    monkeypatch.setattr(guard, "adaptive_search", lambda *a, **kw: search_result(None))
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]])
    assert result["status"] == "baseline_search_plan_unavailable"


def test_baseline_point_failure_is_preserved_with_physical_fallback(monkeypatch):
    plan = guard.realize_baseline(config(), [[6], [0]])["plan"]
    monkeypatch.setattr(guard, "adaptive_search", lambda *a, **kw: search_result(plan))
    monkeypatch.setattr(guard, "solve_pair", lambda *a, **kw:
        dict(success=False, q_lower=0., q_upper=1., status=-1, message="QP failure"))
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]])
    assert result["status"] == "baseline_point_evaluation_failed"
    assert result["selected_origin"] == "baseline"
    assert result["proxy_gain_interval"] is None
    assert len(result["failures"]) == 2


def test_invalid_search_plan_is_never_selected(monkeypatch):
    plan = guard.realize_baseline(config(), [[6], [0]])["plan"]
    plan.u_kw[0, 0] = 7
    monkeypatch.setattr(guard, "adaptive_search", lambda *a, **kw: search_result(plan))
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]])
    assert result["status"] == "baseline_search_plan_audit_failed"
    assert result["selected_origin"] == "baseline"
    assert check_plan(result["plan"], config())["feasible"]


def test_real_search_does_not_mutate_original_pair_arrays_metadata_or_cache():
    # This metadata triggers the real history QR cache, so checking an empty
    # caller cache verifies isolation rather than merely using a cache-free QP.
    original = FrozenPair("qr_pair", np.zeros(2), np.array([[0., 0.], [1., -1.]]),
        np.array([[1.], [0.]]), np.zeros((0, 1)), np.array([-1.]), np.array([1.]),
        metadata={"variable_layout": [{"history_state": [0, 1]}],
                  "free_indices": [0], "future_row_start": 1,
                  "nested": {"label": "unchanged"}})
    fingerprint, metadata = original.fingerprint(), deepcopy(original.metadata)
    result = guard.guarded_adaptive_search(config(), [original], [[3], [3]],
        search_options=dict(max_iterations=1, solver_seconds=2, total_seconds=6))
    assert result["pair_hashes"] == {original.name: fingerprint}
    assert original.fingerprint() == fingerprint
    assert original.metadata == metadata
    assert original._solver_cache == {}
    assert result["baseline_score"]["rows"][0]["solution"]["acceleration"] == "history_QR"
    assert all(not getattr(original, key).flags.writeable
               for key in ("c", "B", "A", "F", "lower", "upper", "scale"))


@pytest.mark.parametrize("bad", [[], [pair([1])], [pair([1, -1]), pair([1, -1])]])
def test_bad_pair_banks_fail_before_baseline_or_search(bad):
    result = guard.guarded_adaptive_search(config(), bad, [[3], [3]])
    assert result["status"] == "guard_input_failure"
    assert result["baseline"] is None and result["search"] is None


def test_truth_noise_and_unknown_options_do_not_enter_the_search_api():
    parameters = inspect.signature(guard.guarded_adaptive_search).parameters
    assert "truth" not in parameters and "future_noise" not in parameters
    result = guard.guarded_adaptive_search(config(), [pair([1, -1])], [[3], [3]],
        search_options={"future_noise": [1, 2]})
    assert result["status"] == "guard_input_failure"
    assert "unknown search options" in result["failures"][0]
