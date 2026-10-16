"""Physical nesting and inheritance tests with independently solved prices."""
from dataclasses import replace

import numpy as np
import pytest

from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .ev_mobile_v4_domain import (
    ACScoreCache, DOMAIN_FLAGS, PARENTS, action_key, assert_nested,
    build_nested_domains, inherit_plan, lift_operation, score_operation,
    select_with_inheritance,
)


def config(**changes):
    args = dict(n_slots=6, n_sites=2, dt_hours=1., p_charge_max_kw=2.,
        p_discharge_max_kw=2., battery_capacity_kwh=10., initial_soc=.5,
        soc_min=0., soc_max=1., terminal_energy_kwh=7., eta_charge=1.,
        eta_discharge=1., setup_slots=1, travel_slots=1,
        travel_energy_kwh=.25, travel_cost=2., setup_cost=1.,
        driver_budget=3., max_moves=1, max_throughput_kwh=6.)
    args.update(changes)
    return MobileServiceConfig(**args)


def solve(domain, prices):
    result = domain.solve_prices(prices, time_limit=10., mip_rel_gap=1e-9)
    assert result.success, result.message
    assert domain.check(result)["feasible"]
    return result


def oracle(prices):
    return lambda power: dict(success=True, score=float(np.sum(prices * power)),
        nearest=[dict(a="reference", half_gap=float(np.sum(prices * power)))])


def test_four_domains_share_task_and_exact_containment_without_resolve():
    domains = build_nested_domains(config())
    prices = np.array([[-2., 0.], [-1., 0.], [0., 1.],
                       [1., 2.], [2., 4.], [3., 7.]])
    results = {key: solve(domain, prices) for key, domain in domains.items()}
    for child, parents in PARENTS.items():
        for parent in parents:
            x = lift_operation(domains[parent], domains[child], results[parent].x)
            assert domains[child].check(x)["feasible"]
            for name in ("u_kw", "charge_kw", "discharge_kw", "energy_kwh"):
                np.testing.assert_array_equal(domains[parent].decode(results[parent])[name],
                    domains[child].decode(x)[name])
            assert domains[child].decode(x)["selected_arcs"] == \
                   domains[parent].decode(results[parent])["selected_arcs"]
            assert results[child].objective_value >= results[parent].objective_value - 1e-7
    assert not domains["fixed_charge"].config.mobile
    assert not domains["mobile_charge"].config.allow_discharge
    assert domains["mobile_bidir"].config.allow_discharge
    assert domains["mobile_bidir"].config.max_moves == 1


def test_mobile_permission_does_not_force_movement_or_driving_energy():
    domains = build_nested_domains(config(travel_energy_kwh=2.))
    prices = np.zeros((6, 2)); prices[:, 0] = 1.
    fixed = solve(domains["fixed_charge"], prices)
    # Literal lift establishes the zero-move witness regardless of solver ties.
    x = lift_operation(domains["fixed_charge"], domains["mobile_bidir"], fixed.x)
    audit = domains["mobile_bidir"].check(x)
    assert audit["moves"] == 0
    assert audit["travel_energy_kwh"] == 0.
    assert audit["driver_cost"] == 0.
    np.testing.assert_array_equal(audit["energy_kwh"], domains["fixed_charge"].check(fixed)["energy_kwh"])


def test_selected_travel_alone_pays_drive_and_setup_has_no_grid_injection():
    domains = build_nested_domains(config())
    prices = np.zeros((6, 2)); prices[:, 1] = 100.
    result = solve(domains["mobile_charge"], prices)
    audit = domains["mobile_charge"].check(result)
    assert audit["moves"] == 1
    assert audit["travel_energy_kwh"] == pytest.approx(.25)
    assert audit["grid_energy_kwh"] == pytest.approx(2.25)
    decoded = domains["mobile_charge"].decode(result)
    for arc in decoded["selected_arcs"]:
        if arc.kind in ("travel", "setup"):
            np.testing.assert_allclose(decoded["u_kw"][arc.departure:arc.arrival], 0., atol=1e-8)


def test_nesting_rejects_removed_permission_and_changed_service():
    domains = build_nested_domains(config())
    with pytest.raises(ValueError, match="removes permission"):
        assert_nested(domains["fixed_bidir"], domains["mobile_charge"])
    with pytest.raises(ValueError, match="removes permission"):
        assert_nested(domains["mobile_charge"], domains["fixed_bidir"])
    changed = build_mobile_service_domain(replace(domains["mobile_bidir"].config,
                                                  terminal_energy_kwh=7.1))
    with pytest.raises(ValueError, match="terminal_energy_kwh"):
        assert_nested(domains["fixed_charge"], changed)


def test_parent_lift_keeps_original_zero_power_travel_timing():
    domains = build_nested_domains(config())
    prices = np.zeros((6, 2)); prices[-1, 1] = 100.
    result = solve(domains["mobile_charge"], prices)
    before = domains["mobile_charge"].decode(result)["selected_arcs"]
    x = lift_operation(domains["mobile_charge"], domains["mobile_bidir"], result.x)
    after = domains["mobile_bidir"].decode(x)["selected_arcs"]
    assert before == after
    assert any(arc.kind == "travel" for arc in before)
    assert not np.shares_memory(x, result.x)


def test_parent_score_cache_and_path_reused_without_any_child_fit():
    domains = build_nested_domains(config())
    prices = np.arange(12.).reshape(6, 2)
    cache = ACScoreCache("same-HV-forecast-references-fit-seed", oracle(prices))
    result = solve(domains["fixed_charge"], prices)
    parent = score_operation("fixed_charge", domains["fixed_charge"], result.x,
                             cache, origin="initial")
    first = inherit_plan(parent, domains, "fixed_bidir", cache)
    second = inherit_plan(parent, domains, "mobile_charge", cache)
    best, report = select_with_inheritance("mobile_bidir", domains, [first, second], [], cache)
    assert best.score == parent.score
    assert report["selected_origin"] == "parent:fixed_bidir"
    assert cache.evaluations == 1
    assert report["nondecreasing_from_parents"]
    assert report["proposal_scoring_complete"]
    assert not best.x.flags.writeable
    # A resumed parent imports its already computed full AC witness as-is.
    resumed_cache = ACScoreCache(cache.objective_id,
                               lambda u: pytest.fail("must not refit a known parent"))
    resumed = inherit_plan(first, domains, "mobile_bidir", resumed_cache)
    assert resumed.score_record == parent.score_record
    assert resumed_cache.evaluations == 0


def test_two_parents_keep_best_common_score_not_componentwise_reference_floor():
    domains = build_nested_domains(config())
    positive = np.zeros((6, 2)); positive[:, 1] = 10.
    cache = ACScoreCache("shared-objective", oracle(positive))
    parents = []
    for name in ("fixed_bidir", "mobile_charge"):
        result = solve(domains[name], positive)
        parents.append(score_operation(name, domains[name], result.x, cache, origin=name))
    selected, report = select_with_inheritance("mobile_bidir", domains, parents, [], cache)
    assert selected.score == max(p.score for p in parents)
    assert report["parent_best_score"] == selected.score
    assert selected.origin == "parent:mobile_charge"
    assert report["selected_parent"]


def test_worse_ac_score_and_invalid_proposal_preserve_parent_and_failure():
    domains = build_nested_domains(config())
    prices = np.zeros((6, 2)); prices[-1, 0] = 10.
    cache = ACScoreCache("shared-objective", oracle(prices))
    good = solve(domains["fixed_charge"], prices)
    parent = score_operation("fixed_charge", domains["fixed_charge"], good.x, cache, origin="anchor")
    bad = solve(domains["mobile_bidir"], -prices)
    selected, report = select_with_inheritance("mobile_bidir", domains, [parent],
                                              [bad.x, np.zeros(3)], cache)
    assert selected.score == parent.score
    assert report["selected_parent"]
    assert not report["proposal_scoring_complete"]
    assert report["failures"][0]["proposal"] == 1
    assert len(report["candidates"]) == 2


def test_cached_score_failure_is_explicit_and_cannot_remove_parent():
    domains = build_nested_domains(config())
    prices = np.zeros((6, 2)); prices[-1, 0] = 10.
    parent_result = solve(domains["fixed_charge"], prices)
    parent_u = domains["fixed_charge"].decode(parent_result)["u_kw"]
    failed_result = solve(domains["mobile_bidir"], -prices)
    def evaluate(power):
        if action_key(power) == action_key(parent_u):
            return dict(success=True, score=8., nearest=[dict(a="a", half_gap=8.)])
        return dict(success=False, failure="joint AC optimizer stopped", reference="b")
    cache = ACScoreCache("same-fit-settings", evaluate)
    parent = score_operation("fixed_charge", domains["fixed_charge"], parent_result.x,
                             cache, origin="parent")
    best, report = select_with_inheritance("mobile_bidir", domains, [parent],
                                          [failed_result.x, failed_result.x], cache)
    assert best.score == 8.
    assert not report["proposal_scoring_complete"]
    assert len(report["failures"]) == 2
    assert cache.evaluations == 2
    assert any(not record["success"] for record in cache.records().values())


def test_objective_mismatch_and_infeasible_parent_are_not_hidden():
    domains = build_nested_domains(config())
    prices = np.ones((6, 2))
    cache = ACScoreCache("objective-A", oracle(prices))
    result = solve(domains["fixed_charge"], prices)
    parent = score_operation("fixed_charge", domains["fixed_charge"], result.x, cache, origin="base")
    with pytest.raises(ValueError, match="different AC objective"):
        inherit_plan(parent, domains, "mobile_bidir", ACScoreCache("objective-B", oracle(prices)))
    bad = result.x.copy(); bad[domains["fixed_charge"].indices.energy[1]] += .2
    with pytest.raises(ValueError, match="parent operation infeasible"):
        lift_operation(domains["fixed_charge"], domains["mobile_bidir"], bad)


def test_no_plan_reports_failure_and_does_not_invent_no_move_fallback():
    domains = build_nested_domains(config())
    cache = ACScoreCache("objective", oracle(np.ones((6, 2))))
    best, report = select_with_inheritance("fixed_charge", domains, [], [], cache)
    assert best is None
    assert not report["success"]
    # The supplied task can make fixed operation empty; it is not relaxed.
    restricted = build_nested_domains(config(terminal_site=1))
    result = restricted["fixed_charge"].solve_prices(np.ones((6, 2)), time_limit=10.)
    assert result.x is None and result.status == 2
