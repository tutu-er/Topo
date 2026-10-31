"""Physical adapter checks against existing resource declarations, no solvers."""
from itertools import product

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_conic_pricing.resources import (
    increments_from_amplitudes, resource_contract_data, translate_net_load,
)
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem, evaluate_plan
from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
    Battery, DT, ResourceDay, SLOTS, ev_reallocation_module, ev_resource,
    shift_module, spatial_module, shift_shape,
)


def make_resource(resource_id="r1", node=201, ratio=.35):
    return ResourceDay(resource_id, node, .5, 1., -.4, .4,
                       reactive_ratio=ratio, declaration_source="test synthetic declaration")


def build(resources, modules, **extra):
    j = len(modules)
    options = dict(node_labels=[201, 202], elasticities=np.full(j, .5),
                   price_upper=np.ones(j), response_factors=np.tile([.2, .1], (j, 1)))
    options.update(extra)
    return resource_contract_data(resources, modules, **options)


def test_profile_order_units_and_no_double_baseline():
    r1, r2 = make_resource(), make_resource("r2", 202, .2)
    modules = [shift_module("early", "r1", [0], [1], .4),
               shift_module("late", "r2", [2], [3], .3)]
    data = build([r1, r2], modules)
    H = np.asarray(data["profiles"])
    assert H.shape == (384, 2)
    assert H[0, 0] == -1 and H[2, 0] == 1
    assert H[5, 1] == -1 and H[7, 1] == 1
    assert H[192, 0] == -.35 and H[197, 1] == -.2
    assert data["reduction_hours"] == [.25, .25]
    assert data["actual_amplitude_upper"] == [.4, .3]
    assert data["exclusive_groups"] == []
    payload = increments_from_amplitudes(data, [.2, .1])
    assert payload["increment_only_payload"]
    p = payload["node_increments"]
    assert p["201"]["delta_kw"][0] == -.2
    assert p["202"]["delta_kw"][2] == -.1
    assert sum(p["201"]["delta_kw"]) == 0
    assert max(p["201"]["delta_kw"]) == .2  # no .5-kW baseline was added
    unpacked = translate_net_load(H @ np.array([.2, .1]), [201, 202])
    assert np.asarray(unpacked["p_kw"]).shape == (96, 2)


def test_device_rows_exactly_match_box_vertex_violations():
    resource = make_resource()
    data = build([resource], [shift_module("a", "r1", [0], [1], .4)])
    A, b, D = (np.asarray(data[k]) for k in ("device_matrix", "device_rhs", "response_factors"))
    for x in (np.array([.2]), np.array([.5])):
        analytical = A @ x + np.abs((A * x[None, :]) @ D).sum(axis=1) - b
        vertices = np.array([A @ (x * (1 + D @ np.array(z))) - b
                             for z in product([-1., 1.], repeat=2)])
        assert np.allclose(analytical, vertices.max(axis=0), atol=1e-14)
    assert analytical.max() > 0  # x=.5 can exceed the .5-kW headroom


def test_same_resource_types_are_exclusive_not_separate_billing():
    modules = [shift_module("a", "r1", [0], [1], .4),
               shift_module("b", "r1", [2], [3], .4)]
    data = build([make_resource()], modules)
    assert data["exclusive_groups"] == [[0, 1]]
    with pytest.raises(ValueError, match="exclusive"):
        build([make_resource()], modules, separate_types=False)


def test_multi_resource_shared_amplitude_is_rejected():
    module = spatial_module("pair", shift_shape([0], [1]), {"r1": 1, "r2": -1}, .2)
    with pytest.raises(ValueError, match="one separately metered"):
        build([make_resource(), make_resource("r2", 202)], [module])


def test_ev_soc_deadline_and_availability_reuse_existing_declaration():
    baseline = np.zeros(SLOTS)
    baseline[20:28] = .5
    available = np.zeros(SLOTS, dtype=bool)
    available[20:28] = True
    resource = ev_resource("ev", 201, baseline, 1., available,
                           Battery(5., 1., 2.), service_deadline_slot=28)
    module = ev_reallocation_module("shift", resource, [20, 21], [26, 27], .3)
    data = build([resource], [module])
    A, b = np.asarray(data["device_matrix"]), np.asarray(data["device_rhs"])
    x = np.array([.2])
    assert np.max(A @ x - b) <= 1e-12
    physical = increments_from_amplitudes(data, x)["node_increments"]["201"]
    delta = np.asarray(physical["delta_kw"])
    energy = 1 + DT * np.cumsum(baseline + delta)
    assert energy[27] == pytest.approx(2.)
    assert energy[-1] == pytest.approx(2.)
    assert np.all(delta[~available] == 0)


def test_ev_profile_must_preserve_task_before_deadline():
    baseline = np.zeros(SLOTS)
    baseline[20:28] = .5
    available = np.zeros(SLOTS, dtype=bool)
    available[20:36] = True
    resource = ev_resource("ev", 201, baseline, 1., available,
                           Battery(5., 1., 2.), service_deadline_slot=28)
    bad = shift_module("late_rebound", "ev", [26], [30], .2)
    with pytest.raises(ValueError, match="deadline"):
        build([resource], [bad])


def test_unavailable_profile_and_invalid_factor_rejected():
    available = np.ones(SLOTS, dtype=bool)
    available[0] = False
    resource = ResourceDay("r1", 201, .5, 1., -.4, .4, available=available)
    bad = shift_module("a", "r1", [0], [1], .4)
    with pytest.raises(ValueError, match="availability"):
        build([resource], [bad])
    good = shift_module("good", "r1", [2], [3], .4)
    with pytest.raises(ValueError, match="row L1 norm"):
        build([resource], [good], response_factors=[[.7, .4]])


def test_resource_projection_composes_with_price_problem_without_baseline_identity():
    data = build([make_resource()], [shift_module("a", "r1", [0], [1], .4)])
    rows = len(data["profiles"])
    direction = np.zeros((1, rows))
    direction[0, 0], direction[0, 2] = -1., 1.
    # The .5-kW resource baseline is one component of network demand. The net
    # background may differ and can include PV; it must not equal that baseline.
    problem = PriceProblem.from_dict(dict(
        **data, mu0=np.full(rows, .1).tolist(),
        background_factors=np.zeros((rows, 2)).tolist(),
        benefit_directions=direction.tolist(), benefit_offsets=[0.],
        risk_allocations=[.1], budget=1.6,
    ))
    x = np.array([.2])
    audit = evaluate_plan(problem, x)
    assert audit["numerical_feasible"]
    assert audit["benefit_means"][0] == pytest.approx(.4)
    assert audit["expected_cost"] == pytest.approx(.25 / .5 * .2**2)
    assert audit["hard_budget_bound"] == pytest.approx(.25 / .5 * 1.3 * .2**2)
    assert np.asarray(audit["net_mean"])[0] == pytest.approx(-.1)
