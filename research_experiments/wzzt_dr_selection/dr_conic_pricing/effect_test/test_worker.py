"""Pure continuous resource/payment gates, not actual topology effects."""
import json
from pathlib import Path

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.physical_worker import audit_plan, load_request
from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import shift_shape


def packet(tmp_path, *, zero=False):
    shapes = shift_shape(range(24, 32), range(32, 40)).tolist()
    resources = [dict(resource_id=k, node_label=node, baseline_kw=.2, power_upper_kw=.4,
                      prefix_delta_lower_kwh=-.4, prefix_delta_upper_kwh=.4,
                      reactive_ratio=.35) for k, node in (("a", 201), ("b", 204))]
    modules = [dict(module_id=k, profiles={k: shapes}, amplitude_upper_kw=.2) for k in ("a", "b")]
    mean = 0. if zero else .1
    price = mean/.1
    value = dict(resources_by_event=[resources]*4, modules_by_event=[modules, [], modules, []],
        expected_amplitudes_by_event=[[mean, mean], [], [mean, mean], []],
        prices_by_event=[[price, price], [], [price, price], []],
        response_factors_by_event=[[[.15, .05], [.15, -.05]], [], [[.15, .05], [.15, -.05]], []],
        hard_budget_bound=0. if zero else .96)
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    request = dict(plan_path=str(path), repo_root=str(Path(__file__).resolve().parents[4]),
                   factors_by_event=[[1., 1.]]*4, labels=[201, 204], incentive_budget=1.6)
    return value, path, request


def test_continuous_delivery_can_exceed_mean_with_exact_payment(tmp_path):
    _, _, req = packet(tmp_path)
    _, events, fees = audit_plan(req)
    assert events[0]["actual_amplitudes"] == pytest.approx([.12, .11])
    assert events[0]["audit"]["resource_feasible"]
    assert fees["actual_payment"] == pytest.approx(.92)
    assert fees["worst_payment"] == pytest.approx(.96)
    assert events[1]["actual_payment"] == events[3]["actual_payment"] == 0.


def test_zero_price_does_not_create_response(tmp_path):
    _, _, req = packet(tmp_path, zero=True)
    _, events, fees = audit_plan(req)
    assert fees["actual_payment"] == 0.
    assert np.array_equal(events[0]["actual_amplitudes"], [0., 0.])


@pytest.mark.parametrize("name", ["elasticity", "factor", "double_contract"])
def test_invalid_continuous_plan_refused_before_physics(tmp_path, name):
    value, path, req = packet(tmp_path)
    if name == "elasticity":
        value["prices_by_event"][0] = [.5, 1.]
    elif name == "factor":
        value["response_factors_by_event"][0][0] = [.7, .7]
    else:
        value["modules_by_event"][0][1]["profiles"] = value["modules_by_event"][0][0]["profiles"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        audit_plan(req)


def test_bad_request_hash_precedes_any_producer_import(tmp_path):
    path = tmp_path/"request.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash"):
        load_request(path, "wrong")
