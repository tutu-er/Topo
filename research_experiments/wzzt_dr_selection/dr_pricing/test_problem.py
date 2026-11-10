"""Small portable-input tests; no archived household or topology data."""
import copy
import numpy as np
import pytest

try:
    from .problem import aggregate_delivery
except ImportError:
    from problem import aggregate_delivery


def test_physical_adapter_adds_only_resource_increments():
    delivery = np.array([[1., -1.], [2., 3.], [-.5, .5]])
    active, reactive = aggregate_delivery(delivery, [0, 0, 1], [.2, 0., -.1], 2)
    np.testing.assert_array_equal(active, [[3., -.5], [2., .5]])
    np.testing.assert_allclose(reactive, [[.2, .05], [-.2, -.05]])
    with pytest.raises(ValueError):
        aggregate_delivery(delivery, [0, .5, 1], [.2, 0., -.1], 2)

try:
    from .problem import Devices, PricingProblem
    from .probability import probability_and_gradient
except ImportError:
    from problem import Devices, PricingProblem
    from probability import probability_and_gradient


def case():
    slots, nodes, resources = 8, 3, 2
    reset = np.zeros((resources, slots), bool)
    reset[:, 0] = True
    terminal = np.zeros_like(reset)
    terminal[:, -1] = True
    config = dict(dt_hours=.5, decision_slots=2, residual_unit="synthetic_residual_units",
        information=dict(parameters_frozen=True, actual_future_used=False, truth_used=False),
        selection=dict(history_count=16, new_count=slots, new_day_weight=1.,
            historical_loss_coarse=1., historical_loss_fine=1., relative_band=.03,
            absolute_tolerance=1e-6, candidate_ids=["coarse", "fine"], support_counts=[1, 2],
            historically_eligible=[True, True], candidate_count=2),
        probability=dict(target=.9, independent_cells=True,
                         shared_candidate_noise=True, scope="conditional_voltage"),
        numerics=dict(cutoff=64., order=32, tolerance=1e-7, chunk=16))
    arrays = dict(baseline_coarse=np.full((slots, nodes), .4),
        baseline_fine=np.full((slots, nodes), .1),
        effect_coarse=np.array([[.1, -.03], [.02, .06], [-.01, .04]]),
        effect_fine=np.zeros((nodes, resources)), sigma=np.full((slots, nodes), .3),
        resource_lower_kw=-np.ones((resources, slots)),
        resource_upper_kw=np.ones((resources, slots)),
        resource_cost=np.ones((resources, slots)), resource_efficiency=np.ones(resources),
        resource_energy_lower_kwh=-np.ones(resources), resource_energy_upper_kwh=np.ones(resources),
        resource_reset=reset, resource_terminal=terminal)
    return config, arrays


def test_arbitrary_shape_score_and_explicit_day_weight():
    config, arrays = case()
    problem = PricingProblem(config, arrays)
    assert problem.shape == (2, 4) and problem.active.size == 8
    requests = np.array([[.2, -.1, .1, -.2], [-.1, .2, -.2, .1]])
    values = problem.requests_to_values(requests)
    result = problem.evaluate(values)
    assert result["nominal_audit"]["passed"] and result["gradient"].shape == values.shape
    observation = problem.score_residuals(np.full((8, 3), .5), np.full((8, 3), .1))
    assert observation["selected"] == "fine" and observation["coarse_excluded"]
    assert observation["identity_error"] < 1e-15
    other = copy.deepcopy(config)
    other["selection"]["new_day_weight"] = 2.
    assert PricingProblem(other, arrays).threshold < problem.threshold


def test_observation_adapter_uses_the_same_configured_selection_rule():
    config, arrays = case()
    problem = PricingProblem(config, arrays)
    power = np.ones((8, 2))
    observed = np.full((8, 3), .5)
    coarse = np.zeros((3, 2))
    fine = np.full((3, 2), .2)
    direct = problem.score_residuals(observed, np.full((8, 3), .1))
    adapted = problem.score_observations(power, observed, coarse, fine)
    assert adapted["selected"] == direct["selected"] == "fine"
    assert adapted["identity_error"] < 1e-15


def test_unsupported_selection_is_rejected_instead_of_silently_scored():
    config, arrays = case()
    for key, value in [("candidate_count", 3), ("new_count", 7),
                       ("support_counts", [2, 1]), ("historically_eligible", [True, False])]:
        invalid = copy.deepcopy(config)
        invalid["selection"][key] = value
        with pytest.raises(ValueError):
            PricingProblem(invalid, arrays)
    del config["selection"]["new_day_weight"]
    with pytest.raises(ValueError, match="new_day_weight"):
        PricingProblem(config, arrays)


def test_interface_requires_units_and_explicit_information_declarations():
    config, arrays = case()
    invalid = copy.deepcopy(config)
    del invalid["residual_unit"]
    with pytest.raises(ValueError, match="residual_unit"):
        PricingProblem(invalid, arrays)
    for name in ("parameters_frozen", "actual_future_used", "truth_used"):
        invalid = copy.deepcopy(config)
        del invalid["information"][name]
        with pytest.raises(ValueError, match=name):
            PricingProblem(invalid, arrays)


def test_reset_cannot_erase_an_unfinished_task_or_active_gap():
    config, arrays = case()
    arrays["resource_reset"][0, 6] = True
    with pytest.raises(ValueError, match="before its previous task deadline"):
        Devices(arrays, config["dt_hours"], config["decision_slots"])
    arrays["resource_terminal"][0, 1] = True
    with pytest.raises(ValueError, match="between a completed task and reset"):
        Devices(arrays, config["dt_hours"], config["decision_slots"])


def test_deadlines_and_gap_have_no_hidden_time_constants():
    config, arrays = case()
    arrays["resource_reset"][0, 6] = True
    arrays["resource_terminal"][0, 1] = True
    arrays["resource_lower_kw"][0, 2:6] = 0.
    arrays["resource_upper_kw"][0, 2:6] = 0.
    device = Devices(arrays, config["dt_hours"], config["decision_slots"])
    assert np.array_equal(device.future_lo[0, 2:6], np.zeros(4))
    delivered = device.nominal(np.full((2, 4), .8))
    assert device.audit(delivered)["passed"]
    assert np.all(delivered[0, 2:6] == 0.)


def test_probability_gradient_matches_a_full_schedule_direction():
    config, arrays = case()
    problem = PricingProblem(config, arrays)
    values = np.array([.12, -.04, .02, -.1, -.05, .02, .03, 0.])
    direction = np.array([.2, -.1, .3, -.2, -.1, .4, -.3, .1])
    result = problem.evaluate(values)
    expected = float(result["gradient"] @ direction)
    step = 1e-6
    actual = (problem.evaluate(values + step * direction)["probability"]
              - problem.evaluate(values - step * direction)["probability"]) / (2 * step)
    assert abs(expected - actual) < 2e-7


def test_strict_analytic_event_bound_and_configurable_numerical_audit():
    result = probability_and_gradient([0., .2], [0., .1], [1., 2.], .2)
    assert result["method"] == "analytic_unreachable_event"
    assert result["probability"] == result["certified_error"] == 0.
    assert result["gradient_valid"] and np.all(result["gradient_coarse"] == 0)
    assert np.all(result["gradient_fine"] == 0)
    fixed = probability_and_gradient([1.], [0.], [0.], .5)
    assert fixed["probability"] == 1. and fixed["moments"]["sd"] == 0.
    assert fixed["analytic_tail_bound"] == 0.
    config, arrays = case()
    config["numerics"]["audit_orders"] = [32, 64, 128]
    assert PricingProblem(config, arrays).audit_orders == (32, 64, 128)
    config["numerics"]["typo"] = 1
    with pytest.raises(ValueError, match="Unknown numerical option"):
        PricingProblem(config, arrays)
