"""Analytic checks for absolute-power pair-response square coefficients."""
import json

import numpy as np
import pytest

from .coefficients import QuadraticTask, build_tasks_from_responses, score_tasks


def responses_at_anchor(base_means, jacobians, anchor):
    return {name: dict(mean=np.asarray(mean) + np.einsum(
        "tmj,tj->tm", jacobians[name], anchor), jacobian=jacobians[name])
        for name, mean in base_means.items()}


def test_single_ev_exact_affine_distance_with_nonuniform_meter_precision():
    anchor = np.array([[.3, 0.], [0., -.5], [.4, 0.]])
    base = {"a": np.array([[1., -2.], [.2, 1.], [-.5, .3]]),
            "b": np.array([[.3, .2], [.1, -.2], [.7, .1]])}
    jac = {"a": np.array([[[1., 2.], [3., -1.]], [[.2, 1.], [-.5, 2.]],
                           [[1.5, -.2], [.7, 1.]]]),
           "b": np.array([[[.3, -.5], [1., -.2]], [[.1, -.2], [.5, .4]],
                           [[-.2, .8], [.1, .3]]])}
    sigma = np.array([[2., 1.], [1.5, .8], [.7, 2.]])
    response = responses_at_anchor(base, jac, anchor)
    task = build_tasks_from_responses(response, sigma, include_intercept=True,
                                      anchor_power=anchor)[0]
    power = np.array([[.4, 0.], [0., -.8], [1.1, 0.]])
    g = jac["a"] - jac["b"]
    complete = (response["a"]["mean"] - response["b"]["mean"]
                + np.einsum("tmj,tj->tm", g, power - anchor))
    assert task.value(power) == pytest.approx(.5 * np.sum((complete / sigma) ** 2))
    assert np.min(task.quadratic) >= 0
    assert task.linear.shape == power.shape


def test_gsp_removes_background_and_keeps_squared_response():
    anchor = np.array([[1., 0.], [0., -1.]])
    response = {"a": dict(mean=np.array([[100.], [-40.]]),
                            jacobian=np.array([[[2., -1.]], [[.5, 3.]]])),
                "b": dict(mean=np.zeros((2, 1)), jacobian=np.zeros((2, 1, 2)))}
    sigma = np.array([[2.], [.5]])
    task = build_tasks_from_responses(response, sigma, include_intercept=False,
                                      anchor_power=anchor)[0]
    power = np.array([[-2., 0.], [0., .3]])
    pure_response = np.einsum("tmj,tj->tm", response["a"]["jacobian"], power)
    assert task.constant == 0
    np.testing.assert_array_equal(task.linear, np.zeros((2, 2)))
    assert task.value(power) == pytest.approx(.5 * np.sum((pure_response / sigma) ** 2))


def test_zero_g_gives_no_action_gain_even_with_a_large_background_difference():
    jacobian = np.array([[[1., -2.]], [[.5, 1.]]])
    response = {"a": dict(mean=np.array([[3.], [4.]]), jacobian=jacobian),
                "b": dict(mean=np.zeros((2, 1)), jacobian=jacobian.copy())}
    full = build_tasks_from_responses(response, np.ones((2, 1)), include_intercept=True)[0]
    pure = build_tasks_from_responses(response, np.ones((2, 1)), include_intercept=False)[0]
    np.testing.assert_array_equal(full.linear, np.zeros((2, 2)))
    np.testing.assert_array_equal(full.quadratic, np.zeros((2, 2)))
    assert full.value([[8., 0.], [0., -7.]]) == full.value(np.zeros((2, 2)))
    assert pure.value([[8., 0.], [0., -7.]]) == 0


def test_nonzero_background_linear_term_can_reverse_equal_energy_action_order():
    response = {"a": dict(mean=np.array([[0.], [-4.]]),
                            jacobian=np.array([[[1.]], [[2.]]])),
                "b": dict(mean=np.zeros((2, 1)), jacobian=np.zeros((2, 1, 1)))}
    sigma = np.ones((2, 1))
    full = build_tasks_from_responses(response, sigma, include_intercept=True)[0]
    pure = build_tasks_from_responses(response, sigma, include_intercept=False)[0]
    early, late = np.array([[2.], [0.]]), np.array([[0.], [2.]])
    assert early.sum() == late.sum()
    assert pure.value(late) > pure.value(early)
    assert full.value(early) > full.value(late)
    assert full.constant == pytest.approx(8.)
    np.testing.assert_allclose(full.linear, [[0.], [-8.]])
    assert full.value([[0.], [-2.]]) > full.value([[0.], [2.]])


def test_connection_change_uses_absolute_power_not_a_diagonal_increment_square():
    anchor, power = np.array([[2., 0.]]), np.array([[0., 2.]])
    g = np.array([[[1., 3.]]])
    response = {"a": dict(mean=np.array([[7.]]), jacobian=g),
                "b": dict(mean=np.zeros((1, 1)), jacobian=np.zeros((1, 1, 2)))}
    task = build_tasks_from_responses(response, [[1.]], include_intercept=True,
                                      anchor_power=anchor)[0]
    increment = power - anchor
    assert np.count_nonzero(increment) == 2
    exact_residual = 7. + float(np.einsum("tmj,tj->tm", g, increment)[0, 0])
    assert task.value(power) == pytest.approx(.5 * exact_residual ** 2)
    assert task.value(power) == pytest.approx(60.5)
    naive_increment = .5 * 7 ** 2 + 7 * float(np.sum(g[0, 0] * increment[0])) + .5 * float(
        np.sum(g[0, 0] ** 2 * increment[0] ** 2))
    assert naive_increment != pytest.approx(task.value(power))


def test_all_unordered_pairs_and_worst_pair_are_kept_in_stable_order():
    response = {name: dict(mean=np.zeros((1, 1)), jacobian=np.array([[[slope]]]))
                for name, slope in (("c", 4.), ("a", 0.), ("b", 1.))}
    tasks = build_tasks_from_responses(response, [[1.]], include_intercept=False)
    assert [t.name for t in tasks] == ["a__vs__b", "a__vs__c", "b__vs__c"]
    assert [t.value([[2.]]) for t in tasks] == pytest.approx([2., 32., 18.])
    assert score_tasks(tasks, [[2.]]) == pytest.approx(2.)
    with pytest.raises(ValueError):
        score_tasks([], [[2.]])


def test_anchor_change_does_not_change_the_same_absolute_affine_model():
    base = {"a": np.array([[1.], [-2.]]), "b": np.array([[.5], [.3]])}
    jac = {"a": np.array([[[1., 2.]], [[-1., .5]]]),
           "b": np.array([[[.3, -.4]], [[.1, .8]]])}
    anchors = (np.zeros((2, 2)), np.array([[0., 1.], [-.5, 0.]]))
    tasks = [build_tasks_from_responses(responses_at_anchor(base, jac, anchor), [[2.], [.5]],
                include_intercept=True, anchor_power=anchor)[0] for anchor in anchors]
    assert tasks[0].constant == pytest.approx(tasks[1].constant)
    np.testing.assert_allclose(tasks[0].linear, tasks[1].linear, atol=1e-13)
    np.testing.assert_allclose(tasks[0].quadratic, tasks[1].quadratic, atol=1e-13)


def test_task_copies_arrays_and_record_roundtrips_through_json():
    linear, quadratic = np.array([[1., 0.]]), np.array([[2., 0.]])
    task = QuadraticTask("pair", 3., linear, quadratic)
    linear[0, 0] = 99.
    assert task.linear[0, 0] == 1.
    with pytest.raises(ValueError):
        task.quadratic[0, 0] = 0.
    restored = QuadraticTask(**json.loads(json.dumps(task.record(), allow_nan=False)))
    assert restored.value([[.5, 0.]]) == task.value([[.5, 0.]])


@pytest.mark.parametrize("constant,linear,quadratic", [
    (np.nan, [[0.]], [[1.]]),
    (0., [[1.]], [[0.]]),
    (0., [[0.]], [[-1.]]),
    (0., [[np.nan]], [[1.]]),
    (0., [[0., 1.]], [[1.]]),
    (0., [0.], [1.]),
])
def test_invalid_task_rejected(constant, linear, quadratic):
    with pytest.raises(ValueError):
        QuadraticTask("bad", constant, linear, quadratic)


@pytest.mark.parametrize("change", [
    "one_candidate", "mean_shape", "jacobian_shape", "different_sites",
    "nan_mean", "nan_jacobian", "nan_sigma", "zero_sigma", "sigma_shape",
    "anchor_shape", "nan_anchor", "missing_mean", "bad_name",
])
def test_invalid_response_dimensions_or_nonfinite_values_rejected(change):
    response = {name: dict(mean=np.zeros((2, 1)), jacobian=np.ones((2, 1, 2)))
                for name in ("a", "b")}
    sigma, anchor = np.ones((2, 1)), np.zeros((2, 2))
    if change == "one_candidate": response.pop("b")
    elif change == "mean_shape": response["a"]["mean"] = np.zeros((3, 1))
    elif change == "jacobian_shape": response["a"]["jacobian"] = np.zeros((2, 2))
    elif change == "different_sites": response["a"]["jacobian"] = np.zeros((2, 1, 3))
    elif change == "nan_mean": response["a"]["mean"][0, 0] = np.nan
    elif change == "nan_jacobian": response["a"]["jacobian"][0, 0, 0] = np.nan
    elif change == "nan_sigma": sigma[0, 0] = np.nan
    elif change == "zero_sigma": sigma[0, 0] = 0.
    elif change == "sigma_shape": sigma = np.ones((1, 1))
    elif change == "anchor_shape": anchor = np.zeros((2, 1))
    elif change == "nan_anchor": anchor[0, 0] = np.nan
    elif change == "missing_mean": response["a"].pop("mean")
    elif change == "bad_name": response[1] = response.pop("a")
    with pytest.raises(ValueError):
        build_tasks_from_responses(response, sigma, include_intercept=True,
                                   anchor_power=anchor)


def test_task_value_rejects_wrong_shape_and_nan():
    task = QuadraticTask("pair", 0., [[0., 0.]], [[1., 1.]])
    with pytest.raises(ValueError): task.value([[1.]])
    with pytest.raises(ValueError): task.value([[np.nan, 0.]])
