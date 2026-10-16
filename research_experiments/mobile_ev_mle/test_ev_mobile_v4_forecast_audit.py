"""Analytic factor-attribution and no-refit forecast diagnostic tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v4_forecast_audit as audit
from .ev_mobile_v4_domain import action_key


def toy_predictor(candidate, r, x, p, q, sp, sq, root):
    p, q = np.asarray(p), np.asarray(q)
    sp, sq = np.asarray(sp), np.asarray(sq)
    R, X = float(r[0]), float(x[0])
    v = 1. - R * p - X * q - .3 * R * p ** 2 - .2 * R * sp[:, None] * p - .2 * X * sq[:, None] * p
    return dict(v=v, toy_p=p, toy_q=q, toy_sp=sp, toy_sq=sq, toy_R=R, toy_X=X)


def toy_derivative(candidate, physical, source=True):
    p, q = physical["toy_p"], physical["toy_q"]
    sp, sq = physical["toy_sp"], physical["toy_sq"]
    R, X = physical["toy_R"], physical["toy_X"]
    J = np.zeros((len(p), 1, 6))
    J[:, :, 0] = -p - .3 * p ** 2 - .2 * sp[:, None] * p
    J[:, :, 1] = -q - .2 * sq[:, None] * p
    J[:, :, 2] = -R - .6 * R * p - .2 * R * sp[:, None] - .2 * X * sq[:, None]
    J[:, :, 3] = -X
    J[:, :, 4] = -.2 * R * p
    J[:, :, 5] = -.2 * X * p
    return dict(v=J)


def response_inputs():
    candidate = SimpleNamespace(name="a", n=1, m=1, supports=((0,),))
    trained = dict(r=[.13], x=[.02])
    truth = dict(r=np.array([.10]), x=np.array([.04]))
    forecast = dict(p=np.array([[.18]]), q=np.array([[.06]]),
        source_by_candidate={"a":dict(source_p=np.array([.12]), source_q=np.array([.02]))})
    inputs = dict(p=np.array([[.20]]), q=np.array([[.07]]), source_p=np.array([.10]),
                  source_q=np.array([.03]), root_v=np.ones(1), base_kw=20.)
    return candidate, trained, truth, forecast, inputs


def test_additive_mse_reduction_has_exact_additive_allocation():
    weights = np.array([1., 4., 9.])
    errors = {mask: float(weights @ (1 - np.asarray(list(map(int, mask))))) for mask in audit.MASKS}
    result = audit.shapley_error_reduction(errors)
    np.testing.assert_allclose(list(result["contributions"].values()), weights)
    assert result["total_reduction"] == 14.
    assert result["identity_error"] < 1e-14
    assert result["is_likelihood_or_decision_power_attribution"] is False


def test_cancelled_error_can_give_negative_correction_contribution():
    errors = {mask: (4. if mask[0] == "1" else 1.) if mask[1] == "0" else 0. for mask in audit.MASKS}
    result = audit.shapley_error_reduction(errors)
    assert result["contributions"]["lines"] == pytest.approx(-1.5)
    assert result["contributions"]["load"] == pytest.approx(2.5)
    assert result["contributions"]["source"] == 0.
    assert result["sum_contributions"] == pytest.approx(1.)


def test_complete_ac_factorial_uses_same_increment_error_and_reuses_zero_action():
    args = response_inputs()
    counter = []
    def predict(*values):
        counter.append(1)
        return toy_predictor(*values)
    baseline = {}
    first = audit._response_factorial(*args, np.array([[2.]]), predict, toy_derivative, baseline)
    second = audit._response_factorial(*args, np.array([[3.]]), predict, toy_derivative, baseline)
    assert first["success"] and second["success"]
    assert first["production_AC_calls"] == 16
    assert second["production_AC_calls"] == 8
    assert len(counter) == 24
    assert first["voltage_increment_errors"]["111"]["RMSE"] == 0.
    assert first["MSE_allocation"]["identity_error"] < 1e-16
    expected = toy_predictor(args[0], [.1], [.04], [[.3]], [[.07]], [.1], [.03], [1.])["v"] - \
               toy_predictor(args[0], [.1], [.04], [[.2]], [[.07]], [.1], [.03], [1.])["v"]
    np.testing.assert_allclose(first["true_delta_V_pu"], expected)
    assert first["derivative_prediction_error"]["V_per_P"]["units"] == "pu_voltage_per_kW"
    assert first["derivative_prediction_error"]["V_per_Q"]["RMSE"] == pytest.approx(.001)
    assert first["R_X_increment_cancellation"]["opposite_R_X_error_count"] == 1
    assert 0. <= first["R_X_increment_cancellation"]["cancellation_ratio"] <= 1.
    assert first["candidate_refits"] == 0


def test_unmatched_topology_edges_do_not_receive_false_parameter_error():
    candidate = SimpleNamespace(m=5, supports=((0, 1, 2), (0,), (1,), (2,), (0, 1)))
    truth = SimpleNamespace(m=5, supports=((0, 1, 2), (0,), (1,), (2,), (1, 2)))
    row = audit._line_comparison(candidate, dict(r=np.array([1.,2.,3.,4.,99.]), x=np.array([1.,2.,3.,4.,99.])),
                                 truth, np.arange(1.,6.), np.arange(1.,6.))
    assert row["common_edge_count"] == 4
    assert row["common_R"]["RMSE"] == 0.
    assert row["extra_fitted_supports"] == [[0, 1]]
    assert row["missed_true_supports"] == [[1, 2]]
    assert row["unmatched_edges_have_no_assigned_parameter_error"]


def diagnostic_fixture():
    candidate = SimpleNamespace(name="a", n=1, m=1, supports=((0,),))
    other = SimpleNamespace(name="b", n=1, m=1, supports=((0,),))
    observed = dict(p=[[.2]] * 3, q=[[.07]] * 3, v=[[.98]] * 3,
                    master_p=[.3] * 3, master_q=[.1] * 3, root_v=[1.] * 3)
    def factory(bank, seed, family_index, regime, H=96):
        physical = dict(p=np.full((4, 1), .2), q=np.full((4, 1), .07),
            source_p=np.full(4, .1), source_q=np.full(4, .03), root_v=np.ones(4))
        if regime == "shift":
            physical["p"][-1] = .25
        historical = {key:np.asarray(value) for key,value in observed.items()}
        return dict(**physical, train=SimpleNamespace(**{k:v[:2] for k,v in historical.items()}),
            validation=SimpleNamespace(**{k:v[2:] for k,v in historical.items()}), future_start=3,
            truth=dict(candidate=candidate, r=np.array([.1]), x=np.array([.04])), bank=[candidate,other])
    _, trained, _, forecast, _ = response_inputs()
    forecast["source_by_candidate"]["b"] = dict(source_p=np.array([.15]), source_q=np.array([.04]))
    power = [[2.]]; key = action_key(power)
    record = dict(spec=dict(bank="n3", seed=1, family_index=0, T=1, regimes=["regular","shift"]),
        observed=observed, true_name="a", config=dict(base_kw=20.), forecast=forecast,
        training_result=dict(fits=[dict(name=name, success=True, **trained) for name in ("a","b")]),
        methods={name:dict(power_kw=power, action_key=key) for name in ("fixed", "mobile")})
    return record, factory


def test_posthoc_forecast_errors_keep_units_and_deduplicate_methods_without_refit():
    record,factory = diagnostic_fixture()
    result = audit.diagnose_record(record, case_factory=factory, predictor=toy_predictor, derivative=toy_derivative)
    assert result["success"], result
    assert result["candidate_refits"] == result["action_selection_calls"] == 0
    assert result["production_AC_calls"] == 32
    assert result["regenerated_case_calls"] == 2
    regular = result["regimes"]["regular"]
    shift = result["regimes"]["shift"]
    assert regular["load_forecast"]["p"]["RMSE"] == pytest.approx(.4)
    assert shift["load_forecast"]["p"]["RMSE"] == pytest.approx(1.4)
    assert regular["source_forecast_by_candidate"]["a"]["source_p"]["RMSE"] == pytest.approx(.4)
    assert regular["source_forecast_by_candidate"]["b"]["source_p"]["difference_from_true_candidate_forecast"]["RMSE"] == pytest.approx(.6)
    assert regular["unique_actions"] == 1 and regular["method_slots"] == 2
    assert result["line_parameter_errors"]["a"]["common_edges"][0]["relative_R_error"] == pytest.approx(.3)
    assert result["posthoc_truth_only"] and not result["selector_modified"]


def test_missing_saved_fields_prevent_generation_and_preserve_failure():
    def forbidden(*args, **kwargs):
        raise AssertionError("no regeneration needed for an incomplete raw record")
    result = audit.diagnose_record(dict(spec={}, failure="saved failure"), case_factory=forbidden)
    assert not result["success"]
    assert result["missing"] and result["candidate_refits"] == 0


def test_failed_ac_settings_remain_in_factor_denominator_and_call_count():
    def failed(*args):
        raise ValueError("declared AC failure")
    result = audit._response_factorial(*response_inputs(), np.array([[2.]]), failed, toy_derivative, {})
    assert not result["success"]
    assert len(result["failures"]) == 8
    assert result["completed_masks"] == []
    assert result["production_AC_calls"] == 8


@pytest.mark.parametrize("errors", [{"000":1.}, {mask:-1. for mask in audit.MASKS}, {mask:float("nan") for mask in audit.MASKS}])
def test_invalid_mse_factorial_rejected(errors):
    with pytest.raises(ValueError):
        audit.shapley_error_reduction(errors)
