"""AC measurement fitting checked against an independent two-bus closed form."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy.optimize")

from scipy.sparse import csr_matrix

from rnj_wzzt.estimation.ac_measurement_fit import ACMeasurementSigma, _joint_jacobian_rank, fit_ac_pqv
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs
from rnj_wzzt.models.network import TerminalizedNetwork


def _network(r=0.05, x=0.035):
    # Z_base = base_kv**2/base_mva = 1 ohm, so the oracle can use ohms directly.
    return TerminalizedNetwork(
        buses=pd.DataFrame({
            "bus_id": [0, 1], "bus_type": ["root", "observed_terminal"],
            "is_observed": [True, True], "has_load": [False, True],
            "pd_kw": [0.0, 120.0], "qd_kvar": [0.0, 30.0],
        }),
        branches=pd.DataFrame({
            "branch_id": ["line_01"], "from_bus": [0], "to_bus": [1],
            "r_ohm": [r], "x_ohm": [x], "is_true_closed": [True],
            "branch_type": ["feeder"],
        }),
        root_bus=0, base_kv=1.0, base_mva=1.0,
        original_to_terminal={1: 1}, terminal_to_original={1: 1}, metadata={},
    )


def _two_bus_oracle(p, q, root, r=0.05, x=0.035):
    """High-voltage root of u² + [2(rP+xQ)-V0²]u + |z|²|S|² = 0."""
    p, q, root = (np.asarray(value, dtype=float) for value in (p, q, root))
    coefficient = root**2 - 2.0 * (r * p + x * q)
    discriminant = coefficient**2 - 4.0 * (r * r + x * x) * (p * p + q * q)
    assert np.all(discriminant > 0.0)
    voltage_squared = 0.5 * (coefficient + np.sqrt(discriminant))
    current_squared = (p * p + q * q) / voltage_squared
    return np.sqrt(voltage_squared), p + r * current_squared, q + x * current_squared


def _observations(*, source_p=None, source_q=None):
    index = pd.Index([10, 20, 30, 40, 50, 60], name="time")
    p = np.array([0.12, 0.2, 0.08, 0.24, -0.05, 0.17])
    q = np.array([0.03, -0.04, 0.09, 0.015, -0.025, 0.11])
    root = np.array([1.0, 1.01, 0.995, 1.005, 1.0, 1.012])
    physical_p = p if source_p is None else p + source_p
    physical_q = q if source_q is None else q + source_q
    voltage, master_p, master_q = _two_bus_oracle(physical_p, physical_q, root)
    return {
        "P_terminal": pd.DataFrame({1: p}, index=index),
        "Q_terminal": pd.DataFrame({1: q}, index=index),
        "V_terminal": pd.DataFrame({1: voltage}, index=index),
        "root_voltage": pd.Series(root, index=index, name="root_voltage"),
        "P0_measured": pd.Series(master_p, index=index, name="P0_measured"),
        "Q0_measured": pd.Series(master_q, index=index, name="Q0_measured"),
    }


def _sigma(**extra):
    return ACMeasurementSigma(p=0.001, q=0.001, v=0.0001, **extra)


def _bounds():
    return {"r_bounds_ohm": (0.001, 0.15), "x_bounds_ohm": (0.001, 0.12)}


class _ObservedOnly(dict):
    forbidden = {"truth", "net", "ac", "P_true", "Q_true", "V_true"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"fitting must not read {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"fitting must not read {key}")
        return super().get(key, default)


def test_true_fixed_network_matches_closed_form_and_ignores_unselected_metadata():
    observed = _ObservedOnly(_observations())
    expected_master_p = observed["P0_measured"].copy()
    expected_master_q = observed["Q0_measured"].copy()
    observed.update({key: object() for key in observed.forbidden})
    # Master readings are not residual channels unless their sigma is supplied.
    observed["P0_measured"] += 10.0
    observed["Q0_measured"] -= 10.0
    net = _network()
    branches_before = net.branches.copy(deep=True)
    result = fit_ac_pqv(net, observed, sigma=_sigma(), fit_impedances=False)
    assert result.success
    assert result.loss < 1e-12
    assert result.initial_loss < 1e-12
    assert set(result.standardized_residuals) == {"P_terminal", "Q_terminal", "V_terminal"}
    for key in ("P_terminal", "Q_terminal", "V_terminal", "root_voltage"):
        np.testing.assert_allclose(result.fitted[key], observed[key], atol=1e-9)
    np.testing.assert_allclose(result.fitted["P0_measured"], expected_master_p, atol=1e-9)
    np.testing.assert_allclose(result.fitted["Q0_measured"], expected_master_q, atol=1e-9)
    pd.testing.assert_frame_equal(net.branches, branches_before)
    pd.testing.assert_frame_equal(result.fitted_net.branches, branches_before)
    assert result.ac["converged"].all()
    assert result.diagnostics["degrees_of_freedom"] == 6
    assert result.diagnostics["jacobian_rank"] == 12


def test_fitting_impedances_recovers_rx_and_beats_frozen_wrong_network():
    observed = _observations()
    initial_net = _network(r=0.09, x=0.012)
    before = initial_net.branches.copy(deep=True)
    frozen = fit_ac_pqv(initial_net, observed, sigma=_sigma(), fit_impedances=False, max_nfev=200)
    fitted = fit_ac_pqv(initial_net, observed, sigma=_sigma(), max_nfev=200, **_bounds())
    assert fitted.success
    assert fitted.loss < 1e-6
    assert fitted.loss < frozen.loss * 1e-5
    assert fitted.loss < fitted.initial_loss
    assert frozen.loss > 0.1
    assert fitted.fitted_net.branches.loc[0, "r_ohm"] == pytest.approx(0.05, abs=1e-5)
    assert fitted.fitted_net.branches.loc[0, "x_ohm"] == pytest.approx(0.035, abs=1e-5)
    np.testing.assert_allclose(fitted.fitted["P_terminal"], observed["P_terminal"], atol=1e-6)
    np.testing.assert_allclose(fitted.fitted["Q_terminal"], observed["Q_terminal"], atol=1e-6)
    pd.testing.assert_frame_equal(initial_net.branches, before)


def test_pq_corrections_and_heteroscedastic_rss_match_direct_recomputation():
    observed = _observations()
    observed["P_terminal"][1] += [0.015, -0.012, 0.008, -0.02, 0.01, -0.009]
    observed["Q_terminal"][1] += [-0.008, 0.011, -0.006, 0.009, -0.007, 0.005]
    observed["V_terminal"][1] += [0.0003, -0.0004, 0.0002, 0.0001, -0.0002, 0.0004]
    index = observed["P_terminal"].index
    scales = {
        "P_terminal": pd.DataFrame({1: [0.015, 0.02, 0.025, 0.018, 0.03, 0.022]}, index=index),
        "Q_terminal": pd.DataFrame({1: [0.01, 0.015, 0.012, 0.02, 0.014, 0.018]}, index=index),
        "V_terminal": pd.DataFrame({1: [0.0004, 0.0006, 0.0005, 0.0008, 0.0004, 0.0007]}, index=index),
    }
    sigma = ACMeasurementSigma(p=scales["P_terminal"], q=scales["Q_terminal"], v=scales["V_terminal"])
    result = fit_ac_pqv(_network(), observed, sigma=sigma, fit_impedances=False, max_nfev=200)
    assert result.success
    assert result.loss < result.initial_loss
    assert np.max(np.abs((result.fitted["P_terminal"] - observed["P_terminal"]).to_numpy())) > 1e-4
    row_rss = np.zeros(len(index))
    for key, scale in scales.items():
        residual = ((result.fitted[key] - observed[key]) / scale).to_numpy()
        np.testing.assert_allclose(np.asarray(result.standardized_residuals[key])**2, residual**2, atol=1e-10)
        row_rss += np.sum(residual**2, axis=1)
    assert result.loss == pytest.approx(row_rss.sum(), rel=1e-9, abs=1e-10)
    np.testing.assert_allclose(result.per_time["loss"], row_rss, atol=1e-10)
    np.testing.assert_array_equal(result.per_time["residual_count"], np.full(len(index), 3))
    np.testing.assert_allclose(result.per_time["mean_squared_residual"], row_rss / 3.0, atol=1e-10)
    np.testing.assert_allclose(result.per_time["quality_score"], 1.0 / (1.0 + row_rss / 3.0), atol=1e-10)
    assert result.per_time["initial_loss"].sum() == pytest.approx(result.initial_loss)
    physical_v, _, _ = _two_bus_oracle(
        result.fitted["P_terminal"][1], result.fitted["Q_terminal"][1], result.fitted["root_voltage"]
    )
    np.testing.assert_allclose(result.fitted["V_terminal"][1], physical_v, atol=1e-9)


def test_fixed_source_and_master_meters_match_physical_addition_with_signed_q():
    extra_p = np.array([0.025, 0.04, 0.018, 0.03, 0.015, 0.02])
    extra_q = np.array([-0.01, 0.015, -0.006, 0.008, -0.005, 0.012])
    observed = _observations(source_p=extra_p, source_q=extra_q)
    index = observed["P_terminal"].index
    source = SourceInputs(
        p_pu=pd.Series(extra_p, index=index).iloc[::-1],
        q_pu=pd.Series(extra_q, index=index).iloc[::-1], provenance="external_test",
    )
    sigma = _sigma(root_v=0.002, master_p=0.003, master_q=0.003)
    result = fit_ac_pqv(_network(), observed, sigma=sigma, fit_impedances=False, source=source, source_bus_id=1)
    assert result.success
    assert result.loss < 1e-12
    assert set(result.standardized_residuals) == {
        "P_terminal", "Q_terminal", "V_terminal", "root_voltage", "P0_measured", "Q0_measured"
    }
    # The latent terminal channels remain the legitimate meter loads.
    np.testing.assert_allclose(result.fitted["P_terminal"], observed["P_terminal"], atol=1e-9)
    np.testing.assert_allclose(result.fitted["Q_terminal"], observed["Q_terminal"], atol=1e-9)
    for key in ("V_terminal", "root_voltage", "P0_measured", "Q0_measured"):
        np.testing.assert_allclose(result.fitted[key], observed[key], atol=1e-9)
    np.testing.assert_array_equal(result.per_time["residual_count"], np.full(len(index), 6))


def test_root_voltage_sigma_allows_root_correction_instead_of_forcing_meter_value():
    observed = _observations()
    true_root = observed["root_voltage"].copy()
    observed["root_voltage"] += 0.008
    sigma = ACMeasurementSigma(p=1e-4, q=1e-4, v=1e-5, root_v=0.02, master_p=1e-4, master_q=1e-4)
    result = fit_ac_pqv(_network(), observed, sigma=sigma, fit_impedances=False, max_nfev=200)
    assert result.success
    assert result.loss < result.initial_loss
    np.testing.assert_allclose(result.fitted["root_voltage"], true_root, atol=1e-5)
    assert np.min(np.abs(result.fitted["root_voltage"] - observed["root_voltage"])) > 0.007
    rss = sum(float(np.sum(np.asarray(values)**2)) for values in result.standardized_residuals.values())
    assert result.loss == pytest.approx(rss, rel=1e-10)


@pytest.mark.parametrize("defect", ["time_labels", "terminal_labels", "root_labels"])
def test_measurement_label_mismatches_are_rejected(defect):
    observed = _observations()
    if defect == "time_labels":
        observed["Q_terminal"].index = pd.Index([10, 20, 30, 40, 50, 99])
    elif defect == "terminal_labels":
        observed["V_terminal"].columns = [99]
    else:
        observed["root_voltage"].index = pd.Index([10, 20, 30, 40, 50, 99])
    with pytest.raises(ValueError):
        fit_ac_pqv(_network(), observed, sigma=_sigma(), fit_impedances=False)


@pytest.mark.parametrize("r_bounds", [None, (-0.01, 0.15), (0.001, np.inf), (0.05, 0.05)],
                         ids=["missing", "negative", "nonfinite", "zero_width"])
def test_impedance_fitting_requires_valid_explicit_bounds(r_bounds):
    with pytest.raises(ValueError):
        fit_ac_pqv(_network(), _observations(), sigma=_sigma(),
                   r_bounds_ohm=r_bounds, x_bounds_ohm=(0.001, 0.12))


@pytest.mark.parametrize("defect", ["unknown_bus", "negative_p", "time_labels"])
def test_fixed_source_rejects_unknown_bus_or_invalid_amplitudes(defect):
    observed = _observations()
    index = observed["P_terminal"].index
    p = pd.Series(0.02, index=index)
    q = pd.Series(-0.005, index=index)
    bus = 999 if defect == "unknown_bus" else 1
    if defect == "negative_p":
        p.iloc[0] = -0.01
    if defect == "time_labels":
        p.index = pd.Index([10, 20, 30, 40, 50, 99])
    with pytest.raises(ValueError):
        fit_ac_pqv(_network(), observed, sigma=_sigma(), fit_impedances=False,
                   source=SourceInputs(p_pu=p, q_pu=q, provenance="external_test"), source_bus_id=bus)


def test_ac_nonconvergence_raises_instead_of_returning_a_good_fit():
    with pytest.raises((RuntimeError, FloatingPointError)):
        fit_ac_pqv(_network(), _observations(), sigma=_sigma(), fit_impedances=False,
                   ac_max_iter=1, ac_tol=1e-14)


def test_optimizer_budget_exhaustion_retains_failure_and_actual_positive_loss():
    result = fit_ac_pqv(_network(r=0.09, x=0.012), _observations(), sigma=_sigma(),
                       max_nfev=1, **_bounds())
    assert not result.success
    assert result.nfev == 1
    assert result.loss > 0.1
    assert result.loss == pytest.approx(result.initial_loss, rel=1e-10)
    rss = sum(float(np.sum(np.asarray(values)**2)) for values in result.standardized_residuals.values())
    assert result.loss == pytest.approx(rss, rel=1e-10)
    assert (result.per_time["quality_score"] < 1.0).any()


@pytest.mark.parametrize(("field", "value"), [("p", 0.0), ("v", np.nan)])
def test_sigma_must_be_finite_and_positive(field, value):
    options = {"p": 0.001, "q": 0.001, "v": 0.0001, field: value}
    with pytest.raises(ValueError):
        fit_ac_pqv(_network(), _observations(), sigma=ACMeasurementSigma(**options), fit_impedances=False)


def test_enabled_master_sigma_requires_its_observation():
    observed = deepcopy(_observations())
    del observed["P0_measured"]
    with pytest.raises(ValueError):
        fit_ac_pqv(_network(), observed, sigma=_sigma(master_p=0.01), fit_impedances=False)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_joint_jacobian_rank_uses_the_true_orthogonal_complement(sparse):
    local = np.array([[1.0, 0.0], [0.0, 1.0], [1e8, 2e8]])
    shared = np.array([[0.0, 0.0], [0.0, 0.0], [3e8, 4e8]])
    jacobian = np.c_[shared, local]
    # The two identity rows and the nonzero third-row shared column are independent.
    # Subtracting a near-equal projected matrix can spuriously invent a fourth rank.
    assert np.linalg.matrix_rank(jacobian) == 3
    if sparse:
        jacobian = csr_matrix(jacobian)
    assert _joint_jacobian_rank(
        jacobian, time_count=1, residual_width=3, state_width=2, shared_width=2
    ) == 3


def test_integer_impedance_input_columns_can_fit_fractional_parameters_without_mutation():
    net = _network(r=1, x=1)
    net.base_kv = 4.0  # Z_base=16 ohm: the oracle's r=.05, x=.035 mean .8 and .56 ohm.
    before = net.branches.copy(deep=True)
    assert pd.api.types.is_integer_dtype(net.branches["r_ohm"])
    assert pd.api.types.is_integer_dtype(net.branches["x_ohm"])
    result = fit_ac_pqv(
        net, _observations(), sigma=_sigma(), max_nfev=200,
        r_bounds_ohm=(0.0, 2.0), x_bounds_ohm=(0.0, 2.0),
    )
    assert result.success
    assert result.loss < 1e-6
    assert result.fitted_net.branches.loc[0, "r_ohm"] == pytest.approx(0.8, abs=1e-4)
    assert result.fitted_net.branches.loc[0, "x_ohm"] == pytest.approx(0.56, abs=1e-4)
    assert pd.api.types.is_float_dtype(result.fitted_net.branches["r_ohm"])
    assert pd.api.types.is_float_dtype(result.fitted_net.branches["x_ohm"])
    pd.testing.assert_frame_equal(net.branches, before)
