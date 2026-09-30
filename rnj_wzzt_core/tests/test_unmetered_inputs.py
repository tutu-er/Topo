"""Input contracts: hand-calculated balances, label order, and AC integration."""

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.estimation.unmetered_inputs import (
    SourceInputs,
    approx_source_inputs,
    compute_meter_balance,
    prepare_source_inputs,
)


def _observed():
    index = pd.Index([10, 20], name="t")
    return {
        "P_terminal": pd.DataFrame([[0.6, 0.4], [0.2, 0.3]], index=index, columns=[101, 102]),
        "Q_terminal": pd.DataFrame([[0.1, 0.2], [-0.1, 0.2]], index=index, columns=[101, 102]),
        "P0_measured": pd.Series([1.2, 0.45], index=index),
        "Q0_measured": pd.Series([0.2, -0.05], index=index),
    }


def _source():
    # Deliberately reverse time order relative to the observations.
    index = pd.Index([20, 10], name="t")
    return SourceInputs(
        p_pu=pd.Series([0.03, 0.02], index=index),
        q_pu=pd.Series([-0.003, 0.004], index=index),
        provenance="external_test",
    )


def test_signed_balance_matches_hand_calculation_after_label_reordering():
    observed = _observed()
    observed["Q_terminal"] = observed["Q_terminal"].loc[[20, 10], [102, 101]]
    for key in ("P0_measured", "Q0_measured"):
        observed[key] = observed[key].loc[[20, 10]]
    # Unusable truth/position values must not be read by this input layer.
    observed.update(truth=object(), net=object(), P_loss_true=object(), source_bus_id=object())
    before = observed["Q_terminal"].copy()
    result = compute_meter_balance(observed)
    np.testing.assert_allclose(result.p_gap_pu, [0.2, -0.05], atol=1e-15)
    np.testing.assert_allclose(result.q_gap_pu, [-0.1, -0.15], atol=1e-15)
    assert result.p_gap_pu.index.equals(observed["P_terminal"].index)
    pd.testing.assert_frame_equal(observed["Q_terminal"], before)


@pytest.mark.parametrize("problem", [
    "nan_terminal", "inf_master", "duplicate_time", "duplicate_terminal",
    "missing_time", "wrong_terminal", "missing_master",
])
def test_balance_rejects_missing_or_ambiguous_meter_values(problem):
    observed = _observed()
    if problem == "nan_terminal":
        observed["P_terminal"].iloc[0, 0] = np.nan
    elif problem == "inf_master":
        observed["Q0_measured"].iloc[0] = np.inf
    elif problem == "duplicate_time":
        observed["P_terminal"].index = [10, 10]
    elif problem == "duplicate_terminal":
        observed["Q_terminal"].columns = [101, 101]
    elif problem == "missing_time":
        observed["Q_terminal"] = observed["Q_terminal"].iloc[:1]
    elif problem == "wrong_terminal":
        observed["Q_terminal"].columns = [101, 999]
    else:
        observed.pop("P0_measured")
    with pytest.raises(ValueError):
        compute_meter_balance(observed)


def test_source_preparation_aligns_independent_copies_and_keeps_signed_q():
    first = _observed()
    second = {"P_terminal": first["P_terminal"].iloc[:1].copy()}
    source = _source()
    original_p, original_q = source.p_pu.copy(), source.q_pu.copy()
    q_only = SourceInputs(
        pd.Series([0.0], index=second["P_terminal"].index),
        pd.Series([-0.01], index=second["P_terminal"].index),
        "capacitive_test",
    )
    result = prepare_source_inputs([first, second], [source, q_only])
    np.testing.assert_array_equal(result.p[0], [0.02, 0.03])
    np.testing.assert_array_equal(result.q[0], [0.004, -0.003])
    np.testing.assert_array_equal(result.q[1], [-0.01])
    assert result.p[0].shape == (2,) and result.p[1].shape == (1,)
    assert result.provenance == ("external_test", "capacitive_test")
    assert not result.all_zero
    result.p[0][0] = 100.0
    result.q[0][0] = 100.0
    pd.testing.assert_series_equal(source.p_pu, original_p)
    pd.testing.assert_series_equal(source.q_pu, original_q)


@pytest.mark.parametrize("problem", [
    "negative_p", "nan_q", "wrong_time", "duplicate_time", "empty_provenance", "count_mismatch",
])
def test_source_preparation_rejects_invalid_external_amplitudes(problem):
    source = _source()
    p, q, provenance = source.p_pu.copy(), source.q_pu.copy(), source.provenance
    if problem == "negative_p":
        p.iloc[0] = -0.01
    elif problem == "nan_q":
        q.iloc[0] = np.nan
    elif problem == "wrong_time":
        p.index = [20, 999]
    elif problem == "duplicate_time":
        q.index = [10, 10]
    elif problem == "empty_provenance":
        provenance = " "
    inputs = [] if problem == "count_mismatch" else [SourceInputs(p, q, provenance)]
    with pytest.raises(ValueError):
        prepare_source_inputs([_observed()], inputs)


def test_none_and_explicit_zero_source_have_distinct_input_states():
    assert prepare_source_inputs([_observed()], None) is None
    index = _observed()["P_terminal"].index
    zero = SourceInputs(pd.Series(0.0, index=index), pd.Series(0.0, index=index), "zero_test")
    result = prepare_source_inputs([_observed()], [zero])
    assert result is not None and result.all_zero


def test_sampled_scenario_balance_and_known_amplitude_use_the_same_grid():
    from rnj_wzzt.estimation.multiscenario import align_scenarios
    from rnj_wzzt.scenario.unmetered_load import UnmeteredLoadSpec, simulate_unmetered_scenario

    index = pd.RangeIndex(96, name="t")
    p_kw = pd.Series(0.0, index=index)
    p_kw.iloc[44:52] = 4.0
    spec = UnmeteredLoadSpec(2, p_kw, 0.4 * p_kw)
    net, bundle = simulate_unmetered_scenario(
        "paper15", source=spec, t_count=16, root_observation="exact",
    )
    observed = bundle.observed
    observed_index = observed["P_terminal"].index
    # The test explicitly converts its known source specification to pu.
    source = SourceInputs(
        spec.p_kw.loc[observed_index] / (1000.0 * net.base_mva),
        spec.q_kvar.loc[observed_index] / (1000.0 * net.base_mva),
        "known_spec_test",
    )
    result = prepare_source_inputs(align_scenarios([observed]), [source])
    balance = compute_meter_balance(observed)
    assert result.p[0].shape == (16,)
    np.testing.assert_allclose(
        balance.p_gap_pu.to_numpy() - result.p[0],
        bundle.truth["P_loss_true"].loc[observed_index], atol=1e-8,
    )
    np.testing.assert_allclose(
        balance.q_gap_pu.to_numpy() - result.q[0],
        bundle.truth["Q_loss_true"].loc[observed_index], atol=1e-8,
    )
    # Full-grid source data cannot silently be truncated to the observed grid.
    with pytest.raises(ValueError):
        prepare_source_inputs([observed], [SourceInputs(p_kw, 0.4 * p_kw, "wrong_grid_test")])


def _approx_observed():
    index = pd.Index([10, 20], name="t")
    return {
        "P_terminal": pd.DataFrame([[0.6, 0.4], [0.25, 0.75]], index=index, columns=[101, 102]),
        "Q_terminal": pd.DataFrame([[0.1, 0.1], [-0.1, 0.3]], index=index, columns=[101, 102]),
        "P0_measured": pd.Series([1.515, 0.909], index=index),
        "Q0_measured": pd.Series([0.101, 0.303], index=index),
    }


def test_approx_source_matches_fixed_ratio_formula_after_reordering_and_copies_inputs():
    observed = _approx_observed()
    observed["Q_terminal"] = observed["Q_terminal"].loc[[20, 10], [102, 101]]
    for name in ("P0_measured", "Q0_measured"):
        observed[name] = observed[name].loc[[20, 10]]
    originals = {name: value.copy(deep=True) for name, value in observed.items()}
    observed.update(truth=object(), net=object(), P_loss_true=object(), source_bus_id=object())

    result = approx_source_inputs(observed)

    # Sum P=1 and sum Q=0.2: (master - 1.01*sum) / 1.01, including the divisor.
    np.testing.assert_allclose(result.p_pu, [0.5, -0.1], atol=1e-15)
    np.testing.assert_allclose(result.q_pu, [-0.1, 0.1], atol=1e-15)
    assert result.p_pu.index.equals(observed["P_terminal"].index)
    assert result.q_pu.index.equals(observed["P_terminal"].index)
    assert result.provenance == "approx_source_inputs"

    result.p_pu.iloc[0] = 100.0
    result.q_pu.iloc[0] = 200.0
    for name, original in originals.items():
        if isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(observed[name], original)
        else:
            pd.testing.assert_series_equal(observed[name], original)


def test_approx_negative_p_is_preserved_until_explicit_projection_or_preparation():
    observed = _approx_observed()
    signed = approx_source_inputs(observed)
    np.testing.assert_allclose(signed.p_pu, [0.5, -0.1], atol=1e-15)
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_source_inputs([observed], [signed])

    projected = approx_source_inputs(observed, negative_p_policy="project")
    prepared = prepare_source_inputs([observed], [projected])
    np.testing.assert_allclose(projected.p_pu, [0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(projected.q_pu, [-0.1, 0.1], atol=1e-15)
    np.testing.assert_allclose(prepared.p[0], [0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(prepared.q[0], [-0.1, 0.1], atol=1e-15)
    assert projected.provenance == "approx_source_inputs_projected"
    # Projection in a separate call must not alter the original signed estimate.
    np.testing.assert_allclose(signed.p_pu, [0.5, -0.1], atol=1e-15)


@pytest.mark.parametrize("problem", ["unknown_policy", "missing_master", "nan_terminal", "inf_master"])
def test_approx_source_rejects_invalid_policy_or_meter_values(problem):
    observed = _approx_observed()
    kwargs = {}
    if problem == "unknown_policy":
        kwargs["negative_p_policy"] = "clip"
    elif problem == "missing_master":
        observed.pop("Q0_measured")
    elif problem == "nan_terminal":
        observed["P_terminal"].iloc[0, 0] = np.nan
    else:
        observed["P0_measured"].iloc[0] = np.inf
    with pytest.raises(ValueError):
        approx_source_inputs(observed, **kwargs)
