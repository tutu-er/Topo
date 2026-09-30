"""Physical extra-load scenarios keep honest terminal meters and AC balance."""

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.scenario import simulation
from rnj_wzzt.scenario.simulation import _simulate_pool
from rnj_wzzt.scenario.unmetered_load import UnmeteredLoadSpec, simulate_unmetered_scenario


def _source(bus_id=2):
    index = pd.RangeIndex(96, name="t")
    p_kw = pd.Series(0.0, index=index)
    q_kvar = pd.Series(0.0, index=index)
    p_kw.iloc[44:52] = 4.0
    q_kvar.iloc[44:52] = 1.6
    return UnmeteredLoadSpec(bus_id=bus_id, p_kw=p_kw, q_kvar=q_kvar)


def _assert_same(actual, expected):
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("scenario_index,t_count", [(0, 16), (0, 96), (2, 16)])
def test_none_and_zero_source_reproduce_existing_observations(scenario_index, t_count):
    kwargs = dict(scenario_index=scenario_index, replicate=1, t_count=t_count, scenario_suite="reference",
                  pq_noise_rel=0.015, v_noise_rel=0.0003)
    _, original = _simulate_pool("paper15", t_count, 1, scenario_index + 1, 0.015, 0.0003,
                                 scenario_suite="reference")
    _, none = simulate_unmetered_scenario("paper15", source=None, **kwargs)
    index = pd.RangeIndex(96, name="t")
    zero = UnmeteredLoadSpec(2, pd.Series(0.0, index=index), pd.Series(0.0, index=index))
    _, all_zero = simulate_unmetered_scenario("paper15", source=zero, **kwargs)
    for bundle in (none, all_zero):
        assert bundle.observed["name"] == original[scenario_index]["name"]
        assert bundle.truth["source_bus_id"] is None
        for key in ("P_terminal", "Q_terminal", "V_terminal", "root_voltage", "drop_target"):
            _assert_same(bundle.observed[key], original[scenario_index][key])
    pd.testing.assert_series_equal(none.observed["P0_measured"], all_zero.observed["P0_measured"])
    pd.testing.assert_series_equal(none.observed["Q0_measured"], all_zero.observed["Q0_measured"])


@pytest.mark.parametrize("bus_id", [2, 101])
def test_extra_physical_load_keeps_terminal_meters_and_obeys_ac_balance(bus_id):
    kwargs = dict(scenario_suite="reference", root_observation="exact")
    net, baseline = simulate_unmetered_scenario("paper15", source=None, **kwargs)
    spec = _source(bus_id)
    _, stolen = simulate_unmetered_scenario("paper15", source=spec, **kwargs)
    for key in ("P_terminal", "Q_terminal", "root_voltage"):
        _assert_same(stolen.observed[key], baseline.observed[key])
    assert stolen.observed["name"] == baseline.observed["name"]
    assert not stolen.observed["V_terminal"].equals(baseline.observed["V_terminal"])
    assert stolen.truth["source_bus_id"] == bus_id
    assert stolen.diagnostics["source_energy_kwh"] == pytest.approx(8.0)
    for kind, expected in (("P", spec.p_kw), ("Q", spec.q_kvar)):
        delta = stolen.truth[f"{kind}_physical"] - baseline.truth[f"{kind}_physical"]
        for col in delta.columns:
            reference = expected.to_numpy() / (1000.0 * net.base_mva) if col == bus_id else 0.0
            np.testing.assert_allclose(delta[col], reference, atol=1e-15)

    for kind in ("P", "Q"):
        amount = stolen.truth[f"{kind}_unmetered"]
        head = stolen.truth[f"{kind}0_true"]
        loss = stolen.truth[f"{kind}_loss_true"]
        physical = stolen.truth[f"{kind}_physical"]
        meter = stolen.truth[f"{kind}_terminal_base"]
        np.testing.assert_allclose(head - physical.sum(axis=1) - loss, 0.0, atol=1e-8)
        np.testing.assert_allclose(head - meter.sum(axis=1) - loss, amount, atol=1e-8)
        np.testing.assert_allclose(
            head - baseline.truth[f"{kind}0_true"],
            amount + loss - baseline.truth[f"{kind}_loss_true"], atol=1e-8,
        )
    assert net.base_mva == pytest.approx(0.2)


def test_master_meter_noise_does_not_change_existing_channels():
    kwargs = dict(source=_source(), scenario_suite="reference", pq_noise_rel=0.015,
                  v_noise_rel=0.0003)
    _, exact = simulate_unmetered_scenario("paper15", **kwargs)
    _, noisy = simulate_unmetered_scenario(
        "paper15", master_p_noise_rel=0.02, master_q_noise_rel=0.02, **kwargs,
    )
    for key in ("P_terminal", "Q_terminal", "V_terminal", "root_voltage", "drop_target"):
        _assert_same(noisy.observed[key], exact.observed[key])
    pd.testing.assert_series_equal(noisy.truth["P0_true"], exact.truth["P0_true"])
    assert noisy.observed["scenario_settings"]["master_p_noise_rel"] == 0.02
    assert exact.observed["scenario_settings"]["master_p_noise_rel"] == 0.0
    assert not noisy.observed["P0_measured"].equals(exact.observed["P0_measured"])
    assert not noisy.observed["Q0_measured"].equals(exact.observed["Q0_measured"])


def test_short_observation_is_subset_of_one_full_physical_day():
    kwargs = dict(source=_source(), scenario_suite="reference", replicate=1,
                  pq_noise_rel=0.015, v_noise_rel=0.0003,
                  master_p_noise_rel=0.02, master_q_noise_rel=0.02)
    _, full = simulate_unmetered_scenario("paper15", t_count=96, **kwargs)
    _, short = simulate_unmetered_scenario("paper15", t_count=16, **kwargs)
    indices = list(range(0, 96, 6))
    assert short.observed["source_sample_indices"] == indices
    assert short.observed["scenario_settings"]["observed_sample_count"] == 16
    for key in ("P_terminal", "Q_terminal", "V_terminal", "root_voltage", "drop_target",
                "P0_measured", "Q0_measured"):
        _assert_same(short.observed[key], full.observed[key].iloc[indices])
    pd.testing.assert_series_equal(short.truth["P_unmetered"], full.truth["P_unmetered"])
    pd.testing.assert_series_equal(short.truth["P0_true"], full.truth["P0_true"])


def test_noise_parameters_follow_existing_numeric_conversion():
    _, numeric = simulate_unmetered_scenario(
        "paper15", source=_source(), pq_noise_rel=0.005, v_noise_rel=0.0003,
    )
    _, text = simulate_unmetered_scenario(
        "paper15", source=_source(), pq_noise_rel="0.005", v_noise_rel="0.0003",
    )
    for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target"):
        _assert_same(text.observed[key], numeric.observed[key])


@pytest.mark.parametrize("source,match", [
    (UnmeteredLoadSpec(1, _source().p_kw, _source().q_kvar), "non-root"),
    (UnmeteredLoadSpec(999, _source().p_kw, _source().q_kvar), "non-root"),
    (UnmeteredLoadSpec(True, _source().p_kw, _source().q_kvar), "integer"),
    (UnmeteredLoadSpec(2, pd.Series(-1.0, index=pd.RangeIndex(96, name="t")), _source().q_kvar),
     "nonnegative"),
    (UnmeteredLoadSpec(2, pd.Series(np.nan, index=pd.RangeIndex(96, name="t")), _source().q_kvar),
     "finite"),
    (UnmeteredLoadSpec(2, _source().p_kw, pd.Series(np.inf, index=pd.RangeIndex(96, name="t"))),
     "finite"),
    (UnmeteredLoadSpec(2, pd.Series(0.0, index=pd.RangeIndex(95, name="t")), _source().q_kvar),
     "full physical time index"),
])
def test_invalid_source_is_rejected(source, match):
    with pytest.raises(ValueError, match=match):
        simulate_unmetered_scenario("paper15", source=source)


def test_only_the_requested_day_runs_one_ac_solve(monkeypatch):
    original_solve = simulation.solve_ac_power_flow_timeseries
    calls = 0

    def counted_ac(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(simulation, "solve_ac_power_flow_timeseries", counted_ac)
    _, bundle = simulate_unmetered_scenario(
        "paper15", source=_source(), scenario_index=2, t_count=16,
    )
    assert calls == 1
    assert bundle.observed["scenario_settings"]["profile_seed"] == 21
    assert bundle.observed["scenario_settings"]["profile_scenario"] == "evening_peak_high_load"


def test_ac_failure_at_unobserved_time_is_not_hidden_by_subsampling(monkeypatch):
    original_solve = simulation.solve_ac_power_flow_timeseries
    calls = 0

    def fail_ac(*args, **kwargs):
        nonlocal calls
        result = original_solve(*args, **kwargs)
        calls += 1
        result["converged"] = result["converged"].copy()
        result["converged"].iloc[1] = False  # 1 is not sampled when t_count=16
        return result

    monkeypatch.setattr(simulation, "solve_ac_power_flow_timeseries", fail_ac)
    with pytest.raises(RuntimeError, match="did not converge"):
        simulate_unmetered_scenario("paper15", source=_source(), t_count=16)
    assert calls == 1
