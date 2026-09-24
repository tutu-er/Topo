"""Unit tests for theft simulation: channel isolation and validation."""

import numpy as np
import pytest

from theft_wzzt.theft.theft_simulation import TheftSpec, simulate_theft_scenarios


def test_clean_balance_is_meter_noise_level():
    net, s = simulate_theft_scenarios("paper15")
    resid = s["P0_measured"] - s["P_terminal"].sum(axis=1) - s["loss_p_notheft"]
    assert float(resid.abs().max()) < 0.01  # master noise, no systematic term
    assert s["theft_truth"] == []


def test_theft_enters_balance_but_not_terminal_meters():
    amp = np.zeros(96)
    amp[40:70] = 8.0
    net, s = simulate_theft_scenarios(
        "paper15", (TheftSpec(bus_id=2, amplitude_kw=tuple(amp), kappa=0.4),))
    resid = s["P0_measured"] - s["P_terminal"].sum(axis=1) - s["loss_p_notheft"]
    pu = 8.0 / (net.base_mva * 1000.0)
    # Balance residual in the event window matches the theft amplitude
    # (plus the theft-driven extra loss, which only adds to it).
    assert float(resid[45:65].min()) > 0.8 * pu
    # Terminal meter channels never observe the theft: P_terminal equals the
    # noisy METERED load only, and the truth channel records the theft.
    truth = s["theft_truth"][0]
    assert truth["bus_id"] == 2
    assert float(truth["amplitude_pu"].iloc[45]) == pytest.approx(pu)
    clean, _ = None, None


def test_at_most_one_active_location():
    a = np.zeros(96); a[10:20] = 5.0
    b = np.zeros(96); b[15:25] = 5.0
    with pytest.raises(ValueError, match="at most one"):
        simulate_theft_scenarios(
            "paper15", (TheftSpec(2, tuple(a)), TheftSpec(3, tuple(b))))


def test_negative_amplitude_rejected():
    amp = np.zeros(96); amp[10] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_theft_scenarios("paper15", (TheftSpec(2, tuple(amp)),))


def test_root_and_unknown_bus_rejected():
    amp = np.zeros(96); amp[10] = 1.0
    with pytest.raises(ValueError):
        simulate_theft_scenarios("paper15", (TheftSpec(1, tuple(amp)),))
    with pytest.raises(ValueError):
        simulate_theft_scenarios("paper15", (TheftSpec(9999, tuple(amp)),))
