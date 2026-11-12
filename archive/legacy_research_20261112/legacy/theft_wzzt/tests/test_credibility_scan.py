"""Unit tests for module 1 (credibility) and baseline A (closed-form scan)."""

import numpy as np
import pytest

from theft_wzzt.theft.credibility import calibrated_rank, evaluate, null_threshold
from theft_wzzt.theft.scan import scan_locations
from theft_wzzt.theft.theft_model import (
    TheftData,
    data_from_scenario,
    nominal_edge_weights,
    tree_from_network,
)
from theft_wzzt.theft.theft_simulation import TheftSpec, simulate_theft_scenarios


def test_calibrated_rank_and_threshold_validation():
    null_gains = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert calibrated_rank(10.0, null_gains) == pytest.approx(1 / 7)
    assert calibrated_rank(0.0, null_gains) == pytest.approx(1.0)
    assert null_threshold(null_gains, 0.05) == pytest.approx(np.quantile(null_gains, 0.95))
    with pytest.raises(ValueError):
        calibrated_rank(1.0, [])
    with pytest.raises(ValueError):
        null_threshold(null_gains, 1.5)


def test_evaluate_channels():
    data = TheftData(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                     np.zeros(2), np.zeros(2), np.zeros(2), 0.5, 2.0)
    out = evaluate(data, np.ones((2, 2)), np.zeros(2))
    assert out["voltage_loss"] == pytest.approx(4 / 0.5)
    assert out["balance_loss"] == 0.0
    assert out["loss"] == out["voltage_loss"]


def test_scan_recovers_location_with_oracle_weights():
    amp = np.zeros(96)
    amp[40:70] = 8.0
    net, s = simulate_theft_scenarios(
        "paper15", (TheftSpec(bus_id=2, amplitude_kw=tuple(amp), kappa=0.4),))
    full = data_from_scenario(s)
    data = TheftData(full.p[44:52], full.q[44:52], full.y[44:52], full.balance[44:52],
                     full.amplitude[44:52], full.extra_q[44:52],
                     full.voltage_scale, full.balance_scale)
    tree = tree_from_network(net)
    r_true, x_true = nominal_edge_weights(net)
    scan = scan_locations(tree, data, r_true, x_true, kappa=0.4)
    # The true location wins the closed-form scan at every event time.
    assert set(scan["best_location"]) == {2}
    assert np.all(scan["best_gain"] > 0)
    # Estimated amplitudes are nonnegative and in the right order of magnitude.
    assert scan["best_amplitude"].min() > 0.0
    assert scan["best_amplitude"].max() < 0.2
