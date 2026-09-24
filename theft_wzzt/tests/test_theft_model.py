"""Unit tests for the theft MILP: oracle localization, nesting, no-free-claim."""

import numpy as np
import pytest

from theft_wzzt.theft.theft_model import (
    TheftData,
    data_from_scenario,
    fit_theft_milp,
    nominal_edge_weights,
    tree_from_network,
)
from theft_wzzt.theft.theft_simulation import TheftSpec, simulate_theft_scenarios


def _fixture():
    amp = np.zeros(96)
    amp[40:70] = 8.0
    net, s = simulate_theft_scenarios(
        "paper15", (TheftSpec(bus_id=2, amplitude_kw=tuple(amp), kappa=0.4),))
    full = data_from_scenario(s)
    data = TheftData(full.p[44:52], full.q[44:52], full.y[44:52], full.balance[44:52],
                     full.amplitude[44:52], full.extra_q[44:52],
                     full.voltage_scale, full.balance_scale)
    return net, data


def test_oracle_fixed_rx_locates_truth():
    net, data = _fixture()
    tree = tree_from_network(net)
    r_true, x_true = nominal_edge_weights(net)
    fit = fit_theft_milp(tree, data, fixed_rx=(r_true, x_true), time_limit=120)
    assert set(fit.selected_locations(tree)) == {2}


def test_h1_nested_in_h0_and_locates_with_tight_bounds():
    net, data = _fixture()
    tree = tree_from_network(net)
    r_true, x_true = nominal_edge_weights(net)
    lo, hi = 0.75 * np.minimum(r_true, x_true), 1.25 * np.maximum(r_true, x_true)
    h0 = fit_theft_milp(tree, data, allow_theft=False, weight_bounds=(lo, hi), time_limit=240)
    h1 = fit_theft_milp(tree, data, allow_theft=True, weight_bounds=(lo, hi), time_limit=240)
    assert h1.objective <= h0.objective + 1e-6
    assert h0.objective - h1.objective > 50.0  # strong gain for an 8 kW theft
    # With refit freedom the MILP may express the parent theft through its
    # children (a documented ambiguity class); exact point localization under
    # refit is evaluated in the experiment matrix, not asserted here.
    assert h1.selection.any()


def test_zero_amplitude_never_reports_detection():
    net, _data = _fixture()
    net, clean = simulate_theft_scenarios("paper15")
    full = data_from_scenario(clean)
    data = TheftData(full.p[44:52], full.q[44:52], full.y[44:52], full.balance[44:52],
                     np.zeros(8), np.zeros(8), full.voltage_scale, full.balance_scale)
    tree = tree_from_network(net)
    r_true, x_true = nominal_edge_weights(net)
    fit = fit_theft_milp(tree, data, weight_bounds=(0.75 * r_true, 1.25 * r_true),
                         time_limit=120)
    assert not fit.selection.any()


def test_unobserved_spur_response_equals_parent():
    from theft_wzzt.data.paper_style_case_bank import CASE_BUILDERS
    from theft_wzzt.theft.theft_simulation import add_unobserved_spur
    from theft_wzzt.theft.scan import response_columns

    net = CASE_BUILDERS["paper15"]()
    net = add_unobserved_spur(net, 5, 900)
    tree = tree_from_network(net)
    r, x = nominal_edge_weights(net)
    f = response_columns(tree, r, x, kappa=0.4)
    i5, i900 = tree.candidates.index(5), tree.candidates.index(900)
    assert np.allclose(f[:, i5], f[:, i900])  # unresolvable by construction
