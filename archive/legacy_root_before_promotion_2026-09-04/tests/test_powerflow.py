import numpy as np

from topoident.powerflow import Branch, build_ybus, injections, solve_power_flow


def test_two_bus_power_flow_matches_specified_injection():
    ybus = build_ybus(2, [Branch(0, 1, 0.02, 0.04)])
    p = np.array([0.1, -0.1])
    q = np.array([0.04, -0.04])
    voltage = solve_power_flow(ybus, p, q)
    actual = injections(ybus, voltage)
    assert np.allclose(actual.real[1], p[1], atol=1e-8)
    assert np.allclose(actual.imag[1], q[1], atol=1e-8)
    assert abs(voltage[1]) < 1.0


def test_ybus_is_symmetric_and_has_zero_row_sum():
    ybus = build_ybus(3, [Branch(0, 1, 0.1, 0.2), Branch(1, 2, 0.2, 0.3)])
    assert np.allclose(ybus, ybus.T)
    assert np.allclose(ybus.sum(axis=1), 0.0)
