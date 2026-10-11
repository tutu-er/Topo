"""Independent nodal-AC and certificate-direction checks for the v8 pilot.

The rectangular nodal solver below does not call the implementation's path
matrix, fixed-point iteration, current recursion, or remainder formula.
"""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).parent))
import ac_shared_v8 as ac


PAIRS = {"cherry12": (0, 1), "cherry13": (0, 2), "cherry23": (1, 2)}


def nodal_solution(candidate, theta, p, q, source):
    """KCL in rectangular node voltages, with root voltage fixed to one."""
    edges = [(0, 1), (1, 2)] + [
        (2 if j in PAIRS[candidate] else 1, j + 3) for j in range(3)
    ]
    z = np.asarray(theta) * (1 + .5j) / 2
    admittance = np.zeros((6, 6), complex)
    for (u, v), impedance in zip(edges, z):
        g = 1 / impedance
        admittance[u, u] += g
        admittance[v, v] += g
        admittance[u, v] -= g
        admittance[v, u] -= g
    loads = np.r_[source * (1 + .3j), 0j, np.asarray(p) + 1j * np.asarray(q)]

    def residual(x):
        voltages = np.r_[1 + 0j, x[:5] + 1j * x[5:]]
        mismatch = (admittance @ voltages)[1:] + np.conj(loads / voltages[1:])
        return np.r_[mismatch.real, mismatch.imag]

    solution = root(residual, np.r_[np.ones(5), np.zeros(5)], tol=1e-11)
    assert np.max(abs(residual(solution.x))) < 3e-9
    voltages = np.r_[1 + 0j, solution.x[:5] + 1j * solution.x[5:]]
    current = np.array([(voltages[u] - voltages[v]) / impedance
                        for (u, v), impedance in zip(edges, z)])
    return voltages[1:], current


@pytest.mark.parametrize("candidate", tuple(PAIRS))
@pytest.mark.parametrize("theta", [
    np.array(ac.BOX)[:, 0], np.array(ac.BOX)[:, 1],
    np.array([.027, .018, .035, .081, .052]),
])
def test_forward_matches_independent_nodal_equations_at_box_boundaries(candidate, theta):
    p, q, source = np.array([.5, .2, .26]), np.array([.06, .06, .078]), .22
    voltage, current = nodal_solution(candidate, theta, p, q, source)
    result = ac.ac_forward(candidate, theta, p, q, source, .3 * source)
    np.testing.assert_allclose(result["voltage"], voltage, atol=2e-10, rtol=0)
    np.testing.assert_allclose(result["ell"], abs(current) ** 2, atol=2e-9, rtol=0)
    assert result["identity_error"] < 1e-12
    assert result["eta"].min() >= 0
    assert abs(voltage).min() >= .9


@pytest.mark.parametrize("candidate", tuple(PAIRS))
def test_loss_planes_cover_independent_ac_states_at_every_box_corner(candidate):
    p, q, source = np.array([.5, .2, .2]), np.array([.06] * 3), .22
    obs = dict(P=p[None, :], Q=q[None, :], source_p_abs_cap=[.22],
               source_q_abs_cap=[.066])
    planes, current_caps = ac.envelope(obs, candidate)
    _, incidence, descendants = ac.geometry(candidate)
    for picks in product((0, 1), repeat=5):
        theta = np.array([ac.BOX[j][pick] for j, pick in enumerate(picks)])
        voltage, current = nodal_solution(candidate, theta, p, q, source)
        ell = abs(current) ** 2
        assert np.all(ell <= np.asarray(current_caps[0], float) + 2e-9)
        # Independently use each physical branch's voltage drop and downstream
        # measured injections to obtain its nonlinear remainder.
        edges = [(0, 1), (1, 2)] + [
            (2 if j in PAIRS[candidate] else 1, j + 3) for j in range(3)
        ]
        full_voltage = np.r_[1 + 0j, voltage]
        effective = p + .5*q
        nonlinear = []
        for e, (u, v) in enumerate(edges):
            linear = theta[e] * (incidence[:, e] @ effective)
            if e == 0:
                linear += theta[e] * 1.15 * source
            nonlinear.append(abs(full_voltage[u])**2 - abs(full_voltage[v])**2 - linear)
        nonlinear = np.asarray(nonlinear)
        assert nonlinear.min() >= -2e-12
        for e in range(5):
            for co, constant in planes[0][e]:
                ceiling = float(sum((value*ac.f(t) for value, t in zip(co, theta)), constant))
                assert nonlinear[e] <= ceiling + 2e-12


def test_true_ac_candidate_retained_with_coordinate_error_corners():
    theta = np.array(ac.BOX)[:, 1]
    obs, truth = ac.make_data("cherry23", theta, 9811, "rich", 2, "endpoint")
    obs["Y_drop"] = np.asarray(truth["Y_true"]) + 1e-5 * np.resize([1., -1.], (36, 3))
    for sharing in ("path", "output"):
        result = ac.solve_outer(obs, "cherry23", sharing=sharing)
        assert result["lower"] == 0
        assert result["upper"] < 2e-12
        assert result["exact_primal_verified"]
        assert result["exact_box_dual_verified"]


def test_shared_path_remainders_strengthen_same_domain_output_relaxation():
    obs, _ = ac.make_data("cherry12", [.02, .01, .06, .06, .06],
                          9801, "flat", 0, "endpoint")
    path = ac.solve_outer(obs, "cherry13", sharing="path")
    output = ac.solve_outer(obs, "cherry13", sharing="output")
    assert path["lower"] > output["upper"] + 1e-4
    assert path["gap"] < 1e-10 and output["gap"] < 1e-10


def test_common_source_projection_does_not_cancel_ac_loss_response():
    theta = [.027, .018, .035, .081, .052]
    p, q = np.array([.5, .2, .2]), np.array([.06] * 3)
    responses = []
    for source in (.02, .22):
        voltage, _ = nodal_solution("cherry12", theta, p, q, source)
        responses.append(1 - abs(voltage[2:])**2)
    change = responses[1] - responses[0]
    # The linear contribution is common; the exact AC response is not.
    assert np.ptp(change) > 1e-6
    assert np.ptp(change) < 1e-5


def test_ignoring_ac_can_falsely_exclude_truth_even_after_parameter_refit():
    obs, _ = ac.make_data("cherry12", [.02, .01, .06, .06, .06],
                          9801, "rich", 0, "endpoint")
    invalid = ac.solve_outer(obs, "cherry12", linear_only=True)
    valid = ac.solve_outer(obs, "cherry12")
    assert invalid["lower"] > 5e-6
    assert valid["lower"] == 0
    assert valid["upper"] < 1e-12


@pytest.mark.parametrize("candidate", tuple(PAIRS))
def test_batched_ac_matches_independent_nodal_and_scalar_solutions(candidate):
    theta = [.027, .018, .035, .081, .052]
    p = np.array([[.2, .2, .2], [.5, .2, .2], [.2, .5, .2], [.2, .2, .5]])
    q = np.full_like(p, .06)
    source = np.array([.02, .08, .14, .22])
    y, voltages, residual = ac.ac_batch(candidate, theta, p, q, source)
    assert residual < 1e-12
    for t in range(len(p)):
        independent, _ = nodal_solution(candidate, theta, p[t], q[t], source[t])
        scalar = ac.ac_forward(candidate, theta, p[t], q[t], source[t], .3*source[t])
        np.testing.assert_allclose(voltages[t], independent, atol=2e-10, rtol=0)
        np.testing.assert_allclose(y[t], 1-abs(independent[2:])**2, atol=3e-10, rtol=0)
        np.testing.assert_allclose(voltages[t], scalar["voltage"], atol=1e-13, rtol=0)
    scalar_source = ac.ac_batch(candidate, theta, p, q, .12)
    vector_source = ac.ac_batch(candidate, theta, p, q, np.full(len(p), .12))
    np.testing.assert_array_equal(scalar_source[0], vector_source[0])
