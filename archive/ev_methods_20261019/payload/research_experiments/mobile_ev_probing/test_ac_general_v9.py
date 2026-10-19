"""Independent nodal-AC, coverage, and numerical-certificate tests for v9.

No recorded topology accuracy is inferred from these unit tests.  The nodal
solver builds admittance directly from graph edges, not implementation B/M.
"""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).parent))
import ac_general_v9 as ac
import ac_shared_v8 as old


def trees():
    return (
        ac.RadialCandidate.from_supports("six_binary", 6, [(0, 1, 2), (3, 4, 5), (0, 1), (3, 4)]),
        ac.RadialCandidate.from_supports("six_nonbinary", 6, [(0, 1, 2, 3), (0, 1)]),
        ac.RadialCandidate.from_supports("twelve_binary", 12,
            [tuple(range(6)), tuple(range(6, 12)), (0, 1, 2), (3, 4, 5),
             (6, 7, 8), (9, 10, 11), (0, 1), (3, 4), (6, 7), (9, 10)]),
    )


def nodal(candidate, theta, p, q, source_p, source_q, kappa=.5):
    """Independent rectangular nodal KCL, using physical edge list only."""
    z = np.asarray(theta)*(1+1j*kappa)/2
    admittance = np.zeros((candidate.m+1, candidate.m+1), complex)
    for (u, v), impedance in zip(candidate.edges, z):
        g = 1/impedance
        admittance[u, u] += g; admittance[v, v] += g
        admittance[u, v] -= g; admittance[v, u] -= g
    loads = np.zeros(candidate.m, complex)
    loads[0] = source_p+1j*source_q
    loads[list(candidate.terminal_edges)] += np.asarray(p)+1j*np.asarray(q)

    def residual(x):
        voltage = np.r_[1+0j, x[:candidate.m]+1j*x[candidate.m:]]
        mismatch = (admittance@voltage)[1:]+np.conj(loads/voltage[1:])
        return np.r_[mismatch.real, mismatch.imag]
    solved = root(residual, np.r_[np.ones(candidate.m), np.zeros(candidate.m)], tol=1e-11)
    assert np.max(abs(residual(solved.x))) < 2e-9
    voltage = np.r_[1+0j, solved.x[:candidate.m]+1j*solved.x[candidate.m:]]
    current = np.array([(voltage[u]-voltage[v])/impedance
                        for (u, v), impedance in zip(candidate.edges, z)])
    return voltage, current


def sample(candidate, count=8, seed=9441, pq_error=0.):
    rng = np.random.default_rng(seed)
    theta = np.array([.014 if len(s) == candidate.n else .025 if len(s) == 1 else .035
                      for s in candidate.supports])
    p = rng.uniform(.03, .11, (count, candidate.n))
    q = .3*p
    source = rng.uniform(.04, .08, count)
    y, voltage, error = ac.ac_batch(candidate, theta, p, q, source, .3*source)
    obs = dict(P=p.copy(), Q=q.copy(), Y_drop=y.copy(), P_error=pq_error, Q_error=pq_error,
               drop_error_bound=2e-5, source_nominal_eff=.069, source_error_eff=.03,
               source_p_abs_cap=.1, source_q_abs_cap=.03)
    box = np.array([[.004, .03] if len(s) == candidate.n else
                    [.005, .08] if len(s) == 1 else [.000001, .055]
                    for s in candidate.supports])
    return obs, theta, box, source, voltage


@pytest.mark.parametrize("candidate", trees(), ids=lambda x: x.name)
@pytest.mark.parametrize("kappa", [0., .5, 1.2])
def test_general_forward_matches_independent_nodal_ac(candidate, kappa):
    theta = np.linspace(.008, .025, candidate.m)
    p, q, source = np.linspace(.025, .11, candidate.n), np.full(candidate.n, .025), .08
    independent, current = nodal(candidate, theta, p, q, source, .03, kappa)
    answer = ac.ac_forward(candidate, theta, p, q, source, .03, kappa=kappa)
    np.testing.assert_allclose(answer["voltage"], independent[1:], atol=2e-10, rtol=0)
    np.testing.assert_allclose(answer["ell"], abs(current)**2, atol=2e-9, rtol=0)
    assert answer["identity_error"] < 2e-12
    assert abs(independent).min() > .9


@pytest.mark.parametrize("candidate", trees(), ids=lambda x: x.name)
def test_loss_planes_cover_nodal_ac_boundary_and_random_parameters(candidate):
    obs, theta, box, source, voltage = sample(candidate, count=1, pq_error=.002)
    planes, caps = ac.envelope(obs, candidate, box)
    rng = np.random.default_rng(9442)
    # Endpoints, alternating corners and random interior parameters probe an
    # unequal per-edge box; validity over the full box follows from McCormick.
    settings = [box[:, 0], box[:, 1], box[np.arange(candidate.m), np.arange(candidate.m) % 2]]
    settings += [rng.uniform(box[:, 0], box[:, 1]) for _ in range(12)]
    p = obs["P"][0]+.002
    q = obs["Q"][0]-.002
    for current_theta in settings:
        v, branch_current = nodal(candidate, current_theta, p, q, .1, .03)
        assert abs(v).min() > .9
        assert np.all(abs(branch_current)**2 <= np.array(caps[0], float)+2e-9)
        # Physical branch drops minus loss-free downstream injections.
        for e, (u, w) in enumerate(candidate.edges):
            downstream = list(candidate.supports[e])
            linear = current_theta[e]*sum(p[j]+.5*q[j] for j in downstream)
            if e == 0:
                linear += current_theta[e]*(.1+.5*.03)
            remainder = abs(v[u])**2-abs(v[w])**2-linear
            assert remainder >= -2e-12
            for co, constant in planes[0][e]:
                ceiling = float(sum((a*ac.f(t) for a, t in zip(co, current_theta)), constant))
                assert remainder <= ceiling+2e-12


def test_canonical_laminar_geometry_rejects_crossing_supports():
    with pytest.raises(ValueError, match="Crossing"):
        ac.RadialCandidate.from_supports("invalid", 6, [(0, 1, 2), (1, 2, 3)])
    with pytest.raises(ValueError, match="out-of-range"):
        ac.RadialCandidate.from_supports("invalid", 6, [(0, 6)])
    first = trees()[0]
    second = ac.RadialCandidate.from_supports("same", 6, reversed(first.supports))
    assert first.supports == second.supports
    np.testing.assert_array_equal(first.M, second.M)
    assert first.parents[0] == -1
    assert all(parent < e for e, parent in enumerate(first.parents))


@pytest.mark.parametrize("sharing", ["path", "output"])
def test_true_six_terminal_ac_retained_at_pq_and_voltage_box_corners(sharing):
    candidate = trees()[0]
    obs, theta, box, source, voltage = sample(candidate, pq_error=.003)
    obs["P"] += .003*np.resize([1., -1.], obs["P"].shape)
    obs["Q"] += .003*np.resize([-1., 1.], obs["Q"].shape)
    obs["Y_drop"] += 2e-5*np.resize([1., -1.], obs["Y_drop"].shape)
    result = ac.solve_outer(obs, candidate, box=box, sharing=sharing)
    assert result["lower"] == 0
    assert result["upper"] < 1e-12
    assert result["exact_box_dual_verified"] and result["exact_primal_verified"]


def test_general_certificate_reduces_to_frozen_v8_certificate():
    obs, _ = old.make_data("cherry12", [.02, .01, .06, .06, .06], 9812, "flat", 0, "endpoint")
    for sharing in ("path", "output"):
        candidate = ac.RadialCandidate.from_supports("cherry13", 3, [(0, 2)])
        new = ac.solve_outer(obs, candidate, box=old.BOX, sharing=sharing)
        prior = old.solve_outer(obs, "cherry13", sharing=sharing)
        assert max(new["lower"], prior["lower"]) <= min(new["upper"], prior["upper"])+1e-12
        assert new["gap"] < 1e-10
        assert new["lower"] > 0


def test_history_intersection_and_measurement_domain_monotonicity():
    candidate = trees()[0]
    rival = ac.RadialCandidate.from_supports("rival", 6, [(0, 2, 4), (1, 3, 5), (0, 2), (1, 3)])
    obs, _, box, _, _ = sample(candidate, count=12)
    history = {key: value[:4] if isinstance(value, np.ndarray) else value for key, value in obs.items()}
    hist = ac.solve_outer(history, rival, box=box)
    joint = ac.solve_outer(obs, rival, box=box)
    loose = ac.solve_outer(dict(obs, drop_error_bound=.004, P_error=.006, Q_error=.006), rival, box=box)
    output = ac.solve_outer(obs, rival, box=box, sharing="output")
    assert joint["upper"]+1e-11 >= hist["lower"]
    assert joint["upper"]+1e-11 >= loose["lower"]
    assert joint["upper"]+1e-11 >= output["lower"]
    assert joint["gap"] < 1e-10


def test_failed_lp_and_local_fit_do_not_exclude(monkeypatch):
    candidate = trees()[0]
    obs, _, box, _, _ = sample(candidate, count=2)
    monkeypatch.setattr(ac, "linprog", lambda *args, **kwargs: SimpleNamespace(
        status=1, success=False, message="test budget stop", x=None))
    result = ac.solve_outer(obs, candidate, box=box)
    assert result["status"] == "solver_failure_unresolved"
    assert result["lower"] is None
    monkeypatch.setattr(ac, "ac_batch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
    reference = ac.fit_ac_reference(obs, candidate, box=box, outer_result=result, max_nfev=1)
    assert not reference["historical_numeric_witness"]
    assert reference["status"] == "local_fit_failed_unresolved"


def test_numeric_reference_checks_original_meter_boxes():
    candidate = trees()[0]
    obs, theta, box, source, voltage = sample(candidate, count=6)
    fit = ac.fit_ac_reference(obs, candidate, box=box, outer_result=dict(theta=theta), max_nfev=100)
    assert fit["historical_numeric_witness"]
    assert fit["max_abs_residual"] <= obs["drop_error_bound"]
    assert fit["ac_equation_residual"] < 1e-11


@pytest.mark.parametrize("sharing", ["path", "output"])
@pytest.mark.parametrize("gain_corner", [False, True])
def test_shared_gain_model_covers_fixed_gains_jitter_and_pq_errors(sharing, gain_corner):
    candidate = trees()[0]
    obs, theta, box, source, voltage = sample(candidate, count=9, pq_error=.002)
    gains = np.resize([.999, 1.001], candidate.n+1) if gain_corner else np.linspace(.9992, 1.0008, candidate.n+1)
    jitter = 2e-5
    full_v = np.c_[np.ones(len(voltage)), abs(voltage[:, candidate.terminal_edges])]
    vhat = full_v*gains+jitter*np.resize([1., -1.], full_v.shape)
    obs["Y_drop"] = vhat[:, :1]**2-vhat[:, 1:]**2
    obs["P"] += .002*np.resize([1., -1.], obs["P"].shape)
    obs["Q"] -= .002*np.resize([1., -1.], obs["Q"].shape)
    gain_model = dict(gain_bounds=(.999, 1.001), root_jitter=jitter, terminal_jitter=jitter, vmax=1.1)
    result = ac.solve_outer(obs, candidate, box=box, sharing=sharing, shared_voltage_gain=gain_model)
    assert result["lower"] == 0
    assert result["upper"] < 1e-12
    assert result["voltage_model"] == "fixed_gain_bounded_jitter"
    assert result["exact_primal_verified"]


def test_identity_gain_model_equals_ordinary_jitter_bound():
    candidate = trees()[0]
    rival = ac.RadialCandidate.from_supports("rival", 6, [(0, 2, 4), (1, 3, 5), (0, 2), (1, 3)])
    obs, theta, box, source, voltage = sample(candidate, count=8)
    jitter = 1e-5
    cap = 2*jitter+jitter**2+2*1.1*jitter+jitter**2
    obs["drop_error_bound"] = cap
    one = ac.solve_outer(obs, rival, box=box)
    two = ac.solve_outer(obs, rival, box=box, shared_voltage_gain=dict(
        gain_bounds=(1., 1.), root_jitter=jitter, terminal_jitter=jitter, vmax=1.1))
    assert max(one["lower"], two["lower"]) <= min(one["upper"], two["upper"])+1e-12


def test_squared_voltage_conversion_is_outward_and_accounts_for_root_meter():
    observed = np.array([.9, 1., 1.1])
    delta = .001
    bound = ac.squared_drop_error(observed, delta, root_error=delta)
    for i, value in enumerate(observed):
        exact = 2*abs(ac.f(value))*ac.f(delta)+ac.f(delta)**2+2*ac.f(delta)+ac.f(delta)**2
        assert Fraction(float(bound[i])) >= exact
        for terminal_sign in (-1, 1):
            for root_sign in (-1, 1):
                actual_drop = 1-(value+terminal_sign*delta)**2
                measured_drop = (1+root_sign*delta)**2-value**2
                assert abs(actual_drop-measured_drop) <= bound[i]+1e-15
