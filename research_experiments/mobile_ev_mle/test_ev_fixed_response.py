"""Component checks only: no fitter, selector, strategy, MILP, or source solver."""
from copy import deepcopy

import numpy as np
import pytest

from . import likelihood as lk
from .ev_fixed_response import FixedResponse, METHODS, exact_current_reconstruction


FAMILIES = ((), ((0, 1),), ((0, 2),), ((1, 2),))


def _case(family=()):
    candidate = lk.RadialCandidate.from_supports("test", 3, family)
    r = np.linspace(.003, .008, candidate.m)
    x = np.linspace(.001, .003, candidate.m)
    # Includes terminal and common-source reverse flow; root voltage is known.
    p = np.array([[.12, .08, .15], [-.10, .04, .02], [0., -.12, .04]])
    q = np.array([[.03, .02, .04], [-.02, .01, .005], [.01, -.03, .01]])
    sp, sq, root = np.array([.10, -.08, -.15]), np.array([.035, -.02, -.04]), np.array([1., .99, 1.01])
    return candidate, r, x, (p, q, sp, sq, root)


def _perturb(inputs, coordinate, step):
    p, q, sp, sq, root = (a.copy() for a in inputs)
    n = p.shape[1]
    if coordinate < n:
        p[:, coordinate] += step
    elif coordinate < 2*n:
        q[:, coordinate-n] += step
    elif coordinate == 2*n:
        sp += step
    else:
        sq += step
    return p, q, sp, sq, root


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_ell_reconstructs_all_nodes_flows_and_root_balance(family):
    candidate, r, x, inputs = _case(family)
    physical = lk.ac_predict(candidate, r, x, *inputs)
    audit = exact_current_reconstruction(candidate, r, x, *inputs, physical=physical)
    assert audit["ac_value_calls"] == 0
    assert all(value < 2e-11 for value in audit["residuals"].values()), audit["residuals"]
    assert audit["reconstructed_squared_voltage"] == pytest.approx(abs(physical["voltage"])**2, abs=2e-11)
    assert np.min(audit["sending_power"].real) < 0
    assert np.min(audit["receiving_power"].real) < 0
    assert np.all(audit["current_squared"] >= 0)
    # Direct edge-voltage recursion is independent of the loss-drop H builder.
    sending, receiving, ell = (audit[k] for k in ("sending_power", "receiving_power", "current_squared"))
    voltage_u = audit["squared_voltage"]
    for edge, parent in enumerate(candidate.parents):
        parent_u = inputs[-1]**2 if parent < 0 else voltage_u[:, parent]
        z = r[edge]+1j*x[edge]
        send_u = parent_u-2*np.real(np.conj(z)*sending[:, edge])+abs(z)**2*ell[:, edge]
        receive_u = parent_u-2*np.real(np.conj(z)*receiving[:, edge])-abs(z)**2*ell[:, edge]
        assert send_u == pytest.approx(voltage_u[:, edge], abs=2e-11)
        assert receive_u == pytest.approx(voltage_u[:, edge], abs=2e-11)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_all_P_Q_source_columns_match_finite_differences(family, method):
    candidate, r, x, inputs = _case(family)
    response = FixedResponse(candidate, r, x, method=method, anchor=inputs if method == "anchored_i2" else None)
    # Move away from the center so the I2 remainder is actually differentiated.
    point = _perturb(inputs, 1, .015)
    result = response.evaluate(*point)
    t, n = inputs[0].shape
    assert result["values"].shape == (t, n+2)
    assert result["jac"].shape == (t, n+2, 2*n+2)
    for coordinate in range(2*n+2):
        for step in (1e-5, 3e-6):
            plus = response.evaluate(*_perturb(point, coordinate, step), derivatives=False)
            minus = response.evaluate(*_perturb(point, coordinate, -step), derivatives=False)
            finite = (plus["values"]-minus["values"])/(2*step)
            assert finite == pytest.approx(result["jac"][:, :, coordinate], rel=1e-5, abs=2e-7)
            assert plus["jac"] is None and minus["jac"] is None
    assert np.isfinite(result["values"]).all() and np.isfinite(result["jac"]).all()
    assert result["diagnostics"]["source_interface"] == 1
    assert result["diagnostics"]["EV_added_by_response"] is False


@pytest.mark.parametrize("family", FAMILIES)
def test_anchor_matches_value_jacobian_and_no_later_AC_calls(family, monkeypatch):
    candidate, r, x, inputs = _case(family)
    exact = FixedResponse(candidate, r, x, method="ac").evaluate(*inputs)
    anchored = FixedResponse(candidate, r, x, method="anchored_i2", anchor=inputs)
    assert anchored.counters["ac_value_calls"] == anchored.counters["ac_jacobian_calls"] == 1
    def forbidden(*args, **kwargs):
        raise AssertionError("anchored evaluation called AC after construction")
    monkeypatch.setattr(lk, "ac_predict", forbidden)
    monkeypatch.setattr(lk, "_response_derivatives", forbidden)
    at_center = anchored.evaluate(*inputs)
    assert at_center["values"] == pytest.approx(exact["values"], abs=1e-14)
    assert at_center["jac"] == pytest.approx(exact["jac"], abs=1e-14)
    shifted = anchored.evaluate(*_perturb(inputs, 0, .05))
    value_only = anchored.evaluate(*_perturb(inputs, 6, -.02), derivatives=False)
    assert np.isfinite(shifted["values"]).all() and value_only["jac"] is None
    assert anchored.counters["ac_value_calls"] == anchored.counters["ac_jacobian_calls"] == 1
    assert shifted["diagnostics"]["voltage_scope"] == "terminal_nodes"


def test_AC_value_then_jacobian_reuses_forward_and_cached_outputs_are_detached():
    candidate, r, x, inputs = _case()
    response = FixedResponse(candidate, r, x, method="ac")
    first = response.evaluate(*inputs, derivatives=False)
    assert response.counters["ac_value_calls"] == 1 and response.counters["ac_jacobian_calls"] == 0
    second = response.evaluate(*inputs)
    assert response.counters["ac_value_calls"] == response.counters["ac_jacobian_calls"] == 1
    expected = second["values"].copy()
    first["values"][:] = 999.; second["jac"][:] = 999.
    third = response.evaluate(*inputs)
    assert third["values"] == pytest.approx(expected)
    assert np.max(abs(third["jac"])) < 2
    changed = deepcopy(inputs); changed[0][0, 0] += .01
    response.evaluate(*changed, derivatives=False)
    assert response.counters["ac_value_calls"] == 2


def test_fixed_lines_are_copied_readonly_and_all_methods_preserve_inputs():
    candidate, r, x, inputs = _case(((0, 2),))
    original_inputs = tuple(a.copy() for a in inputs)
    r0, x0 = r.copy(), x.copy()
    objects = [FixedResponse(candidate, r, x, method=method, anchor=inputs) for method in METHODS]
    r[:] = .9; x[:] = .7
    for response in objects:
        response.evaluate(*inputs)
        assert np.array_equal(response.r, r0) and np.array_equal(response.x, x0)
        assert not response.r.flags.writeable and not response.x.flags.writeable
        with pytest.raises(ValueError):
            response.r[0] = .1
    assert all(np.array_equal(a, b) for a, b in zip(inputs, original_inputs))


def test_I2_drop_has_correct_receiving_sign_and_full_downstream_loss():
    candidate = lk.RadialCandidate.from_supports("two", 2)
    r, x = np.array([.01, .02, .03]), np.array([.004, .006, .009])
    p, q, sp, sq, root = np.array([[.2, .1]]), np.array([[.05, .03]]), np.array([.04]), np.array([.01]), np.ones(1)
    inputs = p, q, sp, sq, root
    linear = FixedResponse(candidate, r, x, method="linear").evaluate(*inputs)
    corrected = FixedResponse(candidate, r, x, method="i2").evaluate(*inputs)
    assert np.all(corrected["values"][:, :2] < linear["values"][:, :2])
    a = np.array([.34+.09j, .2+.05j, .1+.03j]); z = r+1j*x; ell = abs(a)**2
    sending = a+np.array([np.sum(z*ell), z[1]*ell[1], z[2]*ell[2]])
    root_junction_u = 1-2*np.real(np.conj(z[0])*sending[0])+abs(z[0])**2*ell[0]
    leaf_u = root_junction_u-2*np.real(np.conj(z[1:])*sending[1:])+abs(z[1:])**2*ell[1:]
    assert corrected["values"][0, :2] == pytest.approx(np.sqrt(leaf_u), abs=1e-14)
    master = a[0]+np.sum(z*ell)
    assert corrected["values"][0, 2:] == pytest.approx([master.real, master.imag])


def test_anchor_root_shape_invalid_inputs_and_approximation_failure_are_explicit():
    candidate, r, x, inputs = _case()
    with pytest.raises(ValueError, match="requires"):
        FixedResponse(candidate, r, x, method="anchored_i2")
    with pytest.raises(ValueError, match="unknown"):
        FixedResponse(candidate, r, x, method="missing")
    anchored = FixedResponse(candidate, r, x, method="anchored_i2", anchor=inputs)
    changed = tuple(a.copy() for a in inputs); changed[-1][0] += .001
    with pytest.raises(ValueError, match="root-voltage path"):
        anchored.evaluate(*changed)
    with pytest.raises(ValueError, match="batch shape"):
        anchored.evaluate(*(a[:1] for a in inputs))
    with pytest.raises(ValueError, match="nonnegative"):
        FixedResponse(candidate, -r, x)
    for method in ("linear", "i2"):
        response = FixedResponse(candidate, r, x, method=method)
        bad = tuple(a.copy() for a in inputs); bad[0][:] = 1000.
        with pytest.raises(ValueError, match="squared voltage"):
            response.evaluate(*bad)


def test_current_audit_can_make_one_AC_call_and_rejects_mismatched_state():
    candidate, r, x, inputs = _case()
    result = exact_current_reconstruction(candidate, r, x, *inputs)
    assert result["ac_value_calls"] == 1
    physical = lk.ac_predict(candidate, r, x, *inputs)
    with pytest.raises(ValueError, match="different p"):
        exact_current_reconstruction(candidate, r, x, *_perturb(inputs, 0, .1), physical=physical)
    with pytest.raises(ValueError, match="fixed R/X"):
        exact_current_reconstruction(candidate, r*2, x, *inputs, physical=physical)


@pytest.mark.parametrize("method", METHODS)
def test_zero_lines_and_zero_flow_limits(method):
    candidate, _, _, inputs = _case()
    zeros = np.zeros(candidate.m)
    response = FixedResponse(candidate, zeros, zeros, method=method, anchor=inputs)
    result = response.evaluate(*inputs)
    p, q, sp, sq, root = inputs
    assert result["values"][:, :3] == pytest.approx(np.broadcast_to(root[:, None], (3, 3)), abs=1e-14)
    assert result["values"][:, 3] == pytest.approx(p.sum(axis=1)+sp, abs=1e-14)
    assert result["values"][:, 4] == pytest.approx(q.sum(axis=1)+sq, abs=1e-14)
    assert result["jac"][:, :3] == pytest.approx(np.zeros((3, 3, 8)), abs=1e-14)
    zero_inputs = tuple(np.zeros_like(a) if i < 4 else a for i, a in enumerate(inputs))
    zero = response.evaluate(*zero_inputs)
    assert np.isfinite(zero["values"]).all() and np.isfinite(zero["jac"]).all()
