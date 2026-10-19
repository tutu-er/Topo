"""Numerical check of the exact AC gauge without common-source telemetry.

This test checks a necessary identification obstruction, not a statistical
calibration claim. The alternative is constructed from physical edges/KCL;
the forward solver must then reproduce the identical observed leaf channels.
"""
import numpy as np
import pytest

from .likelihood import RadialCandidate, ac_predict


def _edge_currents(candidate, r, x, nonroot_voltage, root):
    """Physical edge currents from voltage differences; no B/Zbus use."""
    voltage = np.c_[root, nonroot_voltage]
    return np.column_stack([
        (voltage[:,u]-voltage[:,v])/(r[e]+1j*x[e])
        for e,(u,v) in enumerate(candidate.edges)
    ])


def _independent_kcl(candidate, r, x, voltage, root, p, q, source):
    """Complex nodal power mismatch assembled directly from graph edges."""
    node_voltage = np.c_[root, voltage]
    current = _edge_currents(candidate,r,x,voltage,root)
    incoming_minus_outgoing = np.zeros_like(voltage)
    for e,(u,v) in enumerate(candidate.edges):
        incoming_minus_outgoing[:,v-1] += current[:,e]
        if u:
            incoming_minus_outgoing[:,u-1] -= current[:,e]
    load = np.zeros_like(voltage)
    load[:,0] = source
    load[:,list(candidate.terminal_edges)] += p+1j*q
    return np.max(abs(node_voltage[:,1:]*np.conj(incoming_minus_outgoing)-load))


@pytest.mark.parametrize("perturbation", ["trunk_only","phase_only","trunk_and_phase"])
def test_exact_ac_source_trunk_phase_gauge_preserves_leaf_channels(perturbation):
    # There is one shared source bus and a nontrivial internal clade. No root
    # power or common-source telemetry is included in the invariant channels.
    candidate = RadialCandidate.from_supports("gauge_fixture",3,[(0,1)])
    r = np.array([.007,.012,.024,.031,.026])
    x = np.array([.004,.009,.010,.015,.013])
    p = np.array([[.080,.110,.070],[.120,.090,.060],
                  [.095,.100,.085],[.075,.125,.065]])
    q = np.array([[.025,.040,.028],[.040,.028,.020],
                  [.035,.030,.033],[.021,.045,.024]])
    source_p = np.array([.055,.062,.080,.070])
    source_q = np.array([.025,.033,.024,.031])
    root = np.array([.998,1.000,1.004,1.001])
    original = ac_predict(candidate,r,x,p,q,source_p,source_q,root)
    alternate_r,alternate_x = r.copy(),x.copy()
    if perturbation != "phase_only":
        alternate_r[0] *= 1.12
        alternate_x[0] *= .92
    phase = (np.zeros(len(p)) if perturbation=="trunk_only" else
             np.array([.00010,-.00015,.00008,-.00012]))
    rotation = np.exp(1j*phase)
    transformed_voltage = original["voltage"]*rotation[:,None]
    # Every branch below the shared junction rotates with the downstream
    # voltage; the trunk alone is recomputed against the fixed physical root.
    original_current = _edge_currents(candidate,r,x,original["voltage"],root)
    children = [e for e,(u,_) in enumerate(candidate.edges) if u==1]
    downstream_current = original_current[:,children].sum(axis=1)*rotation
    alternate_trunk = ((root-transformed_voltage[:,0])/
                       (alternate_r[0]+1j*alternate_x[0]))
    alternate_source = transformed_voltage[:,0]*np.conj(
        alternate_trunk-downstream_current)

    # Both witnesses lie strictly inside the first research model's declared
    # domains; this is not an artifact of clipping or unbounded source power.
    assert np.all((alternate_r>1e-6)&(alternate_r<.1))
    assert np.all((alternate_x>1e-6)&(alternate_x<.1))
    assert np.max(abs(alternate_source.real))<.5
    assert np.max(abs(alternate_source.imag))<.5
    assert .85<abs(transformed_voltage).min()
    assert abs(transformed_voltage).max()<1.15
    assert _independent_kcl(candidate,alternate_r,alternate_x,transformed_voltage,
                            root,p,q,alternate_source)<2e-12

    alternate = ac_predict(candidate,alternate_r,alternate_x,p,q,
                           alternate_source.real,alternate_source.imag,root)
    np.testing.assert_allclose(alternate["voltage"],transformed_voltage,atol=1e-11,rtol=0)
    np.testing.assert_allclose(alternate["p"],original["p"],atol=0,rtol=0)
    np.testing.assert_allclose(alternate["q"],original["q"],atol=0,rtol=0)
    np.testing.assert_allclose(alternate["v"],original["v"],atol=1e-11,rtol=0)
    assert alternate["power_balance_residual"]<1e-11
    assert np.max(abs(alternate_source-(source_p+1j*source_q)))>1e-3
    original_master = original["master_p"]+1j*original["master_q"]
    alternate_master = alternate["master_p"]+1j*alternate["master_q"]
    assert np.max(abs(alternate_master-original_master))>1e-3
    if perturbation != "phase_only":
        assert alternate_r[0]!=r[0] and alternate_x[0]!=x[0]
    else:
        np.testing.assert_array_equal(alternate_r,r)
        np.testing.assert_array_equal(alternate_x,x)
