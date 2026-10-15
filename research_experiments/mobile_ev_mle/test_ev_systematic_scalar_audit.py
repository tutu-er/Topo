"""Explicit feasible-source and Fenchel-bound fixtures; no optimizer calls."""
from copy import deepcopy
import numpy as np
import pytest
from . import ev_systematic_scalar_audit as audit


def node(c=.2, p=1.):
    c = np.array([c, 0.]); b = np.array([1., 0.]); G = np.array([[1., 0.], [0., 0.]])
    lo, hi, center = np.array([-.25, 0.]), np.array([.25, 0.]), np.zeros(2)
    source = np.array([np.clip(-c[0] - p, -.25, .25), 0.])
    residual = c + b * p + G @ source
    primal = float(residual @ residual); g = 2 * G.T @ residual
    correction = float(np.minimum(g * (lo - source), g * (hi - source)).sum())
    guard = 256 * np.finfo(float).eps * (1 + abs(primal) + abs(correction)
        + float(abs(g) @ (abs(lo) + abs(hi) + 2 * abs(source))))
    inner = dict(physical_parameter=source, residual=residual, primal_upper=primal,
        numeric_guard=guard, tangent_lower_numeric=max(0., min(primal, primal + correction - guard)))
    coefficient = G.T @ residual
    intercept = float(residual @ c - .5 * primal + coefficient @ np.where(coefficient >= 0, lo, hi))
    scalar_guard = 1e-12
    dual = max(0., intercept + (b @ residual) * p - scalar_guard)
    out = dict(p=p, source_QP=inner, residual=residual, primal_upper=.5 * primal,
        dual_intercept=intercept - scalar_guard, dual_slope=float(b @ residual),
        dual_numeric_guard=scalar_guard, dual_lower=dual, primal_dual_gap=.5 * primal - dual)
    return (c, b, G, lo, hi, center), out


@pytest.mark.parametrize('p', [-1., 0., 1.])
def test_independent_endpoint_primal_and_dual(p):
    args, row = node(p=p)
    assert audit.node_check(*args, row) == 0.


@pytest.mark.parametrize('field', ['primal_upper', 'dual_lower', 'dual_slope'])
def test_tampered_node_certificate_is_rejected(field):
    args, row = node()
    row = deepcopy(row); row[field] += .01
    with pytest.raises(ValueError):
        audit.node_check(*args, row)


def test_reused_zero_witness_needs_its_site_specific_slope():
    args, row = node(c=.7, p=0.)
    c, b, G, lo, hi, center = args
    row['dual_slope'] *= 2
    assert audit.node_check(c, 2 * b, G, lo, hi, center, row) == 0.


def test_compact_node_still_checks_source_and_residual():
    args, row = node()
    row['physical_source'] = row['source_QP']['physical_parameter']
    row['source_QP'] = dict(success=True, status=3)
    assert audit.node_check(*args, row) == 0.
    bad = deepcopy(row); bad['physical_source'] = np.array([.26, 0.])
    with pytest.raises(ValueError, match='source box'):
        audit.node_check(*args, bad)


def test_compact_node_tampered_residual_is_rejected():
    args, row = node()
    row['physical_source'] = row['source_QP']['physical_parameter']
    row['source_QP'] = dict(success=True, status=3)
    row['residual'] = row['residual'] + .01
    with pytest.raises(ValueError, match='feasible-source residual'):
        audit.node_check(*args, row)
