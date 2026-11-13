"""Offline LinDistFlow ablation from the physical branches, never fitted H.

Only the evaluator imports this module. It consumes the original simulator's
physical network and retains its P/Q convention, power base and meter streams.
"""
from __future__ import annotations

import numpy as np

from ..unified_posterior_dr_20261112.physical import measured_channels


def _oriented_tree(network):
    root = int(network['root_bus'])
    buses = {int(row['bus_id']) for row in network['buses']}
    active = [row for row in network['branches'] if row['status']]
    if root not in buses or len(active) != len(buses)-1:
        raise ValueError('A connected radial physical network is required')
    adjacency = {bus: [] for bus in buses}
    for edge in active:
        a, b = int(edge['from_bus']), int(edge['to_bus'])
        if a not in buses or b not in buses or a == b:
            raise ValueError('Invalid branch endpoints')
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    order, parent, incoming = [root], {root: None}, {}
    for bus in order:
        for child, edge in sorted(adjacency[bus], key=lambda item: item[0]):
            if child == parent[bus]:
                continue
            if child in parent:
                raise ValueError('Physical graph contains a cycle')
            parent[child], incoming[child] = bus, edge
            order.append(child)
    if len(order) != len(buses):
        raise ValueError('Physical graph is disconnected')
    return order, parent, incoming


def build_response(network, labels):
    """Return physical [R X] for root²-v² = [P Q]@[R X].T.

    Positive P/Q denotes consumption. Branch r/x in ohms are converted using
    Zbase=kV_line_line²/MVA_three_phase, exactly as in the original AC source.
    Loads are the passed observed-terminal injections; hidden nodes inject zero.
    """
    labels = tuple(map(int, labels))
    order, parent, incoming = _oriented_tree(network)
    if len(labels) != len(set(labels)) or not labels or not set(labels).issubset(order):
        raise ValueError('Observed terminal labels must be unique physical buses')
    if network['root_bus'] in labels:
        raise ValueError('The root is a voltage reference, not a load terminal')
    zbase = float(network['base_kv'])**2/float(network['base_mva'])
    if not np.isfinite(zbase) or zbase <= 0:
        raise ValueError('Positive finite physical impedance base required')
    # Row C[e,:] indicates all metered terminals below branch e.
    descendants = {bus: np.zeros(len(labels)) for bus in order}
    for index, label in enumerate(labels):
        descendants[label][index] = 1.
    for child in reversed(order[1:]):
        descendants[parent[child]] += descendants[child]
    C = np.array([descendants[child] for child in order[1:]])
    impedance = np.array([[incoming[child]['r_ohm'], incoming[child]['x_ohm']] for child in order[1:]], dtype=float)/zbase
    if not np.isfinite(impedance).all():
        raise ValueError('Finite branch impedances required')
    R = 2*C.T@(impedance[:, 0, None]*C)
    X = 2*C.T@(impedance[:, 1, None]*C)
    return np.column_stack((R, X))


def linear_voltages(network, labels, p, q, root):
    """Lossless radial squared-voltage model with physical impedances."""
    labels = tuple(map(int, labels))
    p, q, root = np.asarray(p, float), np.asarray(q, float), np.asarray(root, float).reshape(-1)
    if p.shape != q.shape or p.ndim != 2 or p.shape != (len(root), len(labels)):
        raise ValueError('P/Q/root dimensions do not match physical terminals')
    if not all(np.isfinite(a).all() for a in (p, q, root)) or np.any(root <= 0):
        raise ValueError('Finite inputs and strictly positive root voltages required')
    H = build_response(network, labels)
    squared_drop = np.column_stack((p, q))@H.T
    squared_voltage = root[:, None]**2-squared_drop
    if np.any(squared_voltage <= 0):
        raise ArithmeticError('LinDistFlow predicts nonpositive squared voltage')
    voltage = np.sqrt(squared_voltage)
    return voltage, dict(model='physical_branch_LinDistFlow', hidden_injection='zero',
        minimum_voltage=float(voltage.min()), maximum_voltage=float(voltage.max()),
        maximum_identity_error=float(np.max(abs(root[:, None]**2-voltage**2-squared_drop))),
        fitted_candidate_used=False, physical_resistance_reactance_used_by='offline_simulator_only')


def observe_linear(module, background, u=None, resource_case=None, response=None):
    """Same evaluator interface as sweep.observe; module is intentionally unused.

    The true load, device response and standardized meter streams are unchanged.
    Only the physical P/Q-to-voltage calculation is replaced for this ablation.
    """
    p, q = background['p'].copy(), background['q'].copy()
    device_audit = None
    if u is not None:
        if resource_case is None:
            raise ValueError('A resource case is required for control delivery')
        actual = u if response is None else np.asarray(u)*np.asarray(response)[None]
        device_audit = resource_case.audit(actual)
        if not device_audit['passed']:
            raise ArithmeticError('Delivered resource action violates its original domain')
        dp, dq = resource_case.aggregate(resource_case.delivery(actual))
        p += dp/background['network']['base_kw']
        q += dq/background['network']['base_kw']
    v, physical_audit = linear_voltages(background['network'], background['labels'], p, q, background['root'])
    observed = measured_channels(background, p, q, v)
    audit = dict(physical_model='physical_branch_LinDistFlow', linear=physical_audit, device=device_audit,
        voltage_cells_outside_screen=int(((v < .95) | (v > 1.05)).sum()),
        minimum_voltage_pu=float(v.min()), maximum_voltage_pu=float(v.max()))
    return observed, audit, (p, q, v)
