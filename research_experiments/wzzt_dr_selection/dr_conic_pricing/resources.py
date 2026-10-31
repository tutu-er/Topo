"""Translate existing declared resource-day profiles into conic-price inputs.

This adapter does not construct a topology, estimate customer behaviour, run
power flow, or add resource baselines to the network a second time. Each column
is one separately metered resource-day contract. Alternative types for a
resource are mutually exclusive; actual amplitudes remain continuous.
"""
from __future__ import annotations

import numpy as np

from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
    DT, SLOTS, DayModule, ResourceDay,
)


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _labels(node_labels):
    labels = tuple(node_labels)
    if (not labels or any(isinstance(label, (bool, np.bool_)) or
                          not isinstance(label, (int, np.integer)) for label in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError("node_labels must be unique integers in explicit network order")
    return tuple(int(label) for label in labels)


def resource_contract_data(resources, modules, *, node_labels, elasticities,
                           price_upper, response_factors, separate_types=True):
    """Build physical profiles and robust amplitude-domain device inequalities.

    Return a partial :class:`PriceProblem` dictionary. Complete it with the
    background/benefit/risk/budget inputs before constructing PriceProblem.
    ``profiles`` has shape ``(2*96*n_nodes, n_contracts)``: the first block is
    P[kW], the second Q[kvar], each flattened time-major in ``node_labels`` order.
    Multiplication by an amplitude in kW produces physical increments.

    A module spanning multiple resources is intentionally rejected: such a
    column would silently impose one random amplitude on several users. Build
    independent single-resource columns instead and declare their correlations
    in ``response_factors``. Their spatial sum is unconstrained.

    ``separate_types=False`` is accepted only when every resource has one type.
    It cannot turn mutually cancelling profiles into separately billed contracts.
    """
    if not isinstance(separate_types, (bool, np.bool_)):
        raise ValueError("separate_types must be boolean")
    resources, modules = list(resources), list(modules)
    if (not resources or any(not isinstance(r, ResourceDay) for r in resources) or
            len({r.resource_id for r in resources}) != len(resources)):
        raise ValueError("resources must be unique existing ResourceDay declarations")
    if (not modules or any(not isinstance(m, DayModule) for m in modules) or
            len({m.module_id for m in modules}) != len(modules)):
        raise ValueError("modules must be unique existing DayModule declarations")
    labels = _labels(node_labels)
    node_index = {node: index for index, node in enumerate(labels)}
    by_resource = {r.resource_id: r for r in resources}
    if any(r.node_label not in node_index for r in resources):
        raise ValueError("resource node is absent from node_labels")
    count = len(modules)
    elasticity = _finite_array(elasticities, (count,), "elasticities")
    prices = _finite_array(price_upper, (count,), "price_upper")
    factors = np.asarray(response_factors, dtype=float)
    if factors.ndim != 2 or factors.shape[0] != count or not np.isfinite(factors).all():
        raise ValueError("response_factors must be a finite [contracts,factors] matrix")
    if (elasticity <= 0).any() or (prices < 0).any():
        raise ValueError("elasticities must be positive and price_upper nonnegative")
    if (np.abs(factors).sum(axis=1) > 1 + 1e-12).any():
        raise ValueError("response_factors row L1 norm cannot exceed one")

    columns_by_resource = {r.resource_id: [] for r in resources}
    shape_by_resource = {r.resource_id: np.zeros((SLOTS, count)) for r in resources}
    profile_p = np.zeros((SLOTS, len(labels), count))
    profile_q = np.zeros_like(profile_p)
    reduction_hours, caps = [], []
    module_resources = []
    for j, module in enumerate(modules):
        if len(module.profiles) != 1:
            raise ValueError("each price contract must refer to one separately metered resource")
        resource_id, profile = next(iter(module.profiles.items()))
        if resource_id not in by_resource:
            raise ValueError(f"module refers to undeclared resource: {resource_id}")
        resource = by_resource[resource_id]
        profile = np.asarray(profile, dtype=float)
        scale = max(1., float(DT * np.abs(profile).sum()))
        if abs(DT * profile.sum()) > 1e-10 * scale:
            raise ValueError("module violates zero incremental daily energy")
        if np.max(np.abs(profile[~resource.available]), initial=0.) > 1e-10:
            raise ValueError("module changes power outside declared availability")
        deadline = resource.service_deadline_slot
        if deadline is not None and abs(DT * profile[:deadline].sum()) > 1e-10 * scale:
            raise ValueError("module violates zero incremental energy at the service deadline")
        reduction = float(DT * np.maximum(-profile, 0.).sum())
        if reduction <= 0:
            raise ValueError("a paid shift contract needs a nonzero reduction/rebound profile")
        columns_by_resource[resource_id].append(j)
        shape_by_resource[resource_id][:, j] = profile
        node = node_index[int(resource.node_label)]
        profile_p[:, node, j] = profile
        profile_q[:, node, j] = resource.reactive_ratio * profile
        reduction_hours.append(reduction)
        caps.append(float(module.amplitude_upper_kw))
        module_resources.append(resource_id)

    groups = [indices for indices in columns_by_resource.values() if len(indices) > 1]
    if groups and not separate_types:
        raise ValueError("multiple module types on one resource require exclusive type selection")
    rows, rhs, descriptions = [], [], []
    for resource in resources:
        matrix = shape_by_resource[resource.resource_id]
        if not columns_by_resource[resource.resource_id]:
            continue
        prefix = DT * np.cumsum(matrix, axis=0)
        lower_energy, upper_energy = resource.energy_bounds()
        for kind, coefficients, bound in (
                ("power_upper", matrix, resource.power_upper_kw - resource.baseline_kw),
                ("power_lower", -matrix, resource.baseline_kw - resource.power_lower_kw),
                ("prefix_upper", prefix, upper_energy),
                ("prefix_lower", -prefix, -lower_energy)):
            for slot in range(SLOTS):
                rows.append(coefficients[slot].copy())
                rhs.append(float(bound[slot]))
                descriptions.append(f"{resource.resource_id}:{kind}:{slot}")
        # Modules are checked above, while these rows also preserve the physical
        # zero-increment requirement in the returned linear device domain.
        for slot in np.flatnonzero(~resource.available):
            for sign in (1., -1.):
                rows.append(sign * matrix[slot])
                rhs.append(0.)
                descriptions.append(f"{resource.resource_id}:unavailable:{slot}:{sign:+g}")

    H = np.vstack((profile_p.reshape(SLOTS * len(labels), count),
                   profile_q.reshape(SLOTS * len(labels), count)))
    return dict(
        profiles=H.tolist(),
        response_factors=factors.tolist(),
        elasticities=elasticity.tolist(),
        reduction_hours=reduction_hours,
        price_upper=prices.tolist(),
        actual_amplitude_upper=caps,
        device_matrix=np.asarray(rows, float).reshape(-1, count).tolist(),
        device_rhs=rhs,
        contract_ids=[module.module_id for module in modules],
        exclusive_groups=groups,
        metadata=dict(
            resource_adapter="existing ResourceDay/DayModule v1",
            node_labels=list(labels), slots=SLOTS, interval_hours=DT,
            stacking="P_then_Q_time_major",
            power_layout="P then Q; each time-major [slot,node] C order",
            physical_units="P kW, Q kvar, amplitude kW, energy kWh",
            contract_resources=module_resources,
            device_row_labels=descriptions,
            resource_declarations={r.resource_id: r.declaration_source for r in resources},
            equality_policy="each module preserves daily/deadline energy; no spatial balance imposed",
            settlement="one separately metered whole-day contract per selected resource type",
            network_feasibility="requires downstream AC execution; not certified by this adapter",
            no_clipping=True,
        ),
    )


def translate_net_load(net_load, node_labels, n_slots=SLOTS):
    """Unstack a P/Q vector into physical matrices, without unit conversion."""
    labels = _labels(node_labels)
    if isinstance(n_slots, bool) or not isinstance(n_slots, (int, np.integer)) or n_slots <= 0:
        raise ValueError("n_slots must be a positive integer")
    vector = _finite_array(net_load, (2 * int(n_slots) * len(labels),), "net_load")
    split = int(n_slots) * len(labels)
    return dict(node_labels=list(labels),
                p_kw=vector[:split].reshape(int(n_slots), len(labels)).tolist(),
                q_kvar=vector[split:].reshape(int(n_slots), len(labels)).tolist(),
                power_layout="time-major [slot,node]; physical kW/kvar")


def increments_from_amplitudes(contract_data, amplitudes_kw):
    """Produce the existing node_increments payload; contains increments only."""
    profiles = np.asarray(contract_data["profiles"], dtype=float)
    if profiles.ndim != 2 or not np.isfinite(profiles).all():
        raise ValueError("contract profiles must be a finite matrix")
    amplitudes = _finite_array(amplitudes_kw, (profiles.shape[1],), "amplitudes_kw")
    if (amplitudes < 0).any():
        raise ValueError("actual contract amplitudes cannot be negative")
    metadata = contract_data["metadata"]
    labels, slots = metadata["node_labels"], metadata["slots"]
    decoded = translate_net_load(profiles @ amplitudes, labels, slots)
    p, q = np.asarray(decoded["p_kw"]), np.asarray(decoded["q_kvar"])
    return dict(
        node_increments={str(label): dict(delta_kw=p[:, i].tolist(), delta_kvar=q[:, i].tolist())
                         for i, label in enumerate(labels)},
        increment_only_payload=True,
        no_clipping=True,
        terminal_payload_semantics="deltas only; add once to the network baseline",
        network_feasibility="not evaluated; requires AC execution",
    )
