"""Verbatim execution/check/scene functions; portable import glue only."""
import numpy as np
import pandas as pd
import networkx as nx
from rnj_wzzt.models.lin_distflow import ohm_to_pu
from rnj_wzzt.scenario.simulation import _terminal_buses

def scene(p, q, v, root, labels, name, day_offset):
    """Match the current observed-only adapter without importing physical code.

    Dates are synthetic ordering labels, not a claim about actual calendar data.
    Days 0..2 are old training; 3..5 old validation; 6..9 new actual events.
    """
    p, q, v, root = (np.asarray(x, dtype=float) for x in (p, q, v, root))
    if p.shape != (96, len(labels)) or q.shape != p.shape or v.shape != p.shape or root.shape != (96,):
        raise ValueError('one full day of 96 aligned P/Q/V/root samples required')
    if not all(np.isfinite(x).all() for x in (p, q, v, root)):
        raise ValueError('nonfinite observation')
    if (v <= 0).any() or (root <= 0).any():
        raise ValueError('nonpositive measured voltage')
    index = pd.date_range(pd.Timestamp('2000-01-01')+pd.Timedelta(days=int(day_offset)),
                          periods=96, freq='15min')
    frames = [pd.DataFrame(a, index=index, columns=labels) for a in (p, q, v)]
    rv = pd.Series(root, index=index, name='V_root_measured')
    target = pd.DataFrame(root[:, None]**2-v**2, index=index, columns=labels)
    return dict(name=str(name), P_terminal=frames[0], Q_terminal=frames[1],
                V_terminal=frames[2], V_root_observed=rv, drop_target=target,
                units=dict(P_terminal='pu', Q_terminal='pu', V_terminal='pu',
                           V_root_observed='pu', drop_target='pu2'))

def validate_context_metadata(context, metadata, labels=None):
    """Post-gate producer checks; no profile is generated until these agree."""
    if context.net.root_bus != metadata['root_label']:
        raise ValueError('physical producer root differs from source metadata')
    if not np.isclose(context.net.base_kv, metadata['base_kv'], rtol=0., atol=1e-12):
        raise ValueError('physical producer voltage base differs from source metadata')
    if not np.isclose(context.net.base_mva, metadata['base_mva'], rtol=0., atol=1e-12):
        raise ValueError('physical producer power base differs from source metadata')
    if labels is not None and list(context.terminals) != list(labels):
        raise ValueError('frozen terminal order differs from physical producer')

def _physical_checks(net, p_pu, q_pu, ac, voltage_limits_pu, equation_tolerance_pu):
    """Independent nodal/branch power checks from returned physical quantities."""
    all_buses = net.buses.bus_id.astype(int).tolist()
    local_p = p_pu.reindex(columns=all_buses, fill_value=0.0)
    local_q = q_pu.reindex(columns=all_buses, fill_value=0.0)
    inflow_p = pd.DataFrame(0.0, index=p_pu.index, columns=all_buses)
    inflow_q = inflow_p.copy()
    outflow_p = inflow_p.copy()
    outflow_q = inflow_p.copy()
    graph = net.to_networkx_graph()
    branch_loss_error, distflow_error = 0.0, 0.0
    for edge in ac["metadata"]["branch_edges"]:
        label, u, v = edge["label"], edge["from_bus"], edge["to_bus"]
        inflow_p.loc[:, v] += ac["branch_p_to_pu"][label]
        inflow_q.loc[:, v] += ac["branch_q_to_pu"][label]
        outflow_p.loc[:, u] += ac["branch_p_from_pu"][label]
        outflow_q.loc[:, u] += ac["branch_q_from_pu"][label]
        r = ohm_to_pu(net, graph.edges[u, v]["r_ohm"])
        x = ohm_to_pu(net, graph.edges[u, v]["x_ohm"])
        ell = ac["I_branch_mag"][label].to_numpy() ** 2
        branch_loss_error = max(branch_loss_error,
            float(np.max(np.abs(ac["branch_loss_p_pu"][label].to_numpy() - r * ell))),
            float(np.max(np.abs(ac["branch_loss_q_pu"][label].to_numpy() - x * ell))))
        d = (ac["V_bus_mag"][u].to_numpy()**2 - ac["V_bus_mag"][v].to_numpy()**2
             - 2 * (r * ac["branch_p_from_pu"][label].to_numpy() + x * ac["branch_q_from_pu"][label].to_numpy())
             + (r*r + x*x) * ell)
        distflow_error = max(distflow_error, float(np.max(np.abs(d))))
    nonroot = [x for x in all_buses if x != net.root_bus]
    nodal_p_error = float(np.max(np.abs((inflow_p - local_p - outflow_p)[nonroot].to_numpy())))
    nodal_q_error = float(np.max(np.abs((inflow_q - local_q - outflow_q)[nonroot].to_numpy())))
    slack_p = outflow_p[net.root_bus] + local_p[net.root_bus]
    slack_q = outflow_q[net.root_bus] + local_q[net.root_bus]
    balance_p = slack_p - local_p.sum(axis=1) - ac["branch_loss_p_pu"].sum(axis=1)
    balance_q = slack_q - local_q.sum(axis=1) - ac["branch_loss_q_pu"].sum(axis=1)
    v = ac["V_bus_mag"].to_numpy()
    lo, hi = voltage_limits_pu
    convergence = ac["converged"].to_numpy(dtype=bool)
    physical_finite = all(np.isfinite(ac[key].to_numpy()).all() for key in (
        "V_bus_mag", "V_bus_angle_deg", "I_branch_mag", "branch_p_from_pu",
        "branch_q_from_pu", "branch_p_to_pu", "branch_q_to_pu",
        "branch_loss_p_pu", "branch_loss_q_pu", "max_voltage_error"))
    residual = max(nodal_p_error, nodal_q_error, branch_loss_error,
                   float(np.max(np.abs(balance_p))), float(np.max(np.abs(balance_q))))
    return {
        "sample_count": len(p_pu), "converged_samples": int(convergence.sum()),
        "nonconverged_samples": int((~convergence).sum()),
        "physical_outputs_finite": bool(physical_finite),
        "max_nodal_p_residual_pu": nodal_p_error, "max_nodal_q_residual_pu": nodal_q_error,
        "max_system_p_balance_residual_pu": float(np.max(np.abs(balance_p))),
        "max_system_q_balance_residual_pu": float(np.max(np.abs(balance_q))),
        "max_branch_loss_formula_error_pu": branch_loss_error,
        "max_distflow_identity_error_pu2": distflow_error,
        "equation_tolerance_pu": equation_tolerance_pu,
        "electrical_equations_pass": bool(physical_finite and residual <= equation_tolerance_pu and distflow_error <= equation_tolerance_pu),
        "physical_voltage_min_pu": float(v.min()), "physical_voltage_max_pu": float(v.max()),
        "voltage_limits_pu": [lo, hi],
        "voltage_violation_cells": int(((v < lo) | (v > hi)).sum()),
        "voltage_limits_pass": bool(((v >= lo) & (v <= hi)).all()),
        "voltage_limit_meaning": "declared scenario screening range, not a regulatory certificate",
        "thermal_limits_checked": False,
        "thermal_reason": "paper-style builder has no branch ampacity ratings",
        "max_iterations": int(ac["iterations"].max()),
        "max_solver_voltage_update_pu": float(ac["max_voltage_error"].max()),
        "no_sample_dropped": True,
    }

def effective_truth_supports(net) -> list[list[int]]:
    """Evaluator-only descendant supports of physical nonzero-impedance lines.

    Shared descendants are deduplicated; zero-impedance edges have no effect in
    the fitted R/X model. This must never supply a planning candidate target.
    """
    graph = net.to_networkx_graph()
    tree = nx.bfs_tree(graph, net.root_bus)
    terminals = _terminal_buses(net)
    supports = set()
    for parent, child in tree.edges:
        edge = graph.edges[parent, child]
        if edge['r_ohm'] == 0 and edge['x_ohm'] == 0:
            continue
        descendants = nx.descendants(tree, child) | {child}
        support = tuple(node for node in terminals if node in descendants)
        if support:
            supports.add(support)
    return [list(s) for s in sorted(supports, key=lambda x: (len(x), x))]

def execute(runtime,bg,dev,u,multiplier,record):
    delivered=dev.project(u*multiplier);audit=dev.audit(delivered)
    if not audit['passed']:raise ArithmeticError('causal delivery violates device domain')
    dp,dq=dev.aggregate(delivered);base=runtime['metadata']['base_kw']
    pf=bg.p_base+dp/base;qf=bg.q_base+dq/base
    sim=runtime['sim'];net=runtime['context'].net;labels=runtime['labels']
    ac=sim.solve_ac_power_flow_timeseries(net,pf,qf,v_root=bg.root_true,max_iter=100,tol=1e-10)
    voltage=ac['V_bus_mag'].loc[:,labels]
    check=runtime['checker'](net,pf,qf,ac,(.9,1.1),1e-8)
    converged=int(np.asarray(ac['converged'],bool).sum())
    summary=dict(status='complete' if converged==96 and check['electrical_equations_pass'] else 'physical_unresolved',
        converged=converged,voltage_min=float(voltage.to_numpy().min()),voltage_max=float(voltage.to_numpy().max()),
        voltage_violation_cells=int(check['voltage_violation_cells']),thermal_limits_checked=False,resource_audit=audit,
        planned_cost=dev.fees(u),recourse_abs_kwh=float(.25*np.abs(delivered-u*multiplier).sum()),
        actual_resource_wear_proxy_fee=dev.fees(delivered),wear_proxy_not_extra_payment=True,
        max_increment_error_kw=float(max(np.max(np.abs((pf-bg.p_base).to_numpy()*base-dp)),np.max(np.abs((qf-bg.q_base).to_numpy()*base-dq)))))
    scene=None
    if summary['status']=='complete':
        measured=sim._observe_terminal_channels(runtime['context'],bg,pf,qf,voltage,pq_noise_rel=runtime['pq_sigma'],v_noise_rel=runtime['voltage_sigma'])
        scene=runtime['legacy'].scene(measured['p_measured'].to_numpy(),measured['q_measured'].to_numpy(),measured['voltage_measured'].to_numpy(),measured['root_observed'].to_numpy(),labels,record,6)
    return dict(summary=summary,scene=scene,private=dict(natural_p=bg.p_base,natural_q=bg.q_base,physical_p=pf,physical_q=qf,
        delivered=delivered,multiplier=multiplier,nominal=u,AC=ac,checks=check,node_delta_p=dp,node_delta_q=dq))
