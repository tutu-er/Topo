"""Observed-data AC diagnostics for no-Source WZZT atom families.

AC convergence is a physical computation check, not a topology certificate.
All rows remain fitting rows; blocked one-SE is a descriptive selection rule.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def atom_network(labels, supports, r_values, x_values):
    from rnj_wzzt.models.network import TerminalizedNetwork
    labels = [int(x) for x in labels]
    family = [frozenset(map(int, s)) for s in supports]
    if len(set(family)) != len(family) or any(not s or not s <= set(range(len(labels))) for s in family):
        raise ValueError("invalid atom family")
    if not all(frozenset((i,)) in family for i in range(len(labels))):
        raise ValueError("all singleton leaves are required")
    for left in family:
        for right in family:
            if left & right and not (left <= right or right <= left):
                raise ValueError("nonlaminar atom family")
    root = min([0, *labels])-1
    next_hidden = root-1
    nodes = {}
    for support in family:
        if len(support) == 1:
            nodes[support] = labels[next(iter(support))]
        else:
            nodes[support] = next_hidden
            next_hidden -= 1
    buses = [{"bus_id": b, "is_observed": b in labels or b == root,
              "bus_type": "terminal" if b in labels else "hidden_internal", "has_load": b in labels}
             for b in [root, *nodes.values()]]
    branches = []
    for index, (support, r, x) in enumerate(zip(family, r_values, x_values, strict=True)):
        parents = [s for s in family if support < s]
        parent = nodes[min(parents, key=len)] if parents else root
        # Squared-voltage atom coefficients include factor 2. Using base Z=1
        # gives an equivalent normalized AC network without reading true bases.
        branches.append(dict(from_bus=parent, to_bus=nodes[support], r_ohm=float(r)/2,
            x_ohm=float(x)/2, branch_type="estimated_atom", is_true_closed=True, atom_index=index))
    return TerminalizedNetwork(pd.DataFrame(buses), pd.DataFrame(branches), root, 1., 1., {}, {},
        {"scope": "normalized no-Source AC candidate; impedance=atom_coefficient/2"})


def ac_score(labels, supports, r_values, x_values, observed, *, refit=False, max_nfev=12):
    from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
    started = perf_counter()
    net = atom_network(labels, supports, r_values, x_values)
    frame_p, frame_q, voltage, root = observed
    if list(frame_p.columns) != list(labels) or not frame_p.index.equals(voltage.index):
        raise ValueError("AC observed inputs must be aligned")
    initial = np.r_[np.asarray(r_values)/2, np.asarray(x_values)/2]
    count = len(supports)

    def calculate(parameters):
        net.branches.loc[:, "r_ohm"] = parameters[:count]
        net.branches.loc[:, "x_ohm"] = parameters[count:]
        result = solve_ac_power_flow_timeseries(net, frame_p, frame_q, v_root=root,
                                               max_iter=200, tol=1e-11)
        if not result["converged"].all():
            raise RuntimeError("candidate AC did not converge")
        predicted = result["V_bus_mag"].loc[:, labels].to_numpy()
        residual = (predicted-voltage.to_numpy())/voltage.to_numpy()
        return residual, result

    try:
        initial_residual, _ = calculate(initial)
        fit_info = {"enabled": refit, "success": None, "nfev": 0}
        parameters = initial
        if refit:
            fit = least_squares(lambda value: calculate(value)[0].ravel(),
                np.clip(initial, 1e-12, 1.-1e-12), bounds=(0., 1.),
                max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-9, x_scale="jac")
            parameters = fit.x
            fit_info.update(success=bool(fit.success), status=int(fit.status), nfev=int(fit.nfev),
                optimality=float(fit.optimality), message=str(fit.message),
                scope="local shared-impedance fit; measured PQ/root held fixed; all 96 rows")
        residual, ac = calculate(parameters)
        by_time = np.mean(residual**2, axis=1)
        blocks = [float(np.mean(b)) for b in np.array_split(by_time, min(4, len(by_time)))]
        head = ac["branch_p_from_pu"].to_numpy()[:, [i for i, e in enumerate(ac["metadata"]["branch_edges"]) if e["from_bus"] == net.root_bus]].sum(axis=1)
        balance = head-frame_p.sum(axis=1).to_numpy()-ac["branch_loss_p_pu"].sum(axis=1).to_numpy()
        return {"valid": True, "ac_converged_points": len(by_time),
            "relative_voltage_rmse": float(np.sqrt(np.mean(by_time))),
            "initial_relative_voltage_rmse": float(np.sqrt(np.mean(initial_residual**2))),
            "relative_voltage_mse": float(np.mean(by_time)), "block_mse": blocks,
            "block_mse_se": float(np.std(blocks, ddof=1)/np.sqrt(len(blocks))) if len(blocks)>1 else 0.,
            "max_active_balance_error": float(np.max(abs(balance))),
            "refit": fit_info, "r_refit": (parameters[:count]*2).tolist(),
            "x_refit": (parameters[count:]*2).tolist(), "seconds": perf_counter()-started,
            "scope": "observed-data training diagnostic; no independent validation or topology probability"}
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}", "seconds": perf_counter()-started}


def select_ac_one_se(points):
    valid = [i for i, p in enumerate(points) if p["ac"]["valid"]]
    if not valid:
        return None
    best = min(valid, key=lambda i: points[i]["ac"]["relative_voltage_mse"])
    threshold = points[best]["ac"]["relative_voltage_mse"]+points[best]["ac"]["block_mse_se"]
    eligible = [i for i in valid if points[i]["ac"]["relative_voltage_mse"] <= threshold]
    return min(eligible, key=lambda i: (len(points[i]["supports"]), i))
