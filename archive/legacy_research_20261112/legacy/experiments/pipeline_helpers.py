"""Higher-level helpers for aggregation and local re-root experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common import simulate_case, terminal_parent_map
from terminal_case33.estimation.preprocessing import squared_voltage_drop_from_observed_root
from terminal_case33.models.lin_distflow import ohm_to_pu
from terminal_case33.models.network import TerminalizedNetwork


def service_corrected_parent_voltage(
    net: TerminalizedNetwork,
    V_child: pd.DataFrame,
    P_child: pd.DataFrame,
    Q_child: pd.DataFrame,
    child_to_parent: dict[int, int],
) -> pd.DataFrame:
    """Estimate hidden-parent voltage by de-embedding service-line drop."""

    graph = net.to_networkx_graph()
    parent_voltage_sq: dict[int, list[pd.Series]] = {}
    for child, parent in child_to_parent.items():
        edge = graph.edges[int(parent), int(child)]
        r_pu = ohm_to_pu(net, float(edge["r_ohm"]))
        x_pu = ohm_to_pu(net, float(edge["x_ohm"]))
        v_parent_sq = V_child[child].pow(2) + 2.0 * (r_pu * P_child[child] + x_pu * Q_child[child])
        parent_voltage_sq.setdefault(parent, []).append(v_parent_sq)
    data = {
        parent: np.sqrt(np.maximum(pd.concat(series_list, axis=1).mean(axis=1), 1e-8))
        for parent, series_list in parent_voltage_sq.items()
    }
    return pd.DataFrame(data, index=V_child.index).sort_index(axis=1)


def aggregate_to_immediate_hidden_parents(
    net: TerminalizedNetwork,
    scenario: dict,
    voltage_mode: str,
) -> dict:
    """Aggregate terminal P/Q and proxy voltage to immediate hidden parents."""

    child_to_parent = terminal_parent_map(net)
    parent_ids = sorted(set(child_to_parent.values()))
    P_child = scenario["P_terminal"]
    Q_child = scenario["Q_terminal"]
    V_child = scenario["V_terminal"]
    P_parent = pd.DataFrame(index=P_child.index)
    Q_parent = pd.DataFrame(index=Q_child.index)
    for parent in parent_ids:
        children = [child for child, par in child_to_parent.items() if par == parent]
        P_parent[parent] = P_child[children].sum(axis=1)
        Q_parent[parent] = Q_child[children].sum(axis=1)
    if voltage_mode == "mean_terminal_voltage":
        V_parent = pd.DataFrame(
            {
                parent: V_child[[child for child, par in child_to_parent.items() if par == parent]].mean(axis=1)
                for parent in parent_ids
            },
            index=V_child.index,
        )
    elif voltage_mode == "service_corrected_oracle":
        V_parent = service_corrected_parent_voltage(net, V_child, P_child, Q_child, child_to_parent)
    else:
        raise ValueError(f"unknown voltage_mode: {voltage_mode}")
    return {
        "name": scenario["name"],
        "V_terminal": V_parent,
        "P_terminal": P_parent,
        "Q_terminal": Q_parent,
        "root_voltage": scenario["root_voltage"],
        "drop_target": squared_voltage_drop_from_observed_root(V_parent, scenario["root_voltage"]),
    }


def fit_and_score(net, nodes: list[int], scenarios: list[dict], recipe: dict, distance_mode: str, alpha: float) -> dict:
    """Run the main fast terminal-equivalent tree fit and score it."""

    from experiments.common import fit_terminal_equivalent_tree
    from experiments.run_large_sweep_filter_distance import _distance_candidates, _fast_projected_multiscenario_fit

    return fit_terminal_equivalent_tree(
        net,
        nodes,
        scenarios,
        recipe,
        distance_mode,
        alpha,
        fit_fn=_fast_projected_multiscenario_fit,
        distance_fn=_distance_candidates,
    )


_simulate_case = simulate_case
_fit_and_score = fit_and_score
_terminal_parent_map = terminal_parent_map
_aggregate_to_immediate_hidden_parents = aggregate_to_immediate_hidden_parents
