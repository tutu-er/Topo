"""Network data structures for terminalized case33 feeders."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass
class TerminalizedNetwork:
    """Terminal-load-only feeder with hidden internal nodes and observed leaves."""

    buses: pd.DataFrame
    branches: pd.DataFrame
    root_bus: int
    base_kv: float
    base_mva: float
    original_to_terminal: dict[int, int]
    terminal_to_original: dict[int, int]
    metadata: dict

    def observed_buses(self) -> list[int]:
        """Return observed buses, including the root."""

        return self.buses.loc[self.buses["is_observed"], "bus_id"].astype(int).tolist()

    def hidden_buses(self) -> list[int]:
        """Return hidden internal buses."""

        return self.buses.loc[self.buses["bus_type"].eq("hidden_internal"), "bus_id"].astype(int).tolist()

    def load_buses(self) -> list[int]:
        """Return buses with load."""

        return self.buses.loc[self.buses["has_load"], "bus_id"].astype(int).tolist()

    def closed_edges(self) -> list[tuple[int, int]]:
        """Return true closed physical edges."""

        rows = self.branches[self.branches["is_true_closed"]]
        return [(int(r.from_bus), int(r.to_bus)) for r in rows.itertuples()]

    def to_networkx_graph(self) -> nx.Graph:
        """Build a NetworkX graph from closed edges."""

        graph = nx.Graph()
        for row in self.buses.itertuples():
            graph.add_node(int(row.bus_id), **row._asdict())
        for row in self.branches[self.branches["is_true_closed"]].itertuples():
            graph.add_edge(
                int(row.from_bus),
                int(row.to_bus),
                r_ohm=float(row.r_ohm),
                x_ohm=float(row.x_ohm),
                branch_type=row.branch_type,
            )
        return graph

    def orient_from_root(self) -> pd.DataFrame:
        """Return closed branches oriented away from the root."""

        graph = self.to_networkx_graph()
        parent = dict(nx.bfs_predecessors(graph, self.root_bus))
        records = []
        for child, par in parent.items():
            edge = graph.edges[par, child]
            records.append({"parent": int(par), "child": int(child), **edge})
        return pd.DataFrame(records)

    def get_leaf_buses(self) -> list[int]:
        """Return graph leaves excluding the root unless it is the only node."""

        graph = self.to_networkx_graph()
        leaves = [int(node) for node, degree in graph.degree() if degree == 1]
        if graph.number_of_nodes() > 1:
            leaves = [node for node in leaves if node != self.root_bus]
        return sorted(leaves)

    def check_terminal_load_only(self, strict: bool = True) -> dict:
        """Validate the terminal-load-only property."""

        from rnj_wzzt.scenario.validate_scenario import validate_terminal_load_only_scenario

        return validate_terminal_load_only_scenario(self, strict=strict)
