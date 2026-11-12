"""Plotting functions for scenario and sensitivity outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_network(net, path: str | Path, title: str = "network") -> None:
    """Plot the closed network graph."""

    graph = net.to_networkx_graph()
    pos = nx.spring_layout(graph, seed=1)
    colors = []
    bus_types = net.buses.set_index("bus_id")["bus_type"].to_dict()
    for node in graph.nodes:
        colors.append({"root": "#d62728", "observed_terminal": "#2ca02c"}.get(bus_types.get(node), "#8c8c8c"))
    plt.figure(figsize=(10, 7))
    nx.draw_networkx(graph, pos=pos, node_color=colors, node_size=110, font_size=6, width=0.8)
    plt.title(title)
    plt.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap(matrix, path: str | Path, title: str = "sensitivity") -> None:
    """Plot a matrix heatmap."""

    plt.figure(figsize=(7, 6))
    plt.imshow(np.asarray(matrix), cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_edge_marginal_hist(edge_marginals, path: str | Path) -> None:
    """Plot posterior edge marginal histogram."""

    plt.figure(figsize=(6, 4))
    vals = edge_marginals["edge_marginal"].to_numpy() if len(edge_marginals) else []
    plt.hist(vals, bins=20, color="#4c78a8", edgecolor="white")
    plt.xlabel("edge marginal")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

