"""Rooted clade helpers for terminal-only RNJ topology."""

from __future__ import annotations

import networkx as nx


def rooted_clades(
    edges: list[tuple] | tuple[tuple, ...],
    root: int,
    terminals: list[int] | tuple[int, ...],
) -> set[frozenset[int]]:
    """Return nontrivial downstream terminal clades of a rooted tree."""

    graph = nx.Graph()
    graph.add_edges_from((int(edge[0]), int(edge[1])) for edge in edges)
    terminal_set = {int(node) for node in terminals}
    parent = {
        int(child): int(upstream) for child, upstream in nx.bfs_predecessors(graph, int(root))
    }
    children: dict[int, list[int]] = {int(node): [] for node in graph.nodes()}
    for child, upstream in parent.items():
        children[upstream].append(child)
    clades = set()

    def visit(node: int) -> set[int]:
        found = {node} if node in terminal_set else set()
        for child in children[node]:
            found.update(visit(child))
        if 1 < len(found) < len(terminal_set):
            clades.add(frozenset(found))
        return found

    visit(int(root))
    return clades
