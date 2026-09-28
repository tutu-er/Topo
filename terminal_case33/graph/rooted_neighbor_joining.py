"""Compatibility exports for core RNJ and the research distance adapter."""

from terminal_case33._standalone_core import load_core_module

_core = load_core_module("rnj_wzzt.graph.rooted_neighbor_joining")
RootedTreeResult = _core.RootedTreeResult
WeightedEdge = _core.WeightedEdge
rooted_neighbor_joining = _core.rooted_neighbor_joining
_contract_zero_internal_edges = _core._contract_zero_internal_edges

from research_experiments.rnj.graph_adapters import shared_paths_from_distances

__all__ = [
    "RootedTreeResult", "WeightedEdge", "rooted_neighbor_joining",
    "shared_paths_from_distances",
]
