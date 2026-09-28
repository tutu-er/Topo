"""Compatibility exports for rooted clades and historical aggregation research."""

from __future__ import annotations

import sys
from pathlib import Path

_repository_root = Path(__file__).resolve().parents[2]
_core_root = _repository_root / "rnj_wzzt_core"
for _path in (_repository_root, _core_root):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from research_experiments.rnj.graph_adapters import rooted_tree_from_clades
from research_experiments.rnj.rooted_aggregation import (
    PseudoCluster,
    aggregate_rooted_scenarios,
    build_local_cluster_scenarios,
    expand_pseudo_clades,
    expand_pseudo_clades_with_local_reidentification,
    expand_pseudo_sibling_pairs,
    reidentify_local_clades_from_shared_paths,
    select_peripheral_clusters,
)

__all__ = [
    "PseudoCluster",
    "aggregate_rooted_scenarios",
    "build_local_cluster_scenarios",
    "expand_pseudo_clades",
    "expand_pseudo_clades_with_local_reidentification",
    "expand_pseudo_sibling_pairs",
    "reidentify_local_clades_from_shared_paths",
    "rooted_clades",
    "rooted_tree_from_clades",
    "select_peripheral_clusters",
]
