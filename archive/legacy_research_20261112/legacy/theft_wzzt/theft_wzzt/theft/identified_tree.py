"""Identified-tree interface: run the mainline on CLEAN data, expose the result
as a TheftTree with per-edge weight estimates for the theft stage.

This replaces oracle (truth-derived) weight bounds by data-derived ones, which
is the realistic chain: daily-demeaned mainline fit -> identified reduced tree
-> theft MILP refits within tolerance bands around the fitted edge weights.

The reduced tree's nodes are terminal leaves plus clade nodes; a clade node is a
REGION: true buses whose downstream terminal set maps into that clade are
indistinguishable from each other at terminal-only resolution (degree-2 chains
are contracted by construction). ``region_label`` performs that mapping for
evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from theft_wzzt import pipeline as _pipeline
from theft_wzzt.estimation.laminar_l1_milp import fit_laminar_l1_sensitivity
from theft_wzzt.theft.theft_model import TheftTree

ROOT_LABEL = -1


@dataclass(frozen=True)
class IdentifiedTree:
    """Mainline output: terminal-level laminar supports with fitted weights."""

    case_key: str
    terminals: tuple[int, ...]
    supports: tuple[frozenset[int], ...]  # bus-id sets, singletons included
    r_values: tuple[float, ...]
    x_values: tuple[float, ...]

    def label_of(self, support: frozenset[int]) -> int:
        if len(support) == 1:
            return next(iter(support))
        return -(self.supports.index(support) + 2)  # synthetic clade node label


def fit_identified_tree(case_key: str, *, time_limit: float = 1800.0) -> IdentifiedTree:
    """Run the RX75-RNJ + wzzT-MILP mainline (non-contracted) on clean data."""

    scenario_options = dict(scenario_suite="reference", root_observation=None,
                            root_meter_noise_rel=None, root_sigma=None,
                            impedance_scale=None)
    case, _ = _pipeline._prepare_case(
        case_key, samples_per_scenario=96, scenario_count=3,
        training_replicate=0, validation_replicate=1,
        pq_noise_rel=0.005, voltage_noise_rel=0.0002,
        scenario_options=scenario_options, root_observation="noisy",
    )
    selection = _pipeline._select_case_rnj(
        case, bootstrap_replicates=100, block_length=4,
        confidence_threshold=0.75, maximum_candidate_count=2,
        tolerance_factor=0.16, seed=20_260_903,
    )
    position = {label: index for index, label in enumerate(case.terminals)}
    selected_supports = [
        tuple(position[label] for label in sorted(clade)) for clade in selection.selected
    ]
    initial = [*_pipeline._leaf_singletons(len(case.terminals)), *selected_supports]
    pseudo = {label: frozenset({label}) for label in case.terminals}
    candidate_supports = _pipeline._rnj_reduced_candidate_pool(
        selection.full_clades, list(case.terminals), pseudo, include_one_edit=False,
    )
    result = fit_laminar_l1_sensitivity(
        case.training, validation_scenarios=case.validation,
        initial_supports=initial, candidate_supports=candidate_supports,
        r_upper_bound=2.0, x_upper_bound=2.0, time_limit=time_limit,
    )
    terminals = tuple(int(label) for label in result.terminal_labels)
    supports = tuple(
        frozenset(int(label) for label in labels) for labels in result.support_labels
    )
    return IdentifiedTree(case_key, terminals, supports,
                          tuple(float(v) for v in result.r_values),
                          tuple(float(v) for v in result.x_values))


def save_identified(tree: IdentifiedTree, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "case_key": tree.case_key, "terminals": list(tree.terminals),
        "supports": [sorted(s) for s in tree.supports],
        "r_values": list(tree.r_values), "x_values": list(tree.x_values),
    }, indent=1), encoding="utf-8")


def load_identified(path: str | Path) -> IdentifiedTree:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return IdentifiedTree(raw["case_key"], tuple(raw["terminals"]),
                          tuple(frozenset(s) for s in raw["supports"]),
                          tuple(raw["r_values"]), tuple(raw["x_values"]))


def to_theft_tree(tree: IdentifiedTree) -> tuple[TheftTree, tuple[np.ndarray, np.ndarray]]:
    """Edges follow the laminar containment hierarchy; weight = atom value."""

    supports = sorted(tree.supports, key=len)
    edges, weights_r, weights_x = [], [], []
    for support in supports:
        parents = [t for t in supports if support < t]
        parent = min(parents, key=len) if parents else None
        child_label = tree.label_of(support)
        parent_label = ROOT_LABEL if parent is None else tree.label_of(parent)
        edges.append((parent_label, child_label))
        index = tree.supports.index(support)
        weights_r.append(tree.r_values[index])
        weights_x.append(tree.x_values[index])
    clade_labels = tuple(tree.label_of(s) for s in tree.supports if len(s) > 1)
    theft_tree = TheftTree(tuple(edges), tree.terminals, clade_labels + tree.terminals)
    theft_tree_weights = (np.array(weights_r), np.array(weights_x))
    return theft_tree, theft_tree_weights


def edge_weight_bounds(weights, factor=(0.5, 2.0), floor: float = 1e-6):
    """Per-edge bounds around fitted weights, in the TheftTree edge order."""

    r, x = weights
    lo = factor[0] * np.minimum(r, x)
    hi = np.maximum(factor[1] * np.maximum(r, x), floor)
    return lo, hi


def region_label(bus: int, true_net, tree: IdentifiedTree) -> int:
    """Map a TRUE-network bus to its identified-tree region label.

    Region of a bus = smallest identified clade containing all terminals
    downstream of the bus; terminals map to themselves; buses with no observed
    descendant (unobserved spurs) inherit the region of their nearest ancestor
    that has one. Buses above every clade map to ROOT_LABEL.
    """

    graph = true_net.to_networkx_graph()
    terminal_set = set(tree.terminals)
    parent = dict(nx.bfs_predecessors(graph, true_net.root_bus))

    node = int(bus)
    children: dict[int, list[int]] = {}
    for child, par in parent.items():
        children.setdefault(par, []).append(child)
    while True:
        subtree = {node}
        stack = [node]
        while stack:
            current = stack.pop()
            for child in children.get(current, []):
                subtree.add(child)
                stack.append(child)
        observed_downstream = subtree & terminal_set
        if observed_downstream or node == int(true_net.root_bus):
            break
        node = parent[node]  # unobserved spur: inherit the ancestor's region
    if not observed_downstream:
        return ROOT_LABEL
    containing = [s for s in tree.supports if observed_downstream <= s]
    if not containing:
        return ROOT_LABEL
    return tree.label_of(min(containing, key=len))
