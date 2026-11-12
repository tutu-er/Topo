"""Hierarchy labels for topology-evaluation levels."""

from __future__ import annotations


TERMINAL_EQUIVALENT = "terminal_equivalent_topology"
PHYSICAL_HIDDEN = "physical_hidden_node_topology"


def topology_level_description(level: str) -> str:
    """Describe the requested topology-evaluation level."""

    if level == TERMINAL_EQUIVALENT:
        return "observed-terminal equivalent tree; not the full physical feeder"
    if level == PHYSICAL_HIDDEN:
        return "physical hidden-node tree, valid only with latent reconstruction or candidate hidden graph"
    raise ValueError(f"unknown topology level: {level}")

