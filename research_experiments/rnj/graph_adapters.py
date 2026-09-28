"""Research adapters for distance inputs and fixed-clade topology studies.

The main RX75-RNJ pipeline consumes shared paths directly. These adapters are
kept outside the installable core for historical comparisons and fixed-tree fits.
"""

from __future__ import annotations

import numpy as np

from rnj_wzzt.graph.rooted_neighbor_joining import RootedTreeResult


def shared_paths_from_distances(distance: np.ndarray, root_depths: np.ndarray) -> np.ndarray:
    """Convert a generic additive distance to root-to-LCA shared-path scores.

    Reduced R/X matrices should use ``sensitivity_geometry`` instead, because
    their off-diagonal entries are already these scores.
    """

    matrix = np.asarray(distance, dtype=float)
    depths = np.asarray(root_depths, dtype=float)
    if matrix.shape != (len(depths), len(depths)):
        raise ValueError("distance shape must match root_depths")
    shared = 0.5 * (depths[:, None] + depths[None, :] - matrix)
    shared = 0.5 * (shared + shared.T)
    shared = np.maximum(shared, 0.0)
    shared = np.minimum(shared, np.minimum(depths[:, None], depths[None, :]))
    np.fill_diagonal(shared, depths)
    return shared



def rooted_tree_from_clades(
    clades: set[frozenset[int]] | frozenset[frozenset[int]],
    terminals: list[int] | tuple[int, ...],
    root: int,
    include_hidden_root_stem: bool = False,
) -> RootedTreeResult:
    """Build the unique reduced rooted topology represented by laminar clades.

    Nontrivial rooted clades determine a reduced tree up to hidden-node labels
    and positive edge lengths. ``include_hidden_root_stem`` preserves a common
    root-to-all-terminals edge, which is intentionally absent from nontrivial
    clade metrics. Returned unit lengths are placeholders; fixed-tree R/X
    fitting estimates physical edge coefficients later.
    """

    terminal_tuple = tuple(int(node) for node in terminals)
    root = int(root)
    if len(set(terminal_tuple)) != len(terminal_tuple):
        raise ValueError("terminals must contain unique labels")
    if root in terminal_tuple:
        raise ValueError("root must not be a terminal")
    terminal_set = frozenset(terminal_tuple)
    normalized = {
        frozenset(int(node) for node in clade)
        for clade in clades
    }
    if any(not clade <= terminal_set for clade in normalized):
        raise ValueError("clades must contain only supplied terminals")
    normalized = {clade for clade in normalized if 1 < len(clade) < len(terminal_tuple)}
    ordered = sorted(normalized, key=lambda item: (len(item), tuple(sorted(item))))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left & right and not (left <= right or right <= left):
                raise ValueError("rooted clades must be laminar")

    first_hidden = min(-1, min((root, *terminal_tuple)) - 1)
    hidden_for_clade = {clade: first_hidden - index for index, clade in enumerate(ordered)}
    stem = first_hidden - len(ordered) if include_hidden_root_stem else root
    containers = [*ordered, terminal_set]
    edges: list[tuple[int, int, float]] = []
    if include_hidden_root_stem:
        edges.append((int(root), stem, 1.0))
    for clade in ordered:
        parent_clade = min(
            (candidate for candidate in containers if clade < candidate),
            key=lambda item: (len(item), tuple(sorted(item))),
        )
        parent = stem if parent_clade == terminal_set else hidden_for_clade[parent_clade]
        edges.append((int(parent), hidden_for_clade[clade], 1.0))
    for terminal in terminal_tuple:
        containing = [clade for clade in ordered if terminal in clade]
        parent = hidden_for_clade[min(containing, key=len)] if containing else stem
        edges.append((int(parent), terminal, 1.0))
    return RootedTreeResult(
        root=int(root),
        terminals=terminal_tuple,
        edges=tuple(edges),
        group_tolerance=0.0,
    )

