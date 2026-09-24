"""Classical Q-criterion NJ with an observed root anchor for fair study controls.

This experiment-only implementation uses exactly the RX75 geometry supplied to
RNJ. The known root is added as an observed point with d(root,i)=S_ii. Internal
edges can be collapsed by a declared threshold; threshold selection belongs to
the validation split, never the true tree. This is a local NJ implementation,
not a reproduction of a complete distribution-grid method from the literature.
"""
from itertools import combinations
import numpy as np
from rnj_wzzt.graph.rooted_neighbor_joining import RootedTreeResult, _contract_zero_internal_edges
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry


def infer_classical_nj(r, x, terminals, root, *, collapse_factor=0.0):
    labels = [int(v) for v in terminals]
    root = int(root)
    if not labels or len(set(labels)) != len(labels) or root in labels:
        raise ValueError('unique nonempty terminal labels, disjoint from root, are required')
    if not np.isfinite(collapse_factor) or collapse_factor < 0:
        raise ValueError('collapse_factor must be finite and nonnegative')
    geo = sensitivity_geometry(r, x, 'RX_75R_25X')
    if geo.distance.shape != (len(labels), len(labels)):
        raise ValueError('matrix dimension must match terminal labels')
    observed = [root, *labels]
    distance = np.zeros((len(observed), len(observed)))
    distance[1:, 1:] = geo.distance
    distance[0, 1:] = distance[1:, 0] = geo.root_depths
    distance = .5 * (distance + distance.T)
    np.fill_diagonal(distance, 0)
    def key(a, b): return (min(a, b), max(a, b))
    values = {key(a,b):float(distance[i,j]) for i,a in enumerate(observed) for j,b in enumerate(observed) if i<j}
    def d(a,b): return 0.0 if a == b else values[key(a,b)]
    active = observed.copy()
    hidden = min([-1, *observed]) - 1
    edges = []
    while len(active) > 2:
        count = len(active)
        totals = {a:sum(d(a,b) for b in active) for a in active}
        a,b = min(combinations(active,2), key=lambda pair: ((count-2)*d(*pair)-totals[pair[0]]-totals[pair[1]],key(*pair)))
        separation = d(a,b)
        limb = .5*separation + (totals[a]-totals[b])/(2*(count-2))
        edges.extend(((hidden,a,max(0.,limb)),(hidden,b,max(0.,separation-limb))))
        remaining = [v for v in active if v not in (a,b)]
        for v in remaining:
            # The classical NJ update retains signed dissimilarities; clipping
            # here would change subsequent Q-criterion choices.
            values[key(hidden,v)] = .5*(d(a,v)+d(b,v)-separation)
        active = [*remaining,hidden]
        hidden -= 1
    edges.append((active[0],active[1],max(0.,d(*active))))
    threshold = float(collapse_factor) * max(float(np.median(geo.root_depths)),1e-12)
    compact = _contract_zero_internal_edges(edges,root,labels,atol=max(1e-10,threshold))
    return RootedTreeResult(root=root,terminals=tuple(labels),edges=compact,group_tolerance=threshold)