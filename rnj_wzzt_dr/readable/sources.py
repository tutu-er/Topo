"""Step 2 of WZZT-C (readable edition): the sources that PROPOSE lines. They never decide.

Each source returns one hierarchy (a tree) over the meters. Every line of every source becomes a candidate; the
exact MIQP of step 3 decides which candidates the data support.

Two kinds of source:
- Distance builders (RNJ, NJ, RG) on the best-fitting matrix R. R[i, j] estimates the resistance shared by the
  paths from the root to meters i and j. It is fitted to the active power only ("P-only"): with a fixed power
  factor Q moves with P, the fit then returns R + tan(phi) X, which has exactly the same tree.
- Clusterings (average or Ward linkage) of the meters by how alike their voltage series or matrix rows are.

Same mathematics as identification/revised.py (ordered_sensitivity), identification/trees.py (geometry, rnj, nj, rg)
and identification/combo.py (source_family).
"""
import math

import numpy as np
import gurobipy as gp
from scipy import sparse
from scipy.cluster.hierarchy import linkage

from .data import basic_lines, canonical, gurobi_env, internal


# ------------------------------------------------------------------ the best-fitting matrix (P-only, ordered)
def fitted_matrix(problem):
    """Symmetric R >= 0 with R[i, i] >= R[i, j] (a meter shares no more path with others than with itself),
    fitted to the active power by least squares after the residual operator (a convex QP, Gurobi barrier)."""
    T, n = problem.T, problem.n
    ii, jj = np.triu_indices(n)
    k = len(ii)
    F = np.zeros((T, n, k))
    F[:, jj, np.arange(k)] = problem.P[:, ii]
    F[:, ii, np.arange(k)] = problem.P[:, jj]
    feature = problem.slot_step(F).reshape(T*n, k)
    target = problem.target.ravel()
    norm = float(np.linalg.norm(target)) or 1.
    # a tiny margin for the order constraints, from an unconstrained fit
    ols = np.linalg.lstsq(feature, target, rcond=None)[0]
    R0 = np.zeros((n, n))
    R0[ii, jj] = ols
    R0[jj, ii] = ols
    R0 = np.maximum(R0, 0.)
    positive = np.diag(R0)[np.diag(R0) > 0.]
    if positive.size:
        margin = 1e-6*float(np.median(positive))
    else:
        margin = 1e-6*max(float(R0.max()), 1e-12)
    scale = max(norm/(float(np.linalg.norm(feature)) or 1.), margin)
    feature_s = feature*scale/norm
    target_s = target/norm
    # order constraints R[i, i] - R[i, j] >= margin, one row per ordered pair i != j
    position = np.empty((n, n), int)
    position[ii, jj] = np.arange(k)
    position[jj, ii] = np.arange(k)
    rows, cols, vals, rhs = [], [], [], []
    count = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                rows += [count, count]
                cols += [position[i, i], position[i, j]]
                vals += [1., -1.]
                rhs.append(margin/scale)
                count += 1
    order = sparse.csr_matrix((vals, (rows, cols)), shape=(count, k))
    order.sort_indices()
    H = feature_s.T@feature_s
    H = .5*(H+H.T)
    g = feature_s.T@target_s
    with gp.Model(env=gurobi_env()) as model:
        model.Params.Threads = 1
        model.Params.BarConvTol = 1e-10
        v = model.addMVar(k, lb=0., ub=np.inf)
        if count:
            model.addMConstr(order, v, ">", np.array(rhs))
        model.setMObjective(.5*H, -g, .5*float(target_s@target_s), sense=gp.GRB.MINIMIZE)
        model.optimize()
        values = np.array(v.X)
    R = np.zeros((n, n))
    R[ii, jj] = values*scale
    R[jj, ii] = R[ii, jj]
    R = np.maximum(R, 0.)
    if problem.c == 1. and n > 1:                       # with a free source term R is known up to a constant
        off = R[~np.eye(n, dtype=bool)]
        R = R-off.min()
    for i in range(n):                                  # keep the diagonal at least as large as its row
        others = [R[i, j] for j in range(n) if j != i]
        if others:
            R[i, i] = max(R[i, i], max(others)+margin)
    return R


def shared_paths(R):
    """Normalise R into shared path lengths and root-to-meter depths (trees.geometry with weight 1)."""
    d = np.diag(R)[:, None]+np.diag(R)[None, :]-2*R
    positive = d[d > 0.]
    unit = max(float(positive.mean()) if positive.size else 1., 1e-12)
    shared = R/unit
    shared = np.maximum(0., .5*(shared+shared.T))
    depth = np.maximum(0., np.diag(shared))
    shared = np.minimum(shared, np.minimum(depth[:, None], depth[None, :]))
    return shared, depth


# ------------------------------------------------------------------ turning a set of tree edges into lines
def lines_from_edges(edges, n, threshold):
    """Edges (parent, child, length) of a tree rooted at -1 (meters 0..n-1, hidden nodes < -1) -> its lines.
    An edge into a hidden node is kept only when it is longer than threshold (short ones are contracted)."""
    graph = {}
    for a, b, length in edges:
        graph.setdefault(a, []).append((b, length))
        graph.setdefault(b, []).append((a, length))
    lines = set(basic_lines(n))

    def walk(node, parent, length):
        below = {node} if 0 <= node < n else set()
        for child, edge_length in graph.get(node, []):
            if child != parent:
                below |= walk(child, node, edge_length)
        keep = (0 <= node < n) or (parent is not None and 0 <= parent < n) or length > threshold
        if node != -1 and below and keep:
            lines.add(tuple(sorted(below)))
        return below

    reached = walk(-1, None, 0.)
    if reached != set(range(n)):
        raise RuntimeError("the tree does not reach every meter")
    return canonical(lines)


def tree_inputs(shared, depth, tolerance):
    depth = np.maximum(np.asarray(depth, float), 0.)
    shared = np.maximum(.5*(np.asarray(shared, float)+np.asarray(shared, float).T), 0.)
    shared = np.minimum(shared, np.minimum(depth[:, None], depth[None, :]))
    threshold = float(tolerance)*max(float(np.median(depth)), 1e-12)
    return shared, depth, threshold


# ------------------------------------------------------------------ distance builders
def rnj(shared, depth, tolerance=0.):
    """Rooted neighbour joining: repeatedly join the two nodes that share the longest path from the root."""
    shared, depth, threshold = tree_inputs(shared, depth, tolerance)
    n = len(depth)
    depth_of = {i: float(depth[i]) for i in range(n)}
    common = {}
    for i in range(n):
        for j in range(i+1, n):
            common[(i, j)] = float(shared[i, j])

    def get(a, b):
        if a == b:
            return depth_of[a]
        return common[(min(a, b), max(a, b))]

    active, edges, hidden = list(range(n)), [], -2
    while len(active) > 1:
        best_pair, best_key = None, None
        for x in range(len(active)):
            for y in range(x+1, len(active)):
                a, b = active[x], active[y]
                key = (get(a, b), tuple(-v for v in sorted((a, b))))
                if best_key is None or key > best_key:
                    best_pair, best_key = (a, b), key
        a, b = best_pair
        if get(a, b) <= threshold:                      # nothing shares enough path: all hang from the root
            for node in active:
                edges.append((-1, node, depth_of[node]))
            active = []
            break
        parent_depth = get(a, b)
        group = [a, b]
        for node in active:
            if node not in group and parent_depth-min(get(a, node), get(b, node)) <= threshold:
                group.append(node)
        depth_of[hidden] = parent_depth
        for node in group:
            edges.append((hidden, node, max(0., depth_of[node]-parent_depth)))
        remaining = [node for node in active if node not in group]
        for node in remaining:
            value = np.median([get(member, node) for member in group])
            common[(min(hidden, node), max(hidden, node))] = float(np.clip(value, 0., min(depth_of[hidden], depth_of[node])))
        active = remaining+[hidden]
        hidden -= 1
    if active:
        edges.append((-1, active[0], depth_of[active[0]]))
    return lines_from_edges(edges, n, 1e-10)


def nj(shared, depth, tolerance=0.):
    """Classical neighbour joining (Q criterion), with the root as an extra observed leaf."""
    shared, depth, threshold = tree_inputs(shared, depth, tolerance)
    n = len(depth)
    nodes = [-1]+list(range(n))
    dist = {}
    for x in range(len(nodes)):
        for y in range(x+1, len(nodes)):
            a, b = nodes[x], nodes[y]
            if a == -1:
                value = depth[b]
            else:
                value = depth[a]+depth[b]-2*shared[a, b]
            dist[(min(a, b), max(a, b))] = float(value)

    def d(a, b):
        if a == b:
            return 0.
        return dist[(min(a, b), max(a, b))]

    active, edges, hidden = list(nodes), [], -2
    while len(active) > 2:
        count = len(active)
        total = {a: sum(d(a, b) for b in active) for a in active}
        best_pair, best_key = None, None
        for x in range(count):
            for y in range(x+1, count):
                a, b = active[x], active[y]
                key = ((count-2)*d(a, b)-total[a]-total[b], (min(a, b), max(a, b)))
                if best_key is None or key < best_key:
                    best_pair, best_key = (a, b), key
        a, b = best_pair
        separation = d(a, b)
        limb = .5*separation+(total[a]-total[b])/(2*(count-2))
        edges.append((hidden, a, max(0., limb)))
        edges.append((hidden, b, max(0., separation-limb)))
        remaining = [node for node in active if node not in (a, b)]
        for node in remaining:
            dist[(min(hidden, node), max(hidden, node))] = .5*(d(a, node)+d(b, node)-separation)
        active = remaining+[hidden]
        hidden -= 1
    edges.append((active[0], active[1], max(0., d(active[0], active[1]))))
    return lines_from_edges(edges, n, max(1e-10, threshold))


def rg(shared, depth, tolerance=0.):
    """Recursive grouping (Choi et al. 2011): find siblings and parents from distance differences, add hidden
    parents, repeat. Meters may sit at internal nodes."""
    shared, depth, threshold = tree_inputs(shared, depth, tolerance)
    n = len(depth)
    eps = max(threshold, 1e-9*max(float(np.median(depth)), 1e-12))
    dist = {}

    def set_d(a, b, value):
        dist[(min(a, b), max(a, b))] = float(value)

    def d(a, b):
        if a == b:
            return 0.
        return dist[(min(a, b), max(a, b))]

    for i in range(n):
        set_d(-1, i, depth[i])
        for j in range(i+1, n):
            set_d(i, j, depth[i]+depth[j]-2*shared[i, j])
    active, edges, hidden = [-1]+list(range(n)), [], -2
    while len(active) > 2:
        m = len(active)
        D = np.array([[d(a, b) for b in active] for a in active])
        phi = D[:, None, :]-D[None, :, :]                      # phi[a, b, k] = d(a, k) - d(b, k)
        idx = np.arange(m)
        phi[idx, :, idx] = np.nan
        phi[:, idx, idx] = np.nan
        with np.errstate(invalid="ignore"):
            spread = np.nanmax(phi, axis=2)-np.nanmin(phi, axis=2)
            is_parent = np.nanmax(np.abs(phi-D[:, :, None]), axis=2) <= eps    # [a, b]: b is the parent of leaf a
        related = (spread <= eps) | is_parent | is_parent.T
        np.fill_diagonal(related, False)
        if not related.any():                                   # nothing related: group the most sibling-like pair
            masked = spread+np.diag(np.full(m, np.inf))
            a, b = np.unravel_index(int(np.argmin(masked)), masked.shape)
            related[a, b] = related[b, a] = True
        group_of = list(range(m))                              # connected components of "related"

        def find(x):
            while group_of[x] != x:
                group_of[x] = group_of[group_of[x]]
                x = group_of[x]
            return x

        for a in range(m):
            for b in range(a+1, m):
                if related[a, b]:
                    group_of[find(a)] = find(b)
        groups = {}
        for a in range(m):
            groups.setdefault(find(a), []).append(a)
        new_active, new_hidden = [], {}
        for members in groups.values():
            if len(members) == 1:
                new_active.append(active[members[0]])
                continue
            parent = None
            for p in members:
                if all(is_parent[i, p] for i in members if i != p):
                    parent = p
                    break
            if parent is not None:
                for i in members:
                    if i != parent:
                        edges.append((active[parent], active[i], max(0., D[i, parent])))
                new_active.append(active[parent])
                continue
            h = hidden
            hidden -= 1
            outside = [k for k in range(m) if k not in members]
            for i in members:
                values = []
                for j in members:
                    if j != i:
                        ks = outside or [k for k in range(m) if k not in (i, j)]
                        mean_phi = float(np.mean(D[i, ks]-D[j, ks])) if ks else 0.
                        values.append(.5*(D[i, j]+mean_phi))
                limb = max(0., float(np.mean(values)))
                set_d(active[i], h, limb)
                edges.append((h, active[i], limb))
            new_hidden[h] = [active[i] for i in members]
            new_active.append(h)
        for h, kids in new_hidden.items():
            for x in new_active:
                if x == h or (min(x, h), max(x, h)) in dist:
                    continue
                if x in new_hidden:
                    value = np.mean([d(i, j)-d(i, h)-d(j, x) for i in kids for j in new_hidden[x]])
                else:
                    value = np.mean([d(i, x)-d(i, h) for i in kids])
                set_d(h, x, max(0., float(value)))
        active = new_active
    if len(active) == 2:
        edges.append((active[0], active[1], max(0., d(active[0], active[1]))))
    return lines_from_edges(edges, n, max(1e-10, threshold))


# ------------------------------------------------------------------ clusterings
def correlation_distance(M):
    """1 - correlation between the columns of M (one column per meter), as a condensed list."""
    M = np.asarray(M, float)
    M = M-M.mean(axis=0)
    sd = M.std(axis=0)
    C = np.zeros((M.shape[1], M.shape[1]))
    ok = sd > 0
    C[np.ix_(ok, ok)] = np.corrcoef(M[:, ok], rowvar=False)
    np.fill_diagonal(C, 1.)
    D = np.clip(1.-C, 0., 2.)
    return D[np.triu_indices(M.shape[1], 1)]


def lines_from_clustering(distances, n, method="average"):
    """Every cluster formed while merging the meters bottom-up is a line."""
    Z = linkage(np.clip(np.nan_to_num(distances, nan=1.), 0., None), method=method)
    members = {i: [i] for i in range(n)}
    lines = list(basic_lines(n))
    for step, row in enumerate(Z):
        merged = members.pop(int(row[0]))+members.pop(int(row[1]))
        members[n+step] = merged
        if 1 < len(merged) < n:
            lines.append(tuple(sorted(merged)))
    return canonical(lines)


# ------------------------------------------------------------------ all sources
def source_lines(problem, name, R, tolerance=0.):
    """The internal lines one source proposes."""
    n = problem.n
    shared, depth = shared_paths(R)
    if name == "rnj":
        tree = rnj(shared, depth, tolerance)
    elif name == "nj":
        tree = nj(shared, depth, tolerance)
    elif name == "rg":
        tree = rg(shared, depth, tolerance)
    elif name == "volt":
        tree = lines_from_clustering(correlation_distance(problem.raw_y), n)
    elif name == "volt_m":
        tree = lines_from_clustering(correlation_distance(problem.target), n)
    elif name == "resid":
        star = basic_lines(n)
        theta, _ = problem.fit(star)
        residual = problem.y-problem.columns(canonical(star))@theta
        tree = lines_from_clustering(correlation_distance(residual.reshape(problem.T, n)), n)
    elif name == "rrows":
        tree = lines_from_clustering(correlation_distance(R.T), n)
    elif name in ("upgma", "ward"):
        d = np.maximum(depth[:, None]+depth[None, :]-2*shared, 0.)
        upper = d[np.triu_indices(n, 1)]
        if name == "upgma":
            tree = lines_from_clustering(upper, n, "average")
        else:
            tree = lines_from_clustering(np.sqrt(upper), n, "ward")
    else:
        raise ValueError(f"unknown source {name}")
    return internal(tree, n)


def propose(problem, settings):
    """Run every source; return {source: its lines} and the union of all lines (the candidates)."""
    R = fitted_matrix(problem)
    by_source = {}
    union = set()
    for name in settings.sources:
        lines = source_lines(problem, name, R, settings.tolerance)
        by_source[name] = lines
        union |= set(lines)
    return by_source, canonical(union), R
