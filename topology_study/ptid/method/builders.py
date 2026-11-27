"""The nine starting trees of draft v16 (Appendix B), which give the initial candidate lines.

R-hat (eq. B1): a symmetric matrix without tree structure, fitted by a convex QP (Gurobi) to the active-power part of
the model under M = D1 (x) S, where D1 removes the meters' offsets (and with a load, its profiles) and S whitens the
shared PCC error: R = R' >= 0 and R_ii - R_ij >= margin for i != j. From it and from the data, nine hierarchies over
the meters:
  rnj, nj, rg    recursive neighbour joining, neighbour joining (the PCC as an anchor leaf) and recursive grouping on
                 the shared paths (R-hat over its mean path distance);
  upgma, ward    average linkage on the path distance d_ij = R_ii + R_jj - 2 R_ij; Ward linkage on sqrt(d);
  volt, volt_m, resid, rrows
                 average linkage on the correlation distance between meters of: the drops y; the drops under M; the
                 residuals of the fit with the service cables alone under M; the rows of R-hat.
A rewrite of the registered predecessor's builders (identification/combo.py, revised.py, trees.py), checked to give the
same trees on the development twins (RESULTS.md, "Builders").
"""
from __future__ import annotations

from itertools import combinations

import gurobipy as gp
import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import linkage

from .model import box_ls, env, gls_gamma
from .sets import basis, canonical, lines_of

NAMES = ("rnj", "nj", "rg", "volt", "volt_m", "resid", "rrows", "upgma", "ward")


def builder_trees(P, Q, Y, V0sq, u=None):
    """The lines (meter groupings) of the nine trees, in the order of NAMES."""
    P, Q, Y, V0sq = (np.asarray(a, float) for a in (P, Q, Y, V0sq))
    n = P.shape[1]
    gamma = gls_gamma(V0sq, Y)
    drop = d1(u)
    Pm, Qm, Ym = drop(P), drop(Q), drop(Y-gamma*Y.mean(1, keepdims=True))
    R = r_hat(Pm, Ym, gamma)
    shared, depth = geometry(R)
    d = np.maximum(depth[:, None]+depth[None, :]-2*shared, 0.)[np.triu_indices(n, 1)]
    families = dict(rnj=rnj(shared, depth), nj=nj(shared, depth), rg=rg(shared, depth),
                    volt=hierarchy(corr_distance(Y), n), volt_m=hierarchy(corr_distance(Ym), n),
                    resid=hierarchy(corr_distance(star_residual(Pm, Qm, Ym, gamma)), n),
                    rrows=hierarchy(corr_distance(R.T), n),
                    upgma=hierarchy(d, n, "average"), ward=hierarchy(np.sqrt(d), n, "ward"))
    return [lines_of(families[name], n) for name in NAMES]


def d1(u):
    """D1 as a function on T x m arrays: remove the time mean and, with a load, its (demeaned) profiles. The
    operations are done in the predecessor's order: recursive grouping tests equalities to 1e-9, so even rounding
    differences of 1e-16 in the data can change its tree."""
    if u is None:
        return lambda A: A-A.mean(0)
    X = np.column_stack(u)
    X = X-X.mean(0)
    U, s, _ = np.linalg.svd(X/np.maximum(np.linalg.norm(X, axis=0), 1e-300), full_matrices=False)
    U = U[:, s > 1e-8]
    return lambda A: (A-A.mean(0))-U@(U.T@(A-A.mean(0)))


# ------------------------------------------------------------------ R-hat (eq. B1)
def r_hat(Pm, Ym, gamma):
    """Symmetric nonnegative R with R_ii - R_ij >= margin, least squares of Ym on Pm R under S (Gurobi barrier).
    The problem is scaled as in the predecessor so that the same QP is solved."""
    T, n = Pm.shape
    ii, jj = np.triu_indices(n)
    k = len(ii)
    F = np.zeros((T, n, k))
    F[:, jj, np.arange(k)] = Pm[:, ii]                              # entry (i, j): meter i sees p_j, meter j sees p_i
    F[:, ii, np.arange(k)] = Pm[:, jj]
    feature = (F-gamma*F.mean(1, keepdims=True)).reshape(T*n, k)
    target = Ym.ravel()
    norm = float(np.linalg.norm(target)) or 1.
    R0 = np.zeros((n, n))
    R0[ii, jj] = R0[jj, ii] = np.linalg.lstsq(feature, target, rcond=None)[0]
    diagonal = np.diag(np.maximum(R0, 0.))
    margin = 1e-6*(float(np.median(diagonal[diagonal > 0])) if (diagonal > 0).any() else max(float(R0.max()), 1e-12))
    scale = max(norm/(float(np.linalg.norm(feature)) or 1.), margin)
    A, b = feature*scale/norm, target/norm
    position = np.empty((n, n), int)
    position[ii, jj] = position[jj, ii] = np.arange(k)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]           # R_ii - R_ij >= margin
    order = sparse.csr_matrix((np.tile([1., -1.], len(pairs)), ([r for r in range(len(pairs)) for _ in (0, 1)],
                               [position[x] for i, j in pairs for x in ((i, i), (i, j))])), shape=(len(pairs), k))
    H = A.T@A
    with gp.Model(env=env()) as m:
        m.Params.Threads, m.Params.BarConvTol = 1, 1e-10
        v = m.addMVar(k, lb=0.)
        if len(pairs):
            m.addMConstr(order, v, ">", np.full(len(pairs), margin/scale))
        m.setMObjective(.25*(H+H.T), -A.T@b, .5*float(b@b), sense=gp.GRB.MINIMIZE)   # 0.5 ||A v - b||^2
        m.optimize()
        if m.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
            raise RuntimeError(f"R-hat QP ended with status {m.Status}")
        values = np.array(v.X)
    R = np.zeros((n, n))
    R[ii, jj] = R[jj, ii] = np.maximum(values*scale, 0.)
    if gamma == 1. and n > 1:
        R -= R[~np.eye(n, dtype=bool)].min()
    off = R.copy()
    np.fill_diagonal(off, -np.inf)
    np.fill_diagonal(R, np.maximum(np.diag(R), off.max(1)+margin))
    return R


def geometry(R):
    """Shared paths (R over its mean positive path distance, capped by the depths) and root depths."""
    d = np.diag(R)[:, None]+np.diag(R)[None, :]-2*R
    shared = R/max(float(d[d > 0].mean()) if (d > 0).any() else 1., 1e-12)
    shared = np.maximum(0., .5*(shared+shared.T))
    depth = np.maximum(0., np.diag(shared))
    return np.minimum(shared, np.minimum(depth[:, None], depth[None, :])), depth


# ------------------------------------------------------------------ the distance builders
def tree_family(edges, n, threshold):
    """The supports of a tree given by edges (a, b, length), rooted at -1: every meter, and every subtree whose edge to
    its parent is longer than `threshold` or touches a meter."""
    graph = {}
    for a, b, length in edges:
        graph.setdefault(a, []).append((b, length))
        graph.setdefault(b, []).append((a, length))
    family = set(basis(n))

    def walk(node, parent, length):
        support = {node} if 0 <= node < n else set()
        for child, edge in graph.get(node, []):
            if child != parent:
                support |= walk(child, node, edge)
        if node != -1 and support and (0 <= node < n or 0 <= parent < n or length > threshold):
            family.add(tuple(sorted(support)))
        return support
    if walk(-1, None, 0.) != set(range(n)):
        raise RuntimeError("the tree does not reach every meter")
    return canonical(family)


def rnj(shared, depth):
    """Recursive neighbour joining: join the pair with the longest shared path, the new node's shared path to the
    others being the median over its members."""
    n = len(depth)
    depths = dict(enumerate(map(float, depth)))
    key = lambda a, b: (min(a, b), max(a, b))
    values = {key(i, j): float(shared[i, j]) for i, j in combinations(range(n), 2)}
    get = lambda a, b: depths[a] if a == b else values[key(a, b)]
    active, edges, hidden = list(range(n)), [], -2
    while len(active) > 1:
        a, b = max(combinations(active, 2), key=lambda pair: (get(*pair), tuple(-v for v in sorted(pair))))
        if get(a, b) <= 0.:
            edges += [(-1, node, depths[node]) for node in active]
            active = []
            break
        top = get(a, b)
        group = [a, b]+[x for x in active if x not in (a, b) and top-min(get(a, x), get(b, x)) <= 0.]
        depths[hidden] = top
        edges += [(hidden, x, max(0., depths[x]-top)) for x in group]
        rest = [x for x in active if x not in group]
        for x in rest:
            values[key(hidden, x)] = float(np.clip(np.median([get(g, x) for g in group]), 0., min(top, depths[x])))
        active, hidden = [*rest, hidden], hidden-1
    if active:
        edges.append((-1, active[0], depths[active[0]]))
    return tree_family(edges, n, 1e-10)


def nj(shared, depth):
    """Neighbour joining (Q criterion) on the path distances, with the PCC as an observed leaf (-1)."""
    n = len(depth)
    nodes = [-1, *range(n)]
    key = lambda a, b: (min(a, b), max(a, b))
    D = np.zeros((n+1, n+1))
    D[1:, 1:] = depth[:, None]+depth[None, :]-2*shared
    D[0, 1:] = D[1:, 0] = depth
    values = {key(a, b): float(D[i, j]) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if i < j}
    dist = lambda a, b: 0. if a == b else values[key(a, b)]
    active, edges, hidden = nodes.copy(), [], -2
    while len(active) > 2:
        count = len(active)
        total = {a: sum(dist(a, b) for b in active) for a in active}
        a, b = min(combinations(active, 2), key=lambda p: ((count-2)*dist(*p)-total[p[0]]-total[p[1]], key(*p)))
        gap = dist(a, b)
        limb = .5*gap+(total[a]-total[b])/(2*(count-2))
        edges += [(hidden, a, max(0., limb)), (hidden, b, max(0., gap-limb))]
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            values[key(hidden, x)] = .5*(dist(a, x)+dist(b, x)-gap)
        active, hidden = [*rest, hidden], hidden-1
    edges.append((active[0], active[1], max(0., dist(*active))))
    return tree_family(edges, n, 1e-10)


def rg(shared, depth):
    """Recursive grouping (Choi, Tan, Anandkumar and Willsky, JMLR 2011) on the path metric with the PCC (-1) observed.
    With Phi_abk = d_ak - d_bk over the other active nodes k: b is the parent of leaf a when Phi_abk = d_ab for every k;
    a and b are siblings when Phi_abk is the same for every k (to eps). Related nodes form families; a family gets its
    observed parent when one passes the parent test against all members, else a new hidden parent. When no pair is
    related, the most sibling-like pair is grouped."""
    n = len(depth)
    eps = 1e-9*max(float(np.median(depth)), 1e-12)
    dist = {}

    def setd(a, b, v):
        dist[(min(a, b), max(a, b))] = float(v)

    def d(a, b):
        return 0. if a == b else dist[(min(a, b), max(a, b))]
    for i in range(n):
        setd(-1, i, depth[i])
        for j in range(i+1, n):
            setd(i, j, depth[i]+depth[j]-2*shared[i, j])
    active, edges, hidden = [-1, *range(n)], [], -2
    while len(active) > 2:
        m = len(active)
        D = np.array([[d(a, b) for b in active] for a in active])
        phi = D[:, None, :]-D[None, :, :]                           # phi[a, b, k] = d(a, k) - d(b, k)
        idx = np.arange(m)
        phi[idx, :, idx] = np.nan
        phi[:, idx, idx] = np.nan                                   # k is neither a nor b
        with np.errstate(invalid="ignore"):
            spread = np.nanmax(phi, axis=2)-np.nanmin(phi, axis=2)
            is_parent = np.nanmax(np.abs(phi-D[:, :, None]), axis=2) <= eps   # [a, b]: b is the parent of leaf a
        related = (spread <= eps) | is_parent | is_parent.T
        np.fill_diagonal(related, False)
        if not related.any():
            a, b = np.unravel_index(int(np.argmin(spread+np.diag(np.full(m, np.inf)))), spread.shape)
            related[a, b] = related[b, a] = True
        root = list(range(m))

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x
        for a, b in zip(*np.nonzero(np.triu(related, 1))):
            root[find(a)] = find(b)
        groups = {}
        for a in range(m):
            groups.setdefault(find(a), []).append(a)
        new_active, new_hidden = [], {}
        for members in groups.values():
            if len(members) == 1:
                new_active.append(active[members[0]])
                continue
            parent = next((p for p in members if all(is_parent[i, p] for i in members if i != p)), None)
            if parent is not None:
                edges += [(active[parent], active[i], max(0., D[i, parent])) for i in members if i != parent]
                new_active.append(active[parent])
                continue
            h, hidden = hidden, hidden-1
            outside = [k for k in range(m) if k not in members]
            for i in members:
                limbs = []
                for j in members:
                    if j != i:
                        ks = outside or [k for k in range(m) if k not in (i, j)]
                        limbs.append(.5*(D[i, j]+(float(np.mean(D[i, ks]-D[j, ks])) if ks else 0.)))
                limb = max(0., float(np.mean(limbs)))
                setd(active[i], h, limb)
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
                setd(h, x, max(0., float(value)))
        active = new_active
    if len(active) == 2:
        edges.append((active[0], active[1], max(0., d(*active))))
    return tree_family(edges, n, 1e-10)


# ------------------------------------------------------------------ the clusterings
def corr_distance(M):
    """1 - correlation between the columns of M (condensed); a constant column has correlation 0 with the others."""
    M = np.asarray(M, float)
    M = M-M.mean(0)
    ok = M.std(0) > 0
    C = np.zeros((M.shape[1], M.shape[1]))
    C[np.ix_(ok, ok)] = np.corrcoef(M[:, ok], rowvar=False)
    np.fill_diagonal(C, 1.)
    return np.clip(1.-C, 0., 2.)[np.triu_indices(M.shape[1], 1)]


def hierarchy(dist, n, method="average"):
    """Every cluster that a hierarchical clustering merges (SciPy's linkage), with the basis."""
    Z = linkage(np.clip(np.nan_to_num(dist, nan=1.), 0., None), method=method)
    members, out = {i: [i] for i in range(n)}, []
    for s, (a, b, *_) in enumerate(Z):
        merged = members.pop(int(a))+members.pop(int(b))
        members[n+s] = merged
        if 1 < len(merged) < n:
            out.append(tuple(sorted(merged)))
    return canonical([*basis(n), *out])


def star_residual(Pm, Qm, Ym, gamma):
    """Residual (T x n) of the bounded fit with each meter's own line and the trunk only, under M."""
    T, n = Pm.shape
    family = canonical(basis(n))
    Z = np.zeros((len(family), n))
    for k, C in enumerate(family):
        Z[k, list(C)] = 1.
    W = Z-gamma*Z.mean(1, keepdims=True)
    A, B = Pm@Z.T, Qm@Z.T                                           # each support's flow, T x K
    F = np.hstack([(A[:, None, :]*W.T[None]).reshape(T*n, -1), (B[:, None, :]*W.T[None]).reshape(T*n, -1)])
    return (Ym.ravel()-F@box_ls(F, Ym.ravel())).reshape(T, n)
