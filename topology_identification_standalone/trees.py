"""Tree geometry and hierarchy operations; only NumPy/SciPy are required.

The RNJ recursion and observed-root classical NJ follow the project's audited
implementations. The ordered sensitivity QP is re-expressed locally; there are
no imports from the parent repository. All supports use zero-based positions
in the observation columns, not physical bus labels.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint, minimize


def canonical(family):
    return tuple(sorted({tuple(sorted(map(int, s))) for s in family}, key=lambda s: (len(s), s)))


def basis(n):
    return canonical([*((i,) for i in range(n)), tuple(range(n))])


def compatible(left, right):
    a, b = set(left), set(right)
    return not a & b or a <= b or b <= a


def is_laminar(family):
    family = canonical(family)
    return all(compatible(a, b) for a, b in combinations(family, 2))


def children(family, n):
    nodes = canonical([*family, *basis(n)])
    if any(not s or min(s) < 0 or max(s) >= n for s in nodes) or not is_laminar(nodes):
        raise ValueError("Supports must be nonempty, in range and laminar")
    return {v: canonical(s for s in nodes if set(s) < set(v)
                         and not any(set(s) < set(t) < set(v) for t in nodes)) for v in nodes}


def max_direct_blocks(family, n):
    return max((len(blocks) for blocks in children(family, n).values()), default=1)


def frontier_candidates(family, n, block_limit=5, hard=()):
    """Pool every proper union of every complete-subtree cut of width <= b.

    Width counts subtrees, not their terminal counts. An old support is
    releasable when one of these cuts opens it. The union across cuts can be
    substantially larger than the 2**b-b-2 candidates of one cut.
    """
    if not isinstance(block_limit, int) or block_limit < 3:
        raise ValueError("block_limit must be an integer >= 3")
    family = canonical([*family, *basis(n), *hard]); hierarchy = children(family, n)

    @lru_cache(None)
    def cuts(node):
        result = {(node,)}
        if hierarchy[node]:
            partial = {()}
            for child in hierarchy[node]:
                partial = {canonical([*a, *b]) for a in partial for b in cuts(child)
                           if len(a)+len(b) <= block_limit}
            result.update(partial)
        return tuple(sorted(result, key=lambda cut: (len(cut), cut)))

    generated, frontiers = set(), []
    for parent in family:
        for blocks in cuts(parent):
            if 3 <= len(blocks) <= block_limit:
                local = {tuple(sorted(i for block in chosen for i in block))
                         for size in range(2, len(blocks)) for chosen in combinations(blocks, size)}
                generated.update(local)
                frontiers.append(dict(parent=parent, blocks=blocks, candidate_count=len(local)))
    protected = set(basis(n)) | set(canonical(hard))
    released = canonical((set(family) & generated)-protected)
    keep = canonical(set(family)-set(released)); pool = canonical([*family, *generated])
    if any(not compatible(s, fixed) for s in pool for fixed in keep if fixed not in protected):
        raise RuntimeError("A candidate crosses an unopened skeleton support")
    return dict(pool=pool, keep=keep, released=released, frontiers=frontiers,
                candidates=canonical(set(pool)-set(keep)), max_direct_blocks=max_direct_blocks(family, n))


def _observations(train):
    values = [train[k] if isinstance(train, dict) else getattr(train, k) for k in ("p", "q", "y")]
    p, q, y = [np.asarray(v, float) for v in values]
    if p.ndim != 2 or p.shape != q.shape or p.shape != y.shape or not p.size:
        raise ValueError("p, q, y must have the same nonempty samples-by-terminals shape")
    if not all(np.isfinite(v).all() for v in (p, q, y)):
        raise ValueError("Observations must be finite")
    return p, q, y


def fit_sensitivity(train):
    """Fit symmetric nonnegative R/X with entrywise diagonal order, no ridge.

    The fixed 1e-6 diagonal margin is derived once from symmetric OLS. SLSQP
    solves a convex QP in equivalent preconditioned coordinates; the spectral
    scale floor changes coordinates only, never the objective's Hessian.
    """
    p, q, y = _observations(train); n = p.shape[1]
    design = np.c_[p, q]; ols = np.linalg.lstsq(design, y, rcond=None)[0]
    initial = np.stack([.5*(v+v.T) for v in (ols[:n], ols[n:])])
    margins = []
    for matrix in np.maximum(initial, 0.):
        positive = np.diag(matrix)[np.diag(matrix) > 0.]
        margins.append(1e-6*(float(np.median(positive)) if positive.size else float(matrix.max())))
    margins = np.asarray(margins)
    target_norm, design_norm = np.linalg.norm(y), np.linalg.norm(design)
    scales = np.ones(2) if not target_norm or not design_norm else np.array([
        target_norm/(np.linalg.norm(a) or design_norm) for a in (p, q)])
    scales = np.maximum(scales, margins)
    ii, jj = np.triu_indices(n); k = len(ii)
    feature = np.zeros((len(y), n, 2*k))
    for block, a in enumerate((p, q)):
        cols = block*k+np.arange(k)
        feature[:, jj, cols] = scales[block]*a[:, ii]
        feature[:, ii, cols] = scales[block]*a[:, jj]
    feature = feature.reshape(y.size, 2*k)/(target_norm or 1.)
    target = y.ravel()/(target_norm or 1.)
    rows, lower = list(np.eye(2*k)), [0.]*int(2*k)
    position = np.empty((n, n), int); position[ii, jj] = position[jj, ii] = np.arange(k)
    for block, margin in enumerate(margins/scales):
        for i in range(n):
            for j in range(n):
                if i != j:
                    row = np.zeros(2*k)
                    row[block*k+position[i, i]] = 1.; row[block*k+position[i, j]] = -1.
                    rows.append(row); lower.append(float(margin))
    constraint, lower = np.asarray(rows), np.asarray(lower)
    starts = []
    for value in (np.maximum(initial, 0.), np.zeros_like(initial)):
        for matrix, margin in zip(value, margins):
            if n > 1:
                off = matrix.copy(); np.fill_diagonal(off, -np.inf)
                np.fill_diagonal(matrix, np.maximum(np.diag(matrix), off.max(axis=1)+margin))
        starts.append((value[:, ii, jj]/scales[:, None]).ravel())
    start = min(starts, key=lambda v: np.linalg.norm(feature@v-target))
    hessian = feature.T@feature; hessian = .5*(hessian+hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian); shift = np.zeros(2*k)
    if eigenvalues[-1] > 0. and eigenvalues[0] > 1e-12*eigenvalues[-1]:
        upper = np.linalg.cholesky(hessian).T
        transform = solve_triangular(upper, np.eye(2*k), lower=False)
        shift = -transform.T@(feature.T@target); start = upper@start+shift
        objective = lambda u: (.5*float(u@u), u)
        coordinates = "Cholesky_whitened"
    else:
        roots = np.sqrt(np.maximum(eigenvalues, 1e-6*eigenvalues[-1])) if eigenvalues[-1] > 0. else np.ones(2*k)
        transform = eigenvectors/roots; start = roots*(eigenvectors.T@start)
        transformed_feature = feature@transform
        def objective(u):
            residual = transformed_feature@u-target
            return .5*float(residual@residual), transformed_feature.T@residual
        coordinates = "spectral_preconditioned"
    c = constraint@transform; d = lower+c@shift; norms = np.linalg.norm(c, axis=1)
    result = minimize(objective, start, jac=True, method="SLSQP",
                      constraints=LinearConstraint(c/norms[:, None], d/norms, np.inf),
                      options={"maxiter": 500, "ftol": 1e-12})
    values = transform@(result.x-shift)
    violation = max(0., float(np.max(lower-constraint@values)))
    if not result.success or not np.isfinite(values).all() or violation > 1e-8:
        raise RuntimeError(f"Ordered sensitivity QP failed: {result.message}; violation={violation:g}")
    matrices = np.zeros((2, n, n)); matrices[:, ii, jj] = values.reshape(2, k)*scales[:, None]
    matrices[:, jj, ii] = matrices[:, ii, jj]; matrices = np.maximum(matrices, 0.)
    for matrix, margin in zip(matrices, margins):
        if n > 1:
            off = matrix.copy(); np.fill_diagonal(off, -np.inf)
            np.fill_diagonal(matrix, np.maximum(np.diag(matrix), off.max(axis=1)+margin))
    residual = p@matrices[0]+q@matrices[1]-y
    diagnostics = dict(solver="SciPy SLSQP convex_QP", success=True, iterations=int(result.nit),
                       coordinates=coordinates, r_margin=float(margins[0]), x_margin=float(margins[1]),
                       squared_residual_sum=float(np.sum(residual**2)), maximum_scaled_violation=violation,
                       constraint_mode="ordered", ridge=0., diagonal_margin_ratio=1e-6)
    return matrices[0], matrices[1], diagnostics


def geometry(R, X, weight=.75):
    """Normalize positive pairwise R/X distances, then form shared paths."""
    R, X = np.asarray(R, float), np.asarray(X, float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or X.shape != R.shape or not 0. <= weight <= 1.:
        raise ValueError("Expected equally sized square R/X and a weight in [0,1]")
    if not np.isfinite(R).all() or not np.isfinite(X).all():
        raise ValueError("R/X must be finite")
    scales = []
    for matrix in (R, X):
        d = np.diag(matrix)[:, None]+np.diag(matrix)[None, :]-2*matrix
        positive = d[d > 0.]
        scales.append(max(float(positive.mean()) if positive.size else 1., 1e-12))
    shared = weight*R/scales[0]+(1-weight)*X/scales[1]
    shared = np.maximum(0., .5*(shared+shared.T)); depth = np.maximum(0., np.diag(shared))
    shared = np.minimum(shared, np.minimum(depth[:, None], depth[None, :]))
    return shared, depth, dict(weight=float(weight), mu_r=scales[0], mu_x=scales[1])


def _tree_family(edges, n, threshold):
    graph = {}
    for a, b, length in edges:
        graph.setdefault(a, []).append((b, length)); graph.setdefault(b, []).append((a, length))
    family = set(basis(n))
    def walk(node, parent, length):
        support = {node} if 0 <= node < n else set()
        for child, edge_length in graph.get(node, []):
            if child != parent:
                support.update(walk(child, node, edge_length))
        if node != -1 and support and (0 <= node < n or 0 <= parent < n or length > threshold):
            family.add(tuple(sorted(support)))
        return support
    if walk(-1, None, 0.) != set(range(n)):
        raise RuntimeError("The inferred tree does not reach every terminal")
    return canonical(family)


def _tree_inputs(shared, depth, factor):
    shared, depth = np.asarray(shared, float), np.asarray(depth, float)
    if depth.ndim != 1 or not len(depth) or shared.shape != (len(depth), len(depth)):
        raise ValueError("Shared paths and root depths have incompatible shapes")
    if not np.isfinite(shared).all() or not np.isfinite(depth).all() or not np.isfinite(factor) or factor < 0.:
        raise ValueError("Tree inputs must be finite and tolerance nonnegative")
    depth = np.maximum(depth, 0.); shared = np.maximum(.5*(shared+shared.T), 0.)
    shared = np.minimum(shared, np.minimum(depth[:, None], depth[None, :]))
    return shared, depth, float(factor)*max(float(np.median(depth)), 1e-12)


def rnj(shared, depth, tolerance=0.):
    """Rooted NJ; tolerance is relative to median root-to-terminal depth."""
    shared, depth, threshold = _tree_inputs(shared, depth, tolerance); n = len(depth)
    depths = {i: float(v) for i, v in enumerate(depth)}
    key = lambda a, b: (min(a, b), max(a, b))
    values = {key(i, j): float(shared[i, j]) for i, j in combinations(range(n), 2)}
    get = lambda a, b: depths[a] if a == b else values[key(a, b)]
    active, edges, hidden = list(range(n)), [], -2
    while len(active) > 1:
        a, b = max(combinations(active, 2), key=lambda pair: (get(*pair), tuple(-v for v in sorted(pair))))
        if get(a, b) <= threshold:
            edges.extend((-1, node, depths[node]) for node in active); active = []; break
        parent_depth = get(a, b); group = [a, b]
        group.extend(node for node in active if node not in group and parent_depth-min(get(a, node), get(b, node)) <= threshold)
        depths[hidden] = parent_depth
        edges.extend((hidden, node, max(0., depths[node]-parent_depth)) for node in group)
        remaining = [node for node in active if node not in group]
        for node in remaining:
            values[key(hidden, node)] = float(np.clip(np.median([get(member, node) for member in group]),
                                                     0., min(depths[hidden], depths[node])))
        active = [*remaining, hidden]; hidden -= 1
    if active:
        edges.append((-1, active[0], depths[active[0]]))
    return _tree_family(edges, n, 1e-10)


def nj(shared, depth, collapse_factor=0.):
    """Classical Q-criterion NJ including the observed root as an anchor leaf."""
    shared, depth, threshold = _tree_inputs(shared, depth, collapse_factor); n = len(depth)
    observed = [-1, *range(n)]; key = lambda a, b: (min(a, b), max(a, b))
    distance = np.zeros((n+1, n+1)); distance[1:, 1:] = depth[:, None]+depth[None, :]-2*shared
    distance[0, 1:] = distance[1:, 0] = depth
    values = {key(a, b): float(distance[i, j]) for i, a in enumerate(observed) for j, b in enumerate(observed) if i < j}
    d = lambda a, b: 0. if a == b else values[key(a, b)]
    active, edges, hidden = observed.copy(), [], -2
    while len(active) > 2:
        count = len(active); total = {a: sum(d(a, b) for b in active) for a in active}
        a, b = min(combinations(active, 2), key=lambda pair: ((count-2)*d(*pair)-total[pair[0]]-total[pair[1]], key(*pair)))
        separation = d(a, b); limb = .5*separation+(total[a]-total[b])/(2*(count-2))
        edges.extend(((hidden, a, max(0., limb)), (hidden, b, max(0., separation-limb))))
        remaining = [node for node in active if node not in (a, b)]
        for node in remaining:
            values[key(hidden, node)] = .5*(d(a, node)+d(b, node)-separation)
        active = [*remaining, hidden]; hidden -= 1
    edges.append((active[0], active[1], max(0., d(*active))))
    return _tree_family(edges, n, max(1e-10, threshold))


def rnj_guidance(train):
    """One ordered fit, fifteen raw RNJ families, no fitted-zero pruning.

    This preserves the original generated supports. Historical experiments
    sometimes lost these supports by exporting only zero-pruned warm families;
    this standalone version makes the candidate-generation contract explicit.
    """
    R, X, diagnostics = fit_sensitivity(train); families, origins = [], []
    for weight in (1., .75, 0.):
        shared, depth, calibration = geometry(R, X, weight)
        for tolerance in (0., .04, .08, .16, .24):
            family = rnj(shared, depth, tolerance)
            families.append(family); origins.append(dict(weight=weight, tolerance=tolerance))
    shared, depth, calibration = geometry(R, X, .75)
    reference = rnj(shared, depth, .16)
    return dict(reference_family=reference, initial=reference,
                pool=canonical(s for family in families for s in family), warm_families=tuple(families),
                family_origins=origins, diagnostics=diagnostics, R=R, X=X,
                shared=shared, depth=depth, calibration=calibration, fitted_zero_pruning=False)
