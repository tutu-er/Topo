"""WZZT-C: the explicit combination (prototype, development only; not registered, not frozen).

Distance-based and clustering sources propose the terminal side, an exact MIQP searches the root side, and the
node-by-node certificate checks the whole tree. Same model, residual operator and BIC score as WZZT-R (revised.py).

1. Matrix: WZZT-R's P-only ordered fit under M (Gurobi QP).
2. Sources (`sources`): each gives one hierarchy (a laminar family) over the meters.
   - distance builders on the matrix: "rnj", "nj", "rg" (at `tolerance`);
   - clustering (average linkage unless stated):
     "volt"   correlation distance between the meters' raw squared-voltage drops (per-meter demeaned);
     "volt_m" the same after the residual operator M (offsets and GLS slot term removed);
     "resid"  the residual of the star fit (each meter's own line) under M;
     "rrows"  correlation distance between rows of the fitted matrix R (sensitivity profiles);
     "upgma"  the path distance d_ij = R_ii + R_jj - 2 R_ij;
     "ward"   Ward linkage on sqrt(d_ij).
   Candidates: every source line (any size).
3. Blocks (`blocks_from`), the groups the root MIQP treats as single meters:
   - "builder":   the first prototype: the largest lines of the first source with at most `limit` x n meters;
   - "validated": T* = the exact best tree over all candidates (pool MIQP; terminal and root lines decided together);
                  the blocks are its largest lines;
   - "margin":    the one-parameter criterion: a line of T* becomes a block only when the data prefer it, by at least
                  m half-BIC units (an approximate log Bayes factor), both to removing it and to every candidate that
                  crosses it (with the lines of T* that candidate crosses removed); a line that fails is opened and its
                  sub-lines in T* are tested instead. No size limit by default: the evidence alone decides.
4. Root search: universe = terminal candidates compatible with the blocks + every union of two or more blocks
   (2^k - k - 2). The exact pool MIQP (revised.select_miqp, warm start and cutoff from a local search) returns the
   best laminar family in it. Skipped when k > k_max.
5. Certificate and repair: miqp_ext.certify_by_nodes over the result; a paying compatible line is added and the
   check repeats (at most `repairs` times).
"""
from __future__ import annotations

from itertools import combinations
import math
import time

import numpy as np
from scipy.cluster.hierarchy import linkage

from . import revised as rv
from . import miqp_ext as mx
from . import trees
from .trees import basis, canonical, compatible

SOURCES = ("rnj", "nj", "rg", "volt", "volt_m", "resid", "rrows", "upgma", "ward")


def setup(train, validation=None, config=None):
    """WZZT-R's scoring set-up with the guidance matrix solved by Gurobi."""
    config = config or rv.RevisedConfig()
    as_series = lambda o: o if isinstance(o, rv.Series) else rv.Series(o.p, o.q, o.y, getattr(o, "v0sq", None))
    train = as_series(train)
    data = train.join(as_series(validation)) if (config.history == "all" and validation is not None) else train
    T, n = data.p.shape
    N = T*n
    c = rv.root_gamma(data.v0sq, data.y) if config.source == "gls" else (1. if config.source == "free" else 0.)
    design = rv.Design(data, rv.Transform(T, n, c, config.offsets), 1.)
    R, info = rv.ordered_sensitivity(design, "gurobi")
    p_s, _ = rv.identifiable_per_support(data, config.pq_sigma)
    sigma2, _ = rv.saturated_sigma2(design)
    alpha = p_s*math.log(N)/N
    design.rescale(math.sqrt(sigma2))
    return dict(design=design, data=data, R=R, alpha=alpha, sigma2=sigma2, p_s=p_s, c=c, N=N, n=n, T=T,
                config=config)


# ------------------------------------------------------------------ sources
def _hierarchy(dist, n, method="average"):
    """Laminar family of every merged cluster of a hierarchical clustering (condensed distances)."""
    Z = linkage(np.clip(np.nan_to_num(dist, nan=1.), 0., None), method=method)
    members, out = {i: [i] for i in range(n)}, []
    for s, (a, b, *_rest) in enumerate(Z):
        merged = members.pop(int(a))+members.pop(int(b))
        members[n+s] = merged
        if 1 < len(merged) < n:
            out.append(tuple(sorted(merged)))
    return canonical([*basis(n), *out])


def _corr_distance(M):
    M = np.asarray(M, float)
    M = M-M.mean(0)
    sd = M.std(0)
    C = np.zeros((M.shape[1], M.shape[1]))
    ok = sd > 0
    C[np.ix_(ok, ok)] = np.corrcoef(M[:, ok], rowvar=False)
    np.fill_diagonal(C, 1.)
    D = np.clip(1.-C, 0., 2.)
    return D[np.triu_indices(M.shape[1], 1)]


def source_family(S, name, tolerance=0., bound=2.):
    """One source's hierarchy over the meters (a laminar family with singletons and the full set)."""
    n, R, design = S["n"], S["R"], S["design"]
    shared, depth, _ = trees.geometry(R, R, 1.)
    if name in ("rnj", "nj", "rg"):
        return canonical(dict(rnj=trees.rnj, nj=trees.nj, rg=trees.rg)[name](shared, depth, tolerance))
    iu = np.triu_indices(n, 1)
    if name == "volt":
        return _hierarchy(_corr_distance(S["data"].y), n)
    if name == "volt_m":
        return _hierarchy(_corr_distance(design.raw_target), n)
    if name == "resid":
        base = canonical(basis(n))
        E = design.residual(base, rv.exact_fit(design, base, bound)["theta"])
        return _hierarchy(_corr_distance(E), n)
    if name == "rrows":
        return _hierarchy(_corr_distance(R.T), n)
    d = np.maximum(depth[:, None]+depth[None, :]-2*shared, 0.)
    if name == "upgma":
        return _hierarchy(d[iu], n, "average")
    if name == "ward":
        return _hierarchy(np.sqrt(d[iu]), n, "ward")
    raise ValueError(f"unknown source {name}")


# ------------------------------------------------------------------ blocks
def maximal(lines):
    return [C for C in lines if not any(set(C) < set(D) for D in lines)]


def blocks_of(lines, n):
    """Blocks from a laminar set of lines: its maximal lines, plus single meters outside them."""
    top = maximal(canonical(lines))
    covered = {i for C in top for i in C}
    return canonical([*top, *((i,) for i in range(n) if i not in covered)])


def terminal_blocks(family, n, limit=.5):
    """Terminal candidates (internal lines with at most limit*n meters) and the blocks of a laminar family."""
    cand = [C for C in canonical(family) if 1 < len(C) <= limit*n]
    return cand, blocks_of(cand, n)


def root_unions(blocks, n):
    """Every union of two or more blocks, except the union of all of them."""
    k = len(blocks)
    out = []
    for size in range(2, k):
        for combo in combinations(range(k), size):
            out.append(tuple(sorted(i for b in combo for i in blocks[b])))
    return canonical(out)


def line_margins(prob, selected, candidates, N):
    """For each selected line C: the BIC-unit evidence for keeping it, min over (a) removing it and (b) replacing it
    by any candidate D that crosses it (the selected lines D crosses are removed with it)."""
    S0 = frozenset(selected)
    s0 = prob.score(S0)
    out = {}
    for C in selected:
        worst = prob.score(S0-{C})-s0
        for D in candidates:
            if not compatible(C, D):
                T = frozenset(E for E in S0 if compatible(E, D)) | {D}
                worst = min(worst, prob.score(T)-s0)
        out[C] = .5*N*worst
    return out


def line_evidence(prob, selected, candidates, N):
    """For each selected line C, the score differences (half-BIC units, i.e. approximate log Bayes factors) of every
    alternative to keeping it: removing C, or replacing it by a candidate D that crosses it (with the selected lines
    D crosses removed). Returns {C: dict(margin, posterior, alternatives)}: margin = the smallest difference;
    posterior = 1 / (1 + sum exp(-difference)), the BIC-approximate probability of C against all its alternatives
    under equal prior odds."""
    S0 = frozenset(selected)
    s0 = prob.score(S0)
    out = {}
    for C in selected:
        diffs = [.5*N*(prob.score(S0-{C})-s0)]
        for D in candidates:
            if not compatible(C, D):
                T = frozenset(E for E in S0 if compatible(E, D)) | {D}
                diffs.append(.5*N*(prob.score(T)-s0))
        d = np.array(diffs)
        out[C] = dict(margin=float(d.min()), posterior=float(1./(1.+np.exp(-np.clip(d, -700., 700.)).sum())),
                      alternatives=len(diffs))
    return out


def margin_blocks(selected, margin_of, m, n, limit):
    """Top-down: a selected terminal line is a block when its margin is at least m; otherwise its selected
    sub-lines are tested. Meters outside every block are single-meter blocks."""
    terminal = [C for C in selected if len(C) <= limit*n]
    chosen = []
    def visit(C):
        if margin_of.get(C, -math.inf) >= m:
            chosen.append(C); return
        for D in maximal([E for E in terminal if set(E) < set(C)]):
            visit(D)
    for C in maximal(terminal):
        visit(C)
    return blocks_of(chosen, n)


# ------------------------------------------------------------------ the method
def fit(train, validation=None, config=None, *, sources=("rnj",), tolerance=0., limit=None, k_max=9,
        blocks_from="builder", m=4.6, search_seconds=30., miqp_seconds=240., node_seconds=60., repairs=10, seed=7,
        pairs=False, pair_rounds=3, reselect=False, S=None):
    """blocks_from="builder" (first prototype): the first source's largest lines with at most `limit` x n meters
    (default 0.5) are the blocks. "validated" / "margin": T* = the exact best tree over every source's lines of any
    size; "validated" takes T*'s largest lines (at most `limit` x n meters, default 1) as blocks, "margin" only those
    whose evidence margin against every alternative is at least `m` (default limit 1: the evidence alone decides).
    The root search then runs over the blocks (skipped when more than k_max blocks remain; the output is then T*
    plus the certificate's repairs)."""
    config = config or rv.RevisedConfig()
    t0 = time.perf_counter()
    S = S or setup(train, validation, config)
    design, alpha, n, N, bound = S["design"], S["alpha"], S["n"], S["N"], config.bound
    timing = dict(setup=time.perf_counter()-t0)
    base = canonical(basis(n))
    crit = lambda fam: rv.exact_fit(design, fam, bound)["objective"]+alpha*rv.internal_count(fam, n)
    if blocks_from not in ("builder", "validated", "margin"):
        raise ValueError("blocks_from must be builder, validated or margin")
    limit = (.5 if blocks_from == "builder" else 1.) if limit is None else limit

    t1 = time.perf_counter()
    families = {name: source_family(S, name, tolerance, bound) for name in sources}
    per_source = {name: [C for C in fam if 1 < len(C) < n] for name, fam in families.items()}
    cand = canonical(C for lines in per_source.values() for C in lines)
    timing["sources"] = time.perf_counter()-t1
    record = dict(sources=list(sources), tolerance=tolerance, limit=limit, blocks_from=blocks_from, m=m,
                  candidates=len(cand), per_source={k: len(v) for k, v in per_source.items()},
                  builder_tree=[list(C) for C in families[sources[0]] if 1 < len(C) < n])

    # ---- blocks
    t1 = time.perf_counter()
    start = frozenset()
    if blocks_from == "builder":
        own = [C for C in per_source[sources[0]] if len(C) <= limit*n]
        blocks = blocks_of(own, n)
        cand = canonical(C for C in cand if len(C) <= limit*n)        # the first prototype's terminal candidates
        start = frozenset(own)
    else:
        rec0 = rv.select_miqp(design, cand, base, alpha, bound, miqp_seconds, seed=seed, gap=1e-4)
        kept = [C for C in rec0.get("family", ()) if 1 < len(C) < n]
        record["joint_selection"] = dict(status=rec0["status"], kept=[list(C) for C in kept], seconds=rec0["seconds"],
                                         gap=rec0.get("gap"))
        start = frozenset(kept)
        if blocks_from == "validated":
            blocks = blocks_of([C for C in kept if len(C) <= limit*n], n)
        else:
            prob0 = rv.Problem(rv.GramFitter(design, [*cand, *base], base, bound), cand, base, alpha)
            ev = line_evidence(prob0, kept, cand, N)
            blocks = margin_blocks(kept, {C: e["margin"] for C, e in ev.items()}, m, n, limit)
            record["evidence"] = [dict(line=list(C), margin=round(e["margin"], 3), posterior=e["posterior"],
                                       alternatives=e["alternatives"]) for C, e in ev.items()]
    k = len(blocks)
    record.update(k=k, blocks=[list(b) for b in blocks])
    timing["blocks"] = time.perf_counter()-t1

    # ---- root search
    inside = [C for C in cand if all(compatible(C, B) for B in blocks)]
    universe = None
    if k > k_max:
        record["root_search"] = f"skipped: {k} blocks > k_max {k_max}"
        family = canonical([*base, *start])
        record.update(root_unions=0)
    else:
        unions = root_unions(blocks, n)
        universe = canonical([*inside, *unions])
        record.update(root_unions=len(unions), universe=len(universe))
        t1 = time.perf_counter()
        fitter = rv.GramFitter(design, [*universe, *base], base, bound)
        prob = rv.Problem(fitter, universe, base, alpha)
        best_S, best = None, math.inf
        deadline = time.perf_counter()+search_seconds
        u = set(universe)
        for s0 in (frozenset(C for C in start if C in u), frozenset()):     # laminar starts only
            Sx, sx = rv.local_search(prob, s0, deadline)
            if sx < best:
                best_S, best = Sx, sx
        timing["search"] = time.perf_counter()-t1
        t1 = time.perf_counter()
        rec = rv.select_miqp(design, universe, base, alpha, bound, miqp_seconds, warm=canonical([*base, *best_S]),
                             seed=seed, gap=1e-4, cutoff=best, solver_cutoff=False)
        timing["root_miqp"] = time.perf_counter()-t1
        fam = rec.pop("family", None)
        record["root_miqp"] = {kk: rec[kk] for kk in ("status", "objective", "bound", "gap", "nodes", "binaries", "seconds")}
        record["search_value"] = best
        if fam is not None and prob.score(frozenset(C for C in fam if 1 < len(C) < n)) < best-1e-12:
            best_S = frozenset(C for C in fam if 1 < len(C) < n)
            record["root_miqp"]["improved_search"] = True
        family = canonical([*base, *best_S])
        if not trees.is_laminar(family):
            raise RuntimeError("root search returned a non-laminar family")

    # ---- certificate and repair over the whole tree
    t1 = time.perf_counter()
    certs, pair_moves = [], []
    def single_repair(family):
        for _ in range(repairs+1):
            current = crit(family)
            c = mx.certify_by_nodes(design, family, alpha, bound, node_seconds)
            add = None
            if c["best"] is not None and c["best"]["supports"]:
                C = tuple(sorted(c["best"]["supports"][0]))
                new = canonical([*family, C])
                if crit(new) < current-1e-9:
                    add, family = C, new
            certs.append(dict(proved=c["proved"], seconds=c["seconds"], nodes=len(c["nodes"]),
                              added=None if add is None else list(add)))
            if add is None:
                break
        return family
    family = single_repair(family)
    for _ in range(pair_rounds if pairs else 0):
        # two lines at once inside one node (nested or disjoint unions of its child blocks): the moves a single-line
        # repair cannot make when neither line pays alone
        current, best_new, best_val, best_rec = crit(family), None, math.inf, None
        for node, width in mx.node_splits(family, n):
            if width < 4:
                continue
            r = mx.extend_tf_miqp(design, family, alpha, bound, node_seconds, k=2, within=node, blocks=True)
            if len(r.get("supports") or []) == 2:
                new = canonical([*family, *(tuple(sorted(s)) for s in r["supports"])])
                if trees.is_laminar(new):
                    val = crit(new)
                    if val < current-1e-9 and val < best_val:
                        best_new, best_val, best_rec = new, val, dict(node=list(node), status=r["status"],
                                                                       added=[list(s) for s in r["supports"]])
        if best_new is None:
            break
        pair_moves.append(best_rec)
        family = single_repair(best_new)
    pool_base = universe if universe is not None else cand
    added = [C for C in family if 1 < len(C) < n and C not in set(pool_base)]
    if reselect and added:
        # re-decide with the enlarged candidate set, so lines the repair added can replace earlier choices
        pool2 = canonical([*pool_base, *(C for C in family if 1 < len(C) < n)])
        rec2 = rv.select_miqp(design, pool2, base, alpha, bound, miqp_seconds, warm=family, seed=seed, gap=1e-4,
                              cutoff=crit(family), solver_cutoff=False)
        fam2 = rec2.get("family")
        record["reselect"] = dict(status=rec2["status"], pool=len(pool2), seconds=rec2["seconds"])
        if fam2 is not None and trees.is_laminar(fam2) and crit(fam2) < crit(family)-1e-9:
            record["reselect"]["changed"] = True
            family = single_repair(canonical(fam2))
    record["pair_moves"] = pair_moves
    timing["certificate"] = time.perf_counter()-t1
    timing["total"] = time.perf_counter()-t0
    record.update(certificates=certs, certified=bool(certs and certs[-1]["proved"] and certs[-1]["added"] is None),
                  criterion=crit(family))
    return dict(selected=family, record=record, timing=timing, alpha=alpha)
