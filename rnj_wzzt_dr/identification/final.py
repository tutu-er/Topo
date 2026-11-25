"""The method for the sealed test: WZZT-C with a structural prior and, when the PCC is metered, the located source (S2).

Differences from the frozen WZZT-C (combo.py, which the running confirmation uses and which is not edited):
1. **Structural prior.** The price of a line is BIC minus `lam` half-BIC units. `lam` = ln(1/rho), rho = the odds of a node
   having more than two children on real feeders (geometric prior on the number of children above two; the sum over
   the nodes is linear in the number of lines, so the MIQPs and certificates keep their form).
2. **Two searches in the pool step.** A multi-start local search and the exact pool MIQP (warm-started and cut off up to
   `warm_n` meters, cold beyond: with many lines the warm-started solver found no solution in time). The better tree is kept
   and polished by local search. Beyond 35 meters the MIQP limits of WZZT-C (240 s) are used.
3. **Completion of S2.** After S2's MIQP every union of child blocks of every node is priced; the best paying line is added to the
   pool and the MIQP decides again, until no line pays (source_cert.complete).
4. The source is projected out (S1) before the tree search, then located (S2).
5. **Price continuation beyond `warm_n` meters.** The tree is first found at the plain BIC price and then used as a start of the search at
   the lower price, so the bonus can only improve the score of a tree that was already good.

`tree` follows combo.fit's "margin" pipeline step by step, with the pool MIQP warm-started; with lam = 0 and no warm
start it returns combo.fit's trees (test_final.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from . import combo, miqp_ext as mx, revised as rv, source as src, source_cert as cert, trees
from .trees import basis, canonical, compatible

LAMBDA = 1.96               # half-BIC units: ln(1/rho), rho = 0.141 on 84 real feeders (IDENTIFICATION_PLAN.md)
SOURCES = ("rnj", "nj", "rg", "volt", "volt_m", "resid", "rrows", "upgma", "ward")


@dataclass(frozen=True)
class Settings:
    lam: float = LAMBDA
    m: float = 4.6                  # evidence margin of a block line (half-BIC units)
    k_max: int = 9                  # root search only up to this many blocks
    pool_seconds: float = 120.
    root_seconds: float = 120.
    node_seconds: float = 60.
    search_seconds: float = 30.
    warm_seconds: float = 20.       # local search per start, pool MIQP
    repairs: int = 10
    pair_rounds: int = 3
    s2_seconds: float = 120.
    complete_rounds: int = 10
    max_blocks: int = 12
    seed: int = 7
    warm: bool = True
    warm_n: int = 35                # beyond this many meters the pool MIQP starts cold, with the longer limit
    long_seconds: float = 240.      # pool and root limit beyond warm_n meters
    polish_seconds: float = 20.
    continuation: bool = True       # beyond warm_n meters: solve at lam = 0 first and start from that tree


def guarded(make, *args):
    """A set-up whose ordered QP ends with a numerical failure is repeated with the SciPy solver (SLSQP)."""
    try:
        return make(*args)
    except RuntimeError as exc:
        if "Ordered QP" not in str(exc):
            raise
    original = rv.ordered_sensitivity
    rv.ordered_sensitivity = lambda design, solver="slsqp": original(design, "slsqp")
    try:
        return make(*args)
    finally:
        rv.ordered_sensitivity = original


def priced(S, lam):
    """The set-up with the line price lowered by `lam` half-BIC units."""
    alpha = S["alpha"]-2*lam/S["N"]
    if alpha <= 0:
        raise ValueError("the structural prior is larger than the BIC price")
    return dict(S, alpha=alpha)


def multistart(prob, fams, seconds):
    """Local search from the empty tree and from the first three sources' trees; the best."""
    best = None
    for start in [frozenset()]+[frozenset(C for C in F if C in prob.crosses) for F in fams[:3]]:
        x = rv.local_search(prob, start, time.perf_counter()+seconds)
        best = x if best is None or x[1] < best[1] else best
    return best


def tree(S, st=Settings(), starts=()):
    """The tree of WZZT-C under the prices of S (S["alpha"] already includes the prior). Returns (family, record).

    starts: further families to start the pool search from (the plain-price tree, in the continuation)."""
    design, alpha, n, N, bound = S["design"], S["alpha"], S["n"], S["N"], 2.
    base = canonical(basis(n))
    crit = lambda fam: rv.exact_fit(design, fam, bound)["objective"]+alpha*rv.internal_count(fam, n)
    t0 = time.perf_counter()
    fams = [combo.source_family(S, s) for s in SOURCES]
    cand = canonical(C for F in fams for C in F if 1 < len(C) < n)
    rec = dict(candidates=len(cand))
    # ---- T*: the exact best tree over every source's lines, warm-started
    fitter = rv.GramFitter(design, [*cand, *base], base, bound)
    prob0 = rv.Problem(fitter, cand, base, alpha)
    kw, short = {}, n <= st.warm_n
    solutions = []
    if st.warm:
        ls_lines, ls_value = multistart(prob0, fams, st.warm_seconds if short else 2*st.warm_seconds)
        solutions.append(frozenset(ls_lines))
        if short:
            kw = dict(warm=canonical([*base, *ls_lines]), cutoff=ls_value)
    r0 = rv.select_miqp(design, cand, base, alpha, bound, st.pool_seconds if short else st.long_seconds, seed=st.seed,
                        gap=1e-4, **kw)
    if r0.get("family"):
        solutions.append(frozenset(C for C in r0["family"] if 1 < len(C) < n))
    for F in starts:
        solutions.append(frozenset(C for C in F if C in prob0.crosses))
    if not solutions:
        solutions.append(frozenset())
    kept = min(solutions, key=prob0.score)
    if st.warm:                                                  # polish the better one
        kept, _ = rv.local_search(prob0, kept, time.perf_counter()+st.polish_seconds)
    kept = list(kept)
    rec["pool"] = dict(status=r0["status"], seconds=r0["seconds"], gap=r0.get("gap"))
    # ---- blocks by evidence margin
    ev = combo.line_evidence(prob0, kept, cand, N)
    margin = {C: e["margin"] for C, e in ev.items()}
    blocks = combo.margin_blocks(kept, margin, st.m, n, 1.)
    rec["evidence"] = [dict(line=list(C), margin=round(e["margin"], 3)) for C, e in ev.items()]
    k = len(blocks)
    # ---- root search
    start = frozenset(kept)
    inside = [C for C in cand if all(compatible(C, B) for B in blocks)]
    if k > st.k_max:
        family = canonical([*base, *start])
    else:
        unions = combo.root_unions(blocks, n)
        universe = canonical([*inside, *unions])
        prob = rv.Problem(rv.GramFitter(design, [*universe, *base], base, bound), universe, base, alpha)
        best_S, best = None, math.inf
        u = set(universe)
        deadline = time.perf_counter()+st.search_seconds
        for s0 in (frozenset(C for C in start if C in u), frozenset()) if short else (frozenset(C for C in start if C in u),):
            Sx, sx = rv.local_search(prob, s0, deadline)
            if sx < best:
                best_S, best = Sx, sx
        r1 = rv.select_miqp(design, universe, base, alpha, bound, st.root_seconds if short else st.long_seconds,
                            warm=canonical([*base, *best_S]),
                            seed=st.seed, gap=1e-4, cutoff=best, solver_cutoff=False)
        fam = r1.get("family")
        if fam is not None and prob.score(frozenset(C for C in fam if 1 < len(C) < n)) < best-1e-12:
            best_S = frozenset(C for C in fam if 1 < len(C) < n)
        family = canonical([*base, *best_S])
        rec["root"] = dict(status=r1["status"], seconds=r1["seconds"], unions=len(unions))
        if not trees.is_laminar(family):
            raise RuntimeError("root search returned a non-laminar family")
    # ---- certificate and repair
    certs = []

    def single_repair(family):
        for _ in range(st.repairs+1):
            current = crit(family)
            c = mx.certify_by_nodes(design, family, alpha, bound, st.node_seconds)
            add = None
            if c["best"] is not None and c["best"]["supports"]:
                C = tuple(sorted(c["best"]["supports"][0]))
                new = canonical([*family, C])
                if crit(new) < current-1e-9:
                    add, family = C, new
            certs.append(dict(proved=c["proved"], added=None if add is None else list(add)))
            if add is None:
                break
        return family
    family = single_repair(family)
    for _ in range(st.pair_rounds):
        current, best_new, best_val = crit(family), None, math.inf
        for node, width in mx.node_splits(family, n):
            if width < 4:
                continue
            r = mx.extend_tf_miqp(design, family, alpha, bound, st.node_seconds, k=2, within=node, blocks=True)
            if len(r.get("supports") or []) == 2:
                new = canonical([*family, *(tuple(sorted(s)) for s in r["supports"])])
                if trees.is_laminar(new):
                    val = crit(new)
                    if val < current-1e-9 and val < best_val:
                        best_new, best_val = new, val
        if best_new is None:
            break
        family = single_repair(best_new)
    universe_all = canonical([*cand, *(C for C in family if 1 < len(C) < n)])
    if any(1 < len(C) < n and C not in set(cand) for C in family):
        r2 = rv.select_miqp(design, universe_all, base, alpha, bound, st.pool_seconds if short else st.long_seconds, warm=family, seed=st.seed, gap=1e-4,
                            cutoff=crit(family), solver_cutoff=False)
        fam2 = r2.get("family")
        if fam2 is not None and trees.is_laminar(fam2) and crit(fam2) < crit(family)-1e-9:
            family = single_repair(canonical(fam2))
    rec.update(certificates=certs, certified=bool(certs and certs[-1]["proved"] and certs[-1]["added"] is None),
               blocks=k, seconds=time.perf_counter()-t0)
    return family, rec


def start_tree(S0, st, n):
    """The tree at the plain price, when the continuation applies (one more family to start from)."""
    if not (st.continuation and st.lam > 0 and n > st.warm_n):
        return ()
    plain = replace(st, lam=0., continuation=False)
    family, _ = tree(priced(S0, 0.), plain)
    return (family,)


def fit(train, validation, ppcc=None, qpcc=None, st=Settings(), config=None):
    """Tree without a PCC (blind), or tree and source with it.

    Returns dict(tree, record) and, with the PCC, dict(located, path, places, source_evidence, s2) as well."""
    started = time.perf_counter()
    config = config or rv.RevisedConfig()
    data = train.join(validation)
    n, T, bound = data.n, data.p.shape[0], 2.
    if ppcc is None:
        S0 = guarded(combo.setup, train, validation, config)
        S = priced(S0, st.lam)
        family, rec = tree(S, st, start_tree(S0, st, n))
        return dict(tree=family, record=rec, S=S, seconds=time.perf_counter()-started)
    up, uq = src.balance(ppcc, qpcc, data)
    S0 = guarded(src.setup_projected, train, validation, up, uq, config)
    S = priced(S0, st.lam)
    family, rec = tree(S, st, start_tree(S0, st, n))                              # S1
    out = dict(tree=family, record=rec, S=S, u=(up, uq), seconds_s1=time.perf_counter()-started)
    transform = rv.Transform(T, n, S["c"], config.offsets)
    scale = math.sqrt(S["sigma2"])
    design = src.LeafDesign(data, transform, scale, up, uq)
    empty = src.LeafDesign(data, transform, scale, 0*up, 0*uq)
    best_place = src.locate(design, family, n, bound)[0][1]
    start = src.attach(family, n, best_place)
    lines = canonical([C for s in SOURCES for C in combo.source_family(S, s) if 1 < len(C) < n]
                      + [C for C in family if 1 < len(C) < n])
    r = src.select_with_source(design, lines, S["alpha"], bound, st.s2_seconds, n, warm=start, seed=st.seed)
    fam = canonical(r.get("family") or start)
    pool = [C for C in src.with_source(lines, n) if 1 < len(C) < n+1]
    fam, crec = cert.complete(design, fam, pool, S["alpha"], bound, n, rounds=st.complete_rounds,
                              seconds=st.s2_seconds, max_blocks=st.max_blocks, seed=st.seed)
    tree_s2, path = src.meters_only(fam, n)
    half = .5*S["N"]
    score = lambda d, f: rv.exact_fit(d, f, bound)["objective"]+S["alpha"]*rv.internal_count(f, n+1)
    places = src.locate(design, tree_s2, n, bound)
    out.update(located=tree_s2, path=path, places=[(half*(o-places[0][0]), D) for o, D in places],
               source_evidence=half*(score(empty, src.attach(tree_s2, n, None))-score(design, fam)),
               s2=dict(status=r["status"], complete=crec), seconds=time.perf_counter()-started)
    return out
