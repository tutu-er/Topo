"""Exact MIQP components added to WZZT-R for experiment M (PROTOCOL_MIQP.md); revised.py is unchanged.

extend_k_miqp generalises WZZT's unknown-support MIQP (`revised.extend_miqp`):
- k = 1 is the original single-support extension (the W step and its no-gain certificate);
- k = 2 is W2: two unknown supports chosen jointly, each compatible with the family and with each other. Under
  the slot term, disjoint supports have negatively correlated columns, so their gains can be super-additive and
  every one-at-a-time search (generators, beam, local search, the single-support MIQP) can miss a pair;
- within = v restricts the new supports to proper subsets of the node v (the certified resolution report);
- force = True makes every new support active (charged or not), so the optimum is the best split itself and
  the solver's bound is a certified upper bound on the evidence any such split can add.

The supports' predictions are linear in u_ij = r_new z_i z_j (and likewise for x), linearised exactly with
z z' and big-M; the rows are the SVD-compressed columns of the family and of every symmetric pair (i <= j).
"""
from __future__ import annotations

import math
import time

import numpy as np

from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.identification.trees import basis, canonical, children, compatible


def extend_k_miqp(design, family, alpha, bound, seconds, *, k=1, within=None, force=False, starts=None,
                  seed=20261111, numeric_focus=2, gap=1e-4, abs_gap=1e-7, threads=1):
    gp = rv._gurobi()
    started = time.perf_counter()
    n, family = design.n, canonical(family)
    proper = [C for C in family if 1 < len(C) < n]
    K = len(family)
    allowed = list(range(n)) if within is None else sorted(within)
    if len(allowed) < 3:
        return dict(status=None, reason="node too small", supports=[], seconds=0.)
    # variables only for the meters a new support may contain (all meters, or the node's): pairs i <= j in it
    na = len(allowed)
    Fp, (ii, jj) = _pair_columns(design, allowed)
    mpairs = len(ii)
    B, uy, const, rank = rv._svd_rows(np.hstack((design.columns(family), Fp)), design.y)
    flat = ii*na+jj
    pos = {i: a for a, i in enumerate(allowed)}
    fit = rv.exact_fit(design, family, bound)
    r0, x0 = fit["theta"][:K], fit["theta"][K:]
    env = rv._env(gp)
    try:
        with gp.Model("WZZT_k_unknown_supports", env=env) as m:
            rv._params(m, seconds, seed, numeric_focus, gap, abs_gap)
            m.Params.Threads = threads
            r = m.addMVar(K, lb=0., ub=bound); x = m.addMVar(K, lb=0., ub=bound)
            Z, A, URs, UXs = [], [], [], []
            for s in range(k):
                z = m.addMVar(na, vtype="B"); a = m.addVar(vtype="B")
                nr, nx = m.addVar(lb=0., ub=bound), m.addVar(lb=0., ub=bound)
                m.addConstr(nr <= bound*a); m.addConstr(nx <= bound*a)
                if force:
                    m.addConstr(a == 1)
                m.addConstr(z.sum() >= 2); m.addConstr(z.sum() <= na-1)
                for C in proper:
                    inside = [pos[i] for i in C if i in pos]
                    outside = [pos[i] for i in allowed if i not in set(C)]
                    if not inside or not outside:
                        continue        # disjoint from the node, or containing it: compatible and distinct
                    rel = m.addMVar(3, vtype="B")
                    m.addConstr(rel.sum() == 1)
                    m.addConstr(z[inside].sum() <= len(inside)*(1-rel[0]))      # disjoint from C
                    m.addConstr(z[outside].sum() <= len(outside)*(1-rel[1]))    # inside C
                    m.addConstr(z[inside].sum() >= len(C)*rel[2])               # contains C
                    m.addConstr(len(C)-z[inside].sum()+z[outside].sum() >= 1)   # differs from C
                g = m.addMVar((na, na), lb=0., ub=1.)
                ur = m.addMVar((na, na), lb=0., ub=bound); ux = m.addMVar((na, na), lb=0., ub=bound)
                m.addConstr(g <= z[:, None]); m.addConstr(g <= z[None, :])
                m.addConstr(g >= z[:, None]+z[None, :]-1)
                for weights, scalar in ((ur, nr), (ux, nx)):
                    m.addConstr(weights <= scalar); m.addConstr(weights <= bound*g)
                    m.addConstr(weights >= scalar-bound*(1-g))
                Z.append(z); A.append(a); URs.append(ur); UXs.append(ux)
            if k == 2:
                z1, z2 = Z
                both = m.addVar(vtype="B")
                m.addConstr(both <= A[0]); m.addConstr(both <= A[1]); m.addConstr(both >= A[0]+A[1]-1)
                m.addConstr(A[0] >= A[1])                                     # symmetry: the first is used first
                m.addConstr(z1.sum() >= z2.sum()-n*(1-both))                  # symmetry: larger first
                rel = m.addMVar(3, vtype="B")
                m.addConstr(rel.sum() >= both)
                m.addConstr(z1+z2 <= 1+(1-rel[0]))                            # disjoint
                m.addConstr(z1 <= z2+(1-rel[1]))                              # z1 inside z2
                m.addConstr(z2 <= z1+(1-rel[2]))                              # z2 inside z1
                d = m.addMVar(na, lb=0., ub=1.)
                m.addConstr(d <= z1+z2); m.addConstr(d <= 2-z1-z2)
                m.addConstr(d.sum() >= both)                                  # distinct when both are used
            elif k != 1:
                raise ValueError("k must be 1 or 2")
            urf = sum(u.reshape(-1)[flat] for u in URs)
            uxf = sum(u.reshape(-1)[flat] for u in UXs)
            res = m.addMVar(rank, lb=-float("inf"))
            m.addConstr(res == B[:, :K]@r+B[:, K:2*K]@x+B[:, 2*K:2*K+mpairs]@urf+B[:, 2*K+mpairs:]@uxf-uy)
            m.setObjective((res@res+const)/design.N+alpha*(len(proper)+sum(A)))
            for s, C in enumerate(starts or []):          # a start only with a branch choice (see extend_tf_miqp)
                if s < k and C is not None and all(i in pos for i in C):
                    zz = np.zeros(na); zz[[pos[i] for i in C]] = 1.
                    Z[s].Start = zz; A[s].Start = 1.
                    r.Start, x.Start = r0, x0
            m.optimize()
            feasible = m.SolCount > 0
            out = dict(status=int(m.Status), objective=rv._finite(m.ObjVal) if feasible else None,
                       bound=rv._finite(m.ObjBound), gap=rv._finite(m.MIPGap) if feasible else None,
                       nodes=float(m.NodeCount), rank=rank, k=k, within=None if within is None else list(allowed),
                       force=force, seconds=time.perf_counter()-started,
                       family_objective=fit["objective"]+alpha*len(proper), supports=[])
            if feasible:
                for s in range(k):
                    if A[s].X > .5:
                        C = tuple(sorted(allowed[int(i)] for i in np.flatnonzero(Z[s].X > .5)))
                        if C in family or not 2 <= len(C) < n or not all(compatible(C, D) for D in family):
                            raise RuntimeError(f"Invalid decoded support {C}")
                        out["supports"].append(C)
                if len(out["supports"]) == 2 and not compatible(*out["supports"]):
                    raise RuntimeError("Decoded supports are not compatible with each other")
    finally:
        env.dispose()
    return out


def extend_tf_miqp(design, family, alpha, bound, seconds, *, k=1, within=None, force=False, starts=None,
                   seed=20261111, numeric_focus=2, gap=1e-4, abs_gap=1e-7, threads=1, ub=None, mc="scaled",
                   blocks=False, cutoff=None, log=None, params=None, compress=None):
    """The same model and optimum as extend_k_miqp, in a time-free form built on the R/X map (scalable).

    WZZT's prediction is Y^ = P R + Q X with the symmetric maps R = sum_C r_C z_C z_C' (and X); the transformed
    fitting error is ||(Y - P R - Q X) S_c||^2 (time-demeaned flows, S_c = I - c 11'/n over the meters). With
    Phi = [P Q] and theta~_m the m-th column of [R; X] S_c, it is sum_m ||y~_m - Phi theta~_m||^2
      = const - 2 sum_m (Phi' y~_m)' theta~_m + sum_m || L' theta~_m ||^2,     L L' = Phi' Phi  (2n x 2n),
    so the data enter only through Phi'Phi, Phi'Y~ and ||Y~||^2: the model size does not depend on the history
    length, and the quadratic part is block-diagonal over meters (O(n^3) nonzeros) instead of the dense
    SVD-compressed rows of every pair column (O(n^4)). theta~ is linear in the old coefficients and in the new
    supports' entries u_ij = r z_i z_j (upper triangle only; u_ii = r z_i), linearised exactly as before.
    ub: a valid upper bound on the new supports' coefficients (tighter big-M); default `bound`.
    mc: "scaled" (default) linearises the off-diagonal entries through the diagonal ones, d_i = r z_i:
        u_ij <= d_i, u_ij <= d_j, u_ij >= d_i + d_j - r (exact at binary z, no z_i z_j variables; its relaxation
        is the envelope of products of fractional memberships d_i / r); "g" is the original big-M form through
        g_ij = z_i z_j (u_ij <= M g_ij, u_ij >= r - M (1 - g_ij)).
    blocks: with `within` a node of the family (or the root), one binary per child block of the node instead of one
        per meter: every compatible split of a node is a union of its child blocks (and no union of two or more
        blocks equals an existing support), so no compatibility constraints are needed and the discrete part has
        k_v binaries and k_v (k_v + 1) / 2 map entries per part.
    cutoff: only solutions with objective below this value are of interest (Gurobi Cutoff).
    """
    import scipy.sparse as sp
    gp = rv._gurobi()
    started = time.perf_counter()
    n, family = design.n, canonical(family)
    proper = [C for C in family if 1 < len(C) < n]
    K = len(family)
    node = tuple(range(n)) if within is None else tuple(sorted(within))
    if blocks:
        kids = children(family, n)
        if node not in kids:
            raise ValueError("blocks=True needs `within` to be a node of the family")
        groups = [tuple(b) for b in kids[node]]
    else:
        groups = [(i,) for i in node]
    allowed = list(range(len(groups)))                    # local units: meters, or the node's child blocks
    na = len(allowed)
    if na < 3:
        return dict(status=None, reason="node too small", supports=[], seconds=0.)
    unit = {i: a for a, grp in enumerate(groups) for i in grp}       # meter -> local unit
    pos = {i: unit[i] for i in node}
    proper_units = []
    if not blocks:
        for C in proper:
            proper_units.append((C, [unit[i] for i in C if i in unit], [unit[i] for i in node if i not in set(C)]))
    c = design.t.c
    M = float(bound if ub is None else min(ub, bound))
    # ---- sufficient statistics (scaled units)
    Phi = np.hstack((design.Pt, design.Qt))/design.scale
    Yt = design.Yt
    lam, V = np.linalg.eigh(Phi.T@Phi)
    keep = lam > max(float(lam.max()), 0.)*1e-13
    Lt = (V[:, keep]*np.sqrt(lam[keep])).T                      # rg x 2n, Lt'Lt = Phi'Phi
    rg = Lt.shape[0]
    bvec = (Phi.T@Yt).T.reshape(-1)                             # index m*2n + j
    const = float(np.sum(Yt**2))
    # ---- theta~ = A_th th + A_u u + A_s sigma, sigma_j = row sums of [R; X] (index m*2n + j)
    iu, ju = np.triu_indices(na)
    tri = {(int(a), int(b)): t for t, (a, b) in enumerate(zip(iu, ju))}
    npair = len(iu)
    tidx = lambda a, b: tri[(a, b) if a <= b else (b, a)]
    rows, cols, vals = [], [], []
    for ci, C in enumerate(family):
        for jp in C:
            for mm in C:
                rows += [mm*2*n+jp, mm*2*n+n+jp]; cols += [ci, K+ci]; vals += [1., 1.]
    A_th = sp.csr_matrix((vals, (rows, cols)), shape=(2*n*n, 2*K))
    rows, cols = [], []
    for jp in node:
        for mm in node:
            t = tidx(unit[jp], unit[mm])
            rows += [mm*2*n+jp, mm*2*n+n+jp]; cols += [t, npair+t]
    A_u = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2*n*n, 2*npair))
    A_s = sp.csr_matrix((np.full(2*n*n, -c/n), (np.arange(2*n*n), np.tile(np.arange(2*n), n))), shape=(2*n*n, 2*n))
    rows, cols, vals = [], [], []
    for ci, C in enumerate(family):
        for jp in C:
            rows += [jp, n+jp]; cols += [ci, K+ci]; vals += [float(len(C)), float(len(C))]
    S_th = sp.csr_matrix((vals, (rows, cols)), shape=(2*n, 2*K))
    rows, cols = [], []
    for jp in node:
        for mm in node:
            t = tidx(unit[jp], unit[mm])
            rows += [jp, n+jp]; cols += [t, npair+t]
    S_u = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(2*n, 2*npair))
    Lk = sp.kron(sp.identity(n, format="csr"), sp.csr_matrix(Lt), format="csr")
    # compressed form: theta~ = A_full x with x = [old coefficients; new map entries] (sigma folded in); the
    # objective is ||Lk A_full x||^2 - 2 (A_full' b)' x + const = ||Rq x||^2 - ... with Rq from a QR of Lk A_full
    # (d x d, d = 2K + 2 npair): tiny when the new supports live on a few child blocks
    d = 2*K+2*npair
    use_compress = compress if compress is not None else (blocks or d <= 600)
    if use_compress:
        A_full = sp.hstack((A_th+A_s@S_th, A_u+A_s@S_u), format="csr")
        Mx = (Lk@A_full).toarray()
        Rq = np.linalg.qr(Mx, mode="r")
        lin = A_full.T@bvec
    fit = rv.exact_fit(design, family, bound)
    th0 = fit["theta"]
    env = rv._env(gp)
    try:
        with gp.Model("WZZT_k_unknown_supports_timefree", env=env) as m:
            rv._params(m, seconds, seed, numeric_focus, gap, abs_gap)
            m.Params.Threads = threads
            th = m.addMVar(2*K, lb=0., ub=bound)
            Z, A, Us, NR, NX = [], [], [], [], []
            for s in range(k):
                z = m.addMVar(na, vtype="B"); a = m.addVar(vtype="B")
                nr, nx = m.addVar(lb=0., ub=M), m.addVar(lb=0., ub=M)
                m.addConstr(nr <= M*a); m.addConstr(nx <= M*a)
                if force:
                    m.addConstr(a == 1)
                if blocks:
                    m.addConstr(z.sum() >= 2); m.addConstr(z.sum() <= na-1)     # 2..k-1 child blocks
                else:
                    m.addConstr(z.sum() >= 2); m.addConstr(z.sum() <= na-1)
                for C, inside, outside in proper_units:
                    if not inside or not outside:
                        continue
                    rel = m.addMVar(3, vtype="B")
                    m.addConstr(rel.sum() == 1)
                    m.addConstr(z[inside].sum() <= len(inside)*(1-rel[0]))
                    m.addConstr(z[outside].sum() <= len(outside)*(1-rel[1]))
                    m.addConstr(z[inside].sum() >= len(C)*rel[2])
                    m.addConstr(len(C)-z[inside].sum()+z[outside].sum() >= 1)
                off = iu != ju
                u = m.addMVar(2*npair, lb=0., ub=M)                      # [r part; x part], upper triangle
                dg = np.flatnonzero(~off); od = np.flatnonzero(off)
                diag_of = {int(iu[t]): int(t) for t in dg}               # meter (local) -> its diagonal entry
                if mc == "g":
                    g = m.addMVar(int(off.sum()), lb=0., ub=1.)          # z_i z_j, i < j
                    zi, zj = z[iu[off]], z[ju[off]]
                    m.addConstr(g <= zi); m.addConstr(g <= zj); m.addConstr(g >= zi+zj-1)
                for base, scalar in ((0, nr), (npair, nx)):
                    ud, uo = u[base+dg], u[base+od]
                    m.addConstr(ud <= scalar); m.addConstr(ud <= M*z[iu[dg]])
                    m.addConstr(ud >= scalar-M*(1-z[iu[dg]]))
                    if mc == "g":
                        m.addConstr(uo <= scalar); m.addConstr(uo <= M*g); m.addConstr(uo >= scalar-M*(1-g))
                    else:
                        di = u[base+np.array([diag_of[int(i)] for i in iu[od]], dtype=int)]
                        dj = u[base+np.array([diag_of[int(j)] for j in ju[od]], dtype=int)]
                        m.addConstr(uo <= di); m.addConstr(uo <= dj); m.addConstr(uo >= di+dj-scalar)
                Z.append(z); A.append(a); Us.append(u); NR.append(nr); NX.append(nx)
            if k == 2:
                z1, z2 = Z
                both = m.addVar(vtype="B")
                m.addConstr(both <= A[0]); m.addConstr(both <= A[1]); m.addConstr(both >= A[0]+A[1]-1)
                m.addConstr(A[0] >= A[1]); m.addConstr(z1.sum() >= z2.sum()-na*(1-both))
                rel = m.addMVar(3, vtype="B")
                m.addConstr(rel.sum() >= both)
                m.addConstr(z1+z2 <= 1+(1-rel[0])); m.addConstr(z1 <= z2+(1-rel[1])); m.addConstr(z2 <= z1+(1-rel[2]))
                d = m.addMVar(na, lb=0., ub=1.)
                m.addConstr(d <= z1+z2); m.addConstr(d <= 2-z1-z2); m.addConstr(d.sum() >= both)
            elif k != 1:
                raise ValueError("k must be 1 or 2")
            U = Us[0] if k == 1 else Us[0]+Us[1]
            if use_compress:
                rho = m.addMVar(Rq.shape[0], lb=-float("inf"))
                m.addConstr(rho == Rq[:, :2*K]@th+Rq[:, 2*K:]@U)
                m.setObjective((rho@rho-2*(lin[:2*K]@th+lin[2*K:]@U)+const)/design.N+alpha*(len(proper)+sum(A)))
            else:
                sig = m.addMVar(2*n, lb=-float("inf"))
                m.addConstr(sig == S_th@th+S_u@U)
                tt = m.addMVar(2*n*n, lb=-float("inf"))
                m.addConstr(tt == A_th@th+A_u@U+A_s@sig)
                rho = m.addMVar(rg*n, lb=-float("inf"))
                m.addConstr(rho == Lk@tt)
                m.setObjective((rho@rho-2*(bvec@tt)+const)/design.N+alpha*(len(proper)+sum(A)))
            # MIP start only with a branch choice: a start holding only continuous values makes Gurobi complete it
            # by a sub-MIP as hard as the problem (it took the whole time limit at 26 meters)
            # A complete start (Gurobi only checks it): the exact refit of the family plus the first start support,
            # every map entry, and the auxiliary rows. An incomplete start is completed by a sub-MIP that can fail
            # on large models (it did at 34 meters).
            C0 = (starts or [None])[0]
            if C0 is not None and all(i in pos for i in C0):
                units = sorted({unit[i] for i in C0})
                if sum(len(groups[a]) for a in units) == len(C0) and C0 not in family:
                    fam1 = canonical([*family, tuple(sorted(C0))])
                    th1 = rv.exact_fit(design, fam1, bound)["theta"]
                    K1 = len(fam1)
                    where = [fam1.index(D) for D in family]
                    thv = np.r_[th1[where], th1[[K1+w for w in where]]]
                    c_new = fam1.index(tuple(sorted(C0)))
                    rn, xn = float(min(th1[c_new], M)), float(min(th1[K1+c_new], M))
                    member = np.zeros(na, bool); member[units] = True
                    inside = member[iu] & member[ju]
                    Uv = np.r_[np.where(inside, rn, 0.), np.where(inside, xn, 0.)]
                    zz = member.astype(float)
                    for s in range(k):
                        Z[s].Start = zz
                    A[0].Start, NR[0].Start, NX[0].Start = 1., rn, xn
                    Us[0].Start = Uv
                    for s in range(1, k):
                        A[s].Start, NR[s].Start, NX[s].Start = 0., 0., 0.
                        Us[s].Start = np.zeros(2*npair)
                    th.Start = thv
                    if use_compress:
                        rho.Start = Rq[:, :2*K]@thv+Rq[:, 2*K:]@Uv
                    else:
                        sv = S_th@thv+S_u@Uv
                        tv = A_th@thv+A_u@Uv+A_s@sv
                        sig.Start, tt.Start, rho.Start = sv, tv, Lk@tv
            if cutoff is not None:
                m.Params.Cutoff = float(cutoff)
            if log:
                m.Params.OutputFlag = 1; m.Params.LogToConsole = 0; m.Params.LogFile = str(log)
            for key, value in (params or {}).items():
                m.setParam(key, value)
            built = time.perf_counter()-started
            m.optimize()
            feasible = m.SolCount > 0
            out = dict(status=int(m.Status), objective=rv._finite(m.ObjVal) if feasible else None,
                       bound=rv._finite(m.ObjBound), gap=rv._finite(m.MIPGap) if feasible else None,
                       nodes=float(m.NodeCount), k=k, within=None if within is None else list(node),
                       force=force, build_seconds=built, seconds=time.perf_counter()-started,
                       family_objective=fit["objective"]+alpha*len(proper), supports=[], form="timefree",
                       blocks=na if blocks else None, compressed=bool(use_compress),
                       size=dict(vars=m.NumVars, constrs=m.NumConstrs, nnz=m.NumNZs, rg=rg))
            if feasible:
                for s in range(k):
                    if A[s].X > .5:
                        C = tuple(sorted(i for a in np.flatnonzero(Z[s].X > .5) for i in groups[int(a)]))
                        if C in family or not 2 <= len(C) < n or not all(compatible(C, D) for D in family):
                            raise RuntimeError(f"Invalid decoded support {C}")
                        out["supports"].append(C)
                if len(out["supports"]) == 2 and not compatible(*out["supports"]):
                    raise RuntimeError("Decoded supports are not compatible with each other")
    finally:
        env.dispose()
    return out


def certify_by_nodes(design, family, alpha, bound, seconds_per_node, *, force=False, cutoff=None, max_blocks=None,
                     **kw):
    """The unknown-support certificate decomposed exactly over the nodes of `family`.

    Every compatible new support is a union of two or more (not all) child blocks of exactly one node, so the best
    single extension anywhere is the best over the nodes, and each node's MIQP has one binary per child block (the
    compressed time-free form: about 2K + k(k+1) continuous variables, whatever the number of meters). force=False:
    does any split pay its price (the no-gain certificate)? force=True: the largest evidence of a split per node.
    Returns the per-node results and the best support over the nodes."""
    family = canonical(family)
    out = []
    best = None
    for node, width in node_splits(family, design.n):
        if max_blocks is not None and width > max_blocks:
            out.append(dict(node=node, width=width, skipped="wider than max_blocks"))
            continue
        r = extend_tf_miqp(design, family, alpha, bound, seconds_per_node, k=1, within=node, force=force,
                           blocks=True, cutoff=cutoff, **kw)
        r["node"], r["width"] = node, width
        out.append(r)
        if r["supports"] and r["objective"] is not None and (best is None or r["objective"] < best["objective"]):
            best = r
    proved = all(r.get("status") == 2 or (r.get("status") == 6 and cutoff is not None) for r in out if "skipped" not in r)
    return dict(nodes=out, best=best, proved=proved, seconds=sum(r.get("seconds", 0.) for r in out))


def _pair_columns(design, allowed):
    """Columns of the symmetric entries (i <= j, both in `allowed`) of R and X under M: meter i gets p_j and meter
    j gets p_i. With allowed = every meter this is design.pair_columns()."""
    allowed = list(allowed)
    ia, ja = np.triu_indices(len(allowed))
    gi, gj = np.asarray(allowed)[ia], np.asarray(allowed)[ja]
    m = len(ia)
    out = []
    for flows in (design.Pt, design.Qt):
        F = np.zeros((design.T, design.n, m))
        F[:, gj, np.arange(m)] = flows[:, gi]
        F[:, gi, np.arange(m)] = flows[:, gj]
        out.append(design.t.slot(F).reshape(design.N, m))
    return np.hstack(out)/design.scale, (ia, ja)


def setup(train, validation=None, config=None):
    """WZZT-R's scoring set-up (as in revised.fit): transform, scaled design, guidance, noise scale and price."""
    config = config or rv.RevisedConfig()
    config.validate()
    as_series = lambda o: o if isinstance(o, rv.Series) else rv.Series(o.p, o.q, o.y, getattr(o, "v0sq", None))
    train = as_series(train)
    data = train.join(as_series(validation)) if (config.history == "all" and validation is not None) else train
    T, n = data.p.shape
    N = T*n
    if config.source == "gls":
        c = rv.root_gamma(data.v0sq, data.y)
    elif config.source == "estimated":
        c, _ = rv.estimated_gamma(data, config.offsets)
    else:
        c = 1. if config.source == "free" else 0.
    design = rv.Design(data, rv.Transform(T, n, c, config.offsets), 1.)
    guide = rv.guidance_ponly(design) if config.guidance == "ponly" else rv.guidance_pq(data, design.t)
    p_s, _ = rv.identifiable_per_support(data, config.pq_sigma)
    sigma2, _ = rv.saturated_sigma2(design)
    alpha = p_s*math.log(N)/N
    design.rescale(math.sqrt(sigma2))
    return dict(design=design, guide=guide, alpha=alpha, sigma2=sigma2, p_s=p_s, c=c, N=N, n=n, T=T, config=config)


def node_splits(family, n):
    """Nodes (the root and every internal support) with at least three child blocks: where a split can go."""
    return [(tuple(sorted(node)), len(blocks)) for node, blocks in children(canonical(family), n).items()
            if len(blocks) >= 3]
