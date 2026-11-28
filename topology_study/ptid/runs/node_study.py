"""Shared pieces of the node MIQP study (experiments/node_miqp/README.md).

A feeder here has one wide junction: the trunk (PCC bus 0 to bus 1) feeds `blocks` laterals, each serving `per`
meters. Optionally a hidden shared segment (bus 1 to bus 2, resistance r_shared) feeds the laterals listed in `shared`.
The meters' groupings are the blocks and, with the segment, the union of the shared blocks. The data are an AC power
flow at four quarter-hours per hour, averaged to hourly readings with meter gain and noise errors.

Everything is deterministic in the feeder's seed: the network, the loads (history and the extra probing day) and the
meter errors. The noise of every hour has its own random stream, so two runs that differ only in a probing block see the
same noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np

from ptid.cases import physics
from ptid.cases.paths import OUTPUTS
from ptid.method.model import Fitter, Model, balance, load_share, GATE
from ptid.method.search import node_miqp
from ptid.method.sets import basis, canonical, children, unions

SEED = 20261128            # study key; every random stream starts from it
BASE_KV, BASE_MVA = .4, .2
BASE_KW = BASE_MVA*1000.
TRUNK = (.016, .008)       # ohm
M = 4.6                    # margin of decisions, score units (as in Algorithm 1)
RESULTS = OUTPUTS/"topology_study"/"node_miqp"


@dataclass(frozen=True)
class Feeder:
    blocks: int = 12              # children of the wide junction
    per: int = 2                  # meters per block
    shared: tuple = ()            # blocks fed through the hidden segment; () = no segment
    r_shared: float = 0.          # ohm (x = 0.4 r)
    excitation: float = .32       # relative size of each meter's own hourly load variation
    hours: int = 192              # history
    seed: int = 0
    load_kw: float = 1.5          # mean load per meter
    roots: tuple = ()             # root lines: ((blocks, r ohm, parent index or -1 for the trunk end), ...)
    v_noise: float = 5e-4         # relative noise of a voltage reading (meter class)

    @property
    def n(self):
        return self.blocks*self.per

    @property
    def name(self):
        s = "x".join(map(str, self.shared)) if self.shared else "none"
        out = (f"d{self.blocks}_p{self.per}_r{self.r_shared:g}_e{self.excitation:g}_h{self.hours}_s{self.seed}"
               f"_S{s}")
        if self.roots:
            out += "_R"+"-".join(f"{'x'.join(map(str, b))}r{r:g}p{p}" for b, r, p in self.roots)
        if self.v_noise != 5e-4:
            out += f"_v{self.v_noise:g}"
        return out


def root_feeder(k_root, per_root, trunk_laterals, per, r, excitation=.32, hours=192, seed=0, inner=0,
                v_noise=5e-4):
    """A feeder whose trunk end feeds k_root short root lines of resistance r, each feeding per_root laterals, and
    trunk_laterals laterals directly; `per` meters per lateral. inner > 0: a further root line of resistance r inside
    the first one, feeding `inner` of its laterals (the trunk continuing: nested near-root lines). The laterals behind
    each root line are drawn from the seed."""
    L = k_root*per_root+trunk_laterals
    rng = np.random.default_rng([SEED, 7, L, k_root, per_root, seed])
    perm = [int(j) for j in rng.permutation(L)]
    groups = [perm[k*per_root:(k+1)*per_root] for k in range(k_root)]
    roots = [(tuple(sorted(g)), float(r), -1) for g in groups]
    if inner and groups and 2 <= inner < len(groups[0]):
        roots.append((tuple(sorted(groups[0][:inner])), float(r), 0))
    return Feeder(blocks=L, per=per, excitation=excitation, hours=hours, seed=seed, roots=tuple(roots),
                  v_noise=v_noise)


def draw_shared(blocks, seed, size=None):
    """A random set of 2 to blocks-2 blocks (the hidden segment's), from the seed."""
    rng = np.random.default_rng([SEED, 1, blocks, seed])
    k = int(rng.integers(2, blocks-1)) if size is None else size
    return tuple(sorted(int(j) for j in rng.choice(blocks, k, replace=False)))


def block_meters(f):
    return [tuple(range(j*f.per, (j+1)*f.per)) for j in range(f.blocks)]


def truth(f):
    """(true meter groupings, the same with the hidden segment contracted, the segment's grouping or None).
    For a feeder with root lines, `flat` is the laterals only and the root lines are in `true` (see root_lines)."""
    blocks = [b for b in block_meters(f) if len(b) >= 2]
    target = tuple(sorted(i for j in f.shared for i in block_meters(f)[j])) if f.shared else None
    flat = canonical(blocks)
    return canonical([*flat, *([target] if target else []), *root_lines(f)]), flat, target


def root_lines(f):
    """The meter groupings of the root lines, in the order of f.roots."""
    bm = block_meters(f)
    return [tuple(sorted(i for j in b for i in bm[j])) for b, _, _ in f.roots]


def lateral_parent(f, j):
    """The bus that feeds lateral j: the deepest root line containing it, else the trunk end (bus 1)."""
    inside = [k for k, (b, _, _) in enumerate(f.roots) if j in b]
    if not inside:
        return 2 if j in f.shared else 1
    k = min(inside, key=lambda k: len(f.roots[k][0]))
    return 200+k


def network(f):
    """The circuit (dict for physics.power_flow) and the meter bus labels."""
    rng = np.random.default_rng([SEED, 2, f.blocks, f.per, f.seed])
    labels = list(range(1000, 1000+f.n))
    lines = []

    def edge(a, b, r):
        lines.append(dict(from_bus=a, to_bus=b, r_ohm=float(r), x_ohm=float(.4*r if b != 1 else TRUNK[1]),
                          status=True))
    edge(0, 1, TRUNK[0])
    if f.shared:
        edge(1, 2, f.r_shared)
    for k, (_, r, parent) in enumerate(f.roots):
        edge(1 if parent < 0 else 200+parent, 200+k, r)
    for j, meters in enumerate(block_meters(f)):
        edge(lateral_parent(f, j), 100+j, rng.uniform(.03, .05))
        for i in meters:
            edge(100+j, labels[i], rng.uniform(.015, .025))
    buses = sorted({e[k] for e in lines for k in ("from_bus", "to_bus")})
    return dict(root_bus=0, buses=[dict(bus_id=b) for b in buses], branches=lines, base_kv=BASE_KV,
                base_mva=BASE_MVA), labels


def loads(f, extra_hours=24):
    """Quarter-hour active and reactive loads (pu) and PCC voltage for the history and `extra_hours` more."""
    H = f.hours+extra_hours
    rng = np.random.default_rng([SEED, 3, f.n, f.seed, int(round(1e4*f.excitation))])
    t = np.arange(4*H)
    day = np.sin(2*np.pi*(t/96-.3))
    mean = f.load_kw/BASE_KW*rng.uniform(.6, 1.4, f.n)
    own = rng.normal(size=(H, f.n)).repeat(4, axis=0)
    p = mean*np.clip(1+.3*day[:, None]+f.excitation*own+.04*rng.normal(size=(4*H, f.n)), .1, 3.)
    q = .3*p+.05*mean*rng.normal(size=(H, f.n)).repeat(4, axis=0)
    v0 = 1.02+.002*day+.001*rng.normal(size=H).repeat(4)
    return p, q, v0


class Meters:
    """Fixed gains from ptid.cases.physics.Meters; the noise of hour h from its own stream (paired runs)."""

    def __init__(self, f):
        self.f = f
        self.base = physics.Meters(f.n, [SEED, f.seed, f.blocks, f.per], v_noise=f.v_noise)

    def read(self, hour0, p, q, v, v0, s):
        b, out = self.base, {k: [] for k in ("P", "Q", "V", "V0", "Ppcc", "Qpcc")}
        for k in range(len(p)):
            rng = np.random.default_rng([SEED, 4, self.f.seed, self.f.n, hour0+k])
            e = lambda shape, sd: sd*rng.normal(size=shape)
            n = self.f.n
            out["P"].append(p[k]*(1+b.p_gain+e(n, b.pq_noise)))
            out["Q"].append(q[k]*(1+b.q_gain+e(n, b.pq_noise)))
            out["V"].append(v[k]*(1+b.v_gain+e(n, b.v_noise)))
            out["V0"].append(v0[k]*(1+b.v0_gain+e((), b.v_noise)))
            out["Ppcc"].append(s[k].real*(1+b.pcc_gain[0]+e((), b.pq_noise)))
            out["Qpcc"].append(s[k].imag*(1+b.pcc_gain[1]+e((), b.pq_noise)))
        out = {k: np.asarray(a, float) for k, a in out.items()}
        return dict(P=out["P"], Q=out["Q"], Y=out["V0"][:, None]**2-out["V"]**2, V0sq=out["V0"]**2,
                    Ppcc=out["Ppcc"], Qpcc=out["Qpcc"])


class Simulator:
    """AC simulation of one feeder: the history, and the probing day hour by hour with an optional block."""

    def __init__(self, f):
        self.f = f
        self.net, self.labels = network(f)
        self.p, self.q, self.v0 = loads(f)
        self.meters = Meters(f)

    def hours(self, first, count, order=None):
        """Readings of hours first..first+count-1; order: None or the active power (pu, one value per meter) added to
        each meter's load during these hours (a battery or EV behind the meter, so the meter measures it)."""
        rows = slice(4*first, 4*(first+count))
        p, q, v0 = self.p[rows].copy(), self.q[rows], self.v0[rows]
        if order is not None:
            p += np.asarray(order, float)[None, :]
        order, v, s, _ = physics.power_flow(self.net, self.labels, p, q, v0)
        v = v[:, [order.index(b) for b in self.labels]]
        h = physics.hourly
        return self.meters.read(first, h(p), h(q), h(v), h(v0), h(s))

    def history(self):
        return self.hours(0, self.f.hours)


def join(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def model_of(data, lam=1.96):
    """The identification model on the data (the gate decides whether an unmetered load is modelled; here the
    feeders have none, so it should not be)."""
    u = balance(data["P"], data["Q"], data["Y"], data["V0sq"], data["Ppcc"], data["Qpcc"])
    load = load_share(data["P"], u) >= GATE
    return Model(data["P"], data["Q"], data["Y"], data["V0sq"], u if load else None, lam), bool(load)


def score(model, lines):
    """H of the tree with these lines (score units: half-BIC, as in the draft)."""
    return Fitter(model, lines).crit(lines)*model.N/2


def widest(model, tree):
    fam = canonical([*basis(model.leaves), *tree])
    node, kids = max(children(fam, model.leaves).items(), key=lambda kv: len(kv[1]))
    return node, list(kids)


def union_search(model, tree, kids, start=None, forbid=None):
    """A quick local search over the unions of `kids` (sets of child indices): the best pair (or `start`), then
    one-child additions and removals while one scores better. `forbid` (a set) is never returned. For warm starts."""
    tree = canonical(tree)
    fam = canonical([*basis(model.leaves), *tree])
    d = len(kids)
    U = lambda S: tuple(sorted(i for a in S for i in kids[a]))

    def best_of(cands):
        cands = [S for S in cands if 2 <= len(S) <= d-1 and S != forbid and U(S) not in set(fam)]
        if not cands:
            return None, math.inf
        fit = Fitter(model, [*fam, *(U(S) for S in cands)])
        return min(((S, fit.crit([*tree, U(S)])) for S in cands), key=lambda x: x[1])

    from itertools import combinations
    S, value = (start, best_of([start])[1]) if start is not None else \
        best_of([frozenset(p) for p in combinations(range(d), 2)])
    while S is not None:
        T, v = best_of([S ^ {a} for a in range(d)])
        if T is None or v >= value-1e-12:
            break
        S, value = T, v
    return S


def node_scan(model, tree, seconds, seed=7, node=None, flip=True, warm=False):
    """The node MIQP at `node` (default: the widest node of `tree`), and its flip test.

    gain:      H(T) - H(T + S*) for the best union S* found (score units, refitted);
    max_gain:  the most any union at the node can gain, from the solver's lower bound (None without a bound);
    runner:    the best other union S2 (node MIQP with S* excluded by a no-good cut), with
    runner_gap = H(T + S2) - H(T + S*) and runner_gap_low, its lower end from that solve's bound: every other union at
               the node scores at least this much worse than S*.
    Decision at margin M: present (gain >= M and runner_gap_low >= M: S* beats the flat node and every other union),
    absent (max_gain <= -M: every union scores at least M worse than the flat node) or open."""
    tree = canonical(tree)
    if node is None:
        node, kids = widest(model, tree)
    else:
        kids = list(children(canonical([*basis(model.leaves), *tree]), model.leaves)[node])
    half = model.N/2
    t0 = time.perf_counter()
    start = union_search(model, tree, kids) if warm else None
    line, value, status, bound = node_miqp(model, tree, kids, seconds, seed, return_bound=True, start=start)
    base = Fitter(model, tree).crit(tree)
    upper = base+model.alpha                       # any union with zero coefficients attains this
    gain = refit = None
    if line is not None:
        refit = Fitter(model, [*tree, line]).crit([*tree, line])
        upper = min(upper, refit)
        gain = (base-refit)*half
    max_gain = None
    if bound is not None and bound <= upper+1e-7:
        max_gain = (base-min(bound, upper))*half
    out = dict(node=list(node), children=len(kids), line=None if line is None else list(line), gain=gain,
               max_gain=max_gain, status=int(status), runner=None, runner_gap=None, runner_gap_low=None,
               runner_status=None)
    if flip and line is not None:
        chosen = frozenset(a for a, kid in enumerate(kids) if set(kid) <= set(line))
        start2 = union_search(model, tree, kids, forbid=chosen) if warm else None
        line2, _, status2, bound2 = node_miqp(model, tree, kids, seconds, seed, return_bound=True, exclude=[chosen],
                                              start=start2)
        out.update(runner_status=int(status2))
        if line2 is not None:
            refit2 = Fitter(model, [*tree, line2]).crit([*tree, line2])
            out.update(runner=list(line2), runner_gap=(refit2-refit)*half)
            if bound2 is not None and bound2 <= refit2+1e-7:
                out["runner_gap_low"] = (min(bound2, refit2)-refit)*half
    out["seconds"] = time.perf_counter()-t0
    present = gain is not None and gain >= M and (not flip or (out["runner_gap_low"] or -math.inf) >= M)
    out["decision"] = "present" if present else ("absent" if max_gain is not None and max_gain <= -M else "open")
    return out


def node_label(model, tree, line, seconds, seed=7, warm=True, pairs=()):
    """The flip test of a line of the tree at its node: remove it (its children join its parent node v), then compare
    the tree with every tree that differs at v by one line: the tree without it, and the tree with any other union S'
    of v's children instead (node MIQP with the line's union excluded; its bound covers every S').
    pairs: lines of the tree to test jointly with it (pair_test) when they are its siblings at v (disjoint from it, same
    parent node); node refinement passes the lines it added, so the given laterals are not paired.
    Returns dict(gap_low, gap_flat, gap_other, other, status, children, pairs, decision): gaps H(alternative) - H(tree)
    in score units; decided when gap_low and every pair's gap_low are at least M."""
    tree = canonical(tree)
    line = tuple(line)
    rest = canonical(C for C in tree if C != line)
    fam = canonical([*basis(model.leaves), *rest])
    parent = min((C for C in fam if set(line) < set(C)), key=len)
    kids = list(children(fam, model.leaves)[parent])
    half = model.N/2
    h_tree = Fitter(model, tree).crit(tree)
    gap_flat = (Fitter(model, rest).crit(rest)-h_tree)*half
    chosen = frozenset(a for a, kid in enumerate(kids) if set(kid) <= set(line))
    out = dict(line=list(line), children=len(kids), gap_flat=gap_flat, gap_other=None, other=None, status=None)
    gap_other_low = math.inf
    if len(kids) >= 3:
        start = union_search(model, rest, kids, forbid=chosen) if warm else None
        other, _, status, bound = node_miqp(model, rest, kids, seconds, seed, return_bound=True, exclude=[chosen],
                                            start=start)
        out["status"] = int(status)
        if other is not None:
            refit = Fitter(model, [*rest, other]).crit([*rest, other])
            out.update(other=list(other), gap_other=(refit-h_tree)*half)
            gap_other_low = ((min(bound, refit) if bound is not None and bound <= refit+1e-7 else -math.inf)
                             - h_tree)*half
        elif bound is not None:
            gap_other_low = (bound-h_tree)*half
        else:
            gap_other_low = -math.inf
    out["gap_low"] = min(gap_flat, gap_other_low)
    out["pairs"] = []
    for D in canonical(pairs):
        if D in rest:
            if D == parent or set(D) & set(line) or not set(D) < set(parent):
                continue
            if min((C for C in fam if set(D) < set(C)), key=len) == parent:      # a sibling at the same node
                out["pairs"].append(pair_test(model, tree, line, D, parent, seconds, seed))
    low = min([out["gap_low"], *(p["gap_low"] for p in out["pairs"])])
    out["decision"] = "decided" if low >= M else "open"
    return out


def pair_test(model, tree, C, D, parent, seconds, seed=7):
    """The two-line flip test of sibling lines C and D below node `parent`: the tree against the tree without both,
    and against every tree with up to two other disjoint lines there instead (search.node_miqp_multi with {C, D}
    excluded; its bound covers every such pair). A lateral swapped between C and D is such a change; the one-line test
    cannot see it, because moving the lateral changes both lines. Returns dict(sibling, gap_none, gap_other, other,
    status, gap_low) in score units."""
    from ptid.method.search import node_miqp_multi
    rest = canonical(E for E in tree if E not in (tuple(C), tuple(D)))
    fam = canonical([*basis(model.leaves), *rest])
    kids = list(children(fam, model.leaves)[parent])
    half = model.N/2
    h_tree = Fitter(model, tree).crit(tree)
    gap_none = (Fitter(model, rest).crit(rest)-h_tree)*half
    pick = lambda E: frozenset(a for a, kid in enumerate(kids) if set(kid) <= set(E))
    found, _, status, bound = node_miqp_multi(model, rest, kids, seconds, 2, seed, exclude=[[pick(C), pick(D)]])
    out = dict(sibling=list(D), gap_none=gap_none, gap_other=None, other=[list(E) for E in found], status=int(status))
    low = -math.inf
    if found:
        refit = Fitter(model, [*rest, *found]).crit([*rest, *found])
        out["gap_other"] = (refit-h_tree)*half
        if bound is not None and bound <= refit+1e-7:
            low = (bound-h_tree)*half
    elif bound is not None:
        low = (bound-h_tree)*half
    out["gap_low"] = min(gap_none, low)
    return out


def price_loop(model, tree, seconds, gamma=0., K=9, max_add=10):
    """Node refinement alone (Algorithm 1's pricing without re-solves): add the best paying line while one exists
    (enumeration at nodes of at most K children, the node MIQP beyond; extra price gamma ln(#unions) when gamma > 0).
    Returns (tree, added lines, info)."""
    from ptid.method.search import best_addition
    tree, added, by_miqp, unproved = list(canonical(tree)), [], 0, 0
    certified = False
    while len(added) < max_add:
        best, info = best_addition(model, tree, K, seconds, node_gamma=gamma)
        by_miqp += info["by_miqp"]
        unproved = info["unproved"]
        certified = info["certified"]
        if best is None:
            break
        tree.append(tuple(best))
        added.append(tuple(best))
    return canonical(tree), added, dict(node_miqp_solves=by_miqp, last_unproved=unproved, certified=certified)


def local_loop(model, tree, seconds=None, gamma=0., K=9, max_add=10):
    """price_loop with a local search (union_search: best pair, then one-child moves) in place of the node MIQP at
    nodes wider than K; enumeration at the others, the same price. The baseline that shows what exactness adds."""
    from ptid.method.search import PRICING_TOL, best_addition, node_price
    tree, added = list(canonical(tree)), []
    crit = lambda t: Fitter(model, t).crit(t)
    while len(added) < max_add:
        best, info = best_addition(model, tree, K, wide_node_miqp=False, node_gamma=gamma)
        gain = info["gain"] if best is not None else 0.
        current = crit(tree)
        fam = canonical([*basis(model.leaves), *tree])
        for node, kids in children(fam, model.leaves).items():
            if len(kids) <= K:
                continue
            S = union_search(model, tree, list(kids))
            if S is None:
                continue
            line = tuple(sorted(i for a in S for i in kids[a]))
            g = current-crit([*tree, line])-node_price(model, len(kids), gamma)
            if g > max(gain, PRICING_TOL):
                best, gain = line, g
        if best is None:
            break
        tree.append(tuple(best))
        added.append(tuple(best))
    return canonical(tree), added, dict(local=True)


def price_loop_multi(model, tree, seconds, gamma=0., K=9, lines=2, max_add=10):
    """Node refinement with sibling lines: at every step the best single line (search.best_addition) competes with
    the best set of up to `lines` disjoint new lines at each node wider than K (search.node_miqp_multi); the larger
    net gain is added. After each step an added line is removed again if the tree scores better without it (a line
    that only stood in for others). Returns (tree, added lines still in the tree, info)."""
    from ptid.method.search import PRICING_TOL, best_addition, node_miqp_multi, node_price
    tree, added, solves = list(canonical(tree)), [], 0
    crit = lambda t: Fitter(model, t).crit(t)
    for _ in range(max_add):
        current = crit(tree)
        best, info = best_addition(model, tree, K, seconds, node_gamma=gamma)
        solves += info["by_miqp"]
        choice = ([tuple(best)], info["gain"]) if best is not None else ([], 0.)
        fam = canonical([*basis(model.leaves), *tree])
        for node, kids in children(fam, model.leaves).items():
            if len(kids) <= K:
                continue
            found, value, status, bound = node_miqp_multi(model, tree, list(kids), seconds, lines, gamma=gamma)
            solves += 1
            if found:
                net = current-crit([*tree, *found])-len(found)*node_price(model, len(kids), gamma)
                if net > max(choice[1], PRICING_TOL):
                    choice = (found, net)
        if not choice[0]:
            break
        tree += choice[0]
        added += choice[0]
        for C in list(added):                       # drop a line that the others made useless
            rest = [D for D in tree if D != C]
            if crit(rest) < crit(tree)-PRICING_TOL:
                tree, added = rest, [D for D in added if D != C]
    return canonical(tree), added, dict(node_miqp_solves=solves)


def enumerate_node(model, tree, node=None, chunk=200):
    """Every union of 2 to c-1 children of the node, each refitted (the exhaustive reference). Returns dict."""
    tree = canonical(tree)
    fam = canonical([*basis(model.leaves), *tree])
    if node is None:
        node, kids = widest(model, tree)
    else:
        kids = list(children(fam, model.leaves)[node])
    t0 = time.perf_counter()
    base = Fitter(model, tree).crit(tree)
    top = [(math.inf, None), (math.inf, None)]               # the best two unions
    count = 0
    new = [U for U in unions(kids) if U not in set(fam)]
    for k in range(0, len(new), chunk):
        part = new[k:k+chunk]
        fit = Fitter(model, [*fam, *part])
        for U in part:
            v = fit.crit([*tree, U])
            count += 1
            if v < top[1][0]:
                top = sorted([top[0], (v, U)], key=lambda x: x[0])
    half = model.N/2
    return dict(line=list(top[0][1]), gain=(base-top[0][0])*half, runner=list(top[1][1]),
                runner_gap=(top[1][0]-top[0][0])*half, unions=count, seconds=time.perf_counter()-t0)


def grouping_f1(found, true):
    found, true = set(canonical(found)), set(canonical(true))
    if not found and not true:
        return 1.
    return 2*len(found & true)/(len(found)+len(true))


def write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=_plain)+"\n")


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _plain(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x))


def feeder_dict(f):
    return dict(asdict(f), name=f.name, n=f.n)
