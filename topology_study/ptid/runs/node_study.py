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

    @property
    def n(self):
        return self.blocks*self.per

    @property
    def name(self):
        s = "x".join(map(str, self.shared)) if self.shared else "none"
        return (f"d{self.blocks}_p{self.per}_r{self.r_shared:g}_e{self.excitation:g}_h{self.hours}_s{self.seed}"
                f"_S{s}")


def draw_shared(blocks, seed, size=None):
    """A random set of 2 to blocks-2 blocks (the hidden segment's), from the seed."""
    rng = np.random.default_rng([SEED, 1, blocks, seed])
    k = int(rng.integers(2, blocks-1)) if size is None else size
    return tuple(sorted(int(j) for j in rng.choice(blocks, k, replace=False)))


def block_meters(f):
    return [tuple(range(j*f.per, (j+1)*f.per)) for j in range(f.blocks)]


def truth(f):
    """(true meter groupings, the same with the hidden segment contracted, the segment's grouping or None)."""
    blocks = [b for b in block_meters(f) if len(b) >= 2]
    target = tuple(sorted(i for j in f.shared for i in block_meters(f)[j])) if f.shared else None
    flat = canonical(blocks)
    return canonical([*flat, *([target] if target else [])]), flat, target


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
    for j, meters in enumerate(block_meters(f)):
        edge(2 if j in f.shared else 1, 100+j, rng.uniform(.03, .05))
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
        self.base = physics.Meters(f.n, [SEED, f.seed, f.blocks, f.per])

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
