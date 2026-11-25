"""Feeder plans: the old star cases, their street versions, and random street feeders (see DESIGN.md).

A feeder is first drawn as a *plan*: a dictionary of nodes. Every node except the root has one line to its parent
(length in metres and a conductor, or r and x when they come from an old case). A node with `customers` > 0 carries a
meter. `to_network` turns a plan into the network format of the other studies.

    star_plan(case)            the old case as it is (lines leave the transformer one by one)
    street_plan(case)          the same meters, customers and lines, opened out along one street
    random_plan(seed, size)    a new street feeder
"""
from __future__ import annotations

import numpy as np

from rnj_wzzt_dr.physical import backend

from .conductors import impedance

ROOT = 1
OLD_CASES = ("soumalas11", "paper15", "flynn16", "pengwah18")

# ------------------------------------------------ lengths on real feeders (84 EU LV feeders, metres)
TRUNK = dict(median=50., spread=.65, low=10., high=200.)
SECTION = dict(median=20., spread=.80, low=4., high=120.)
SERVICE = dict(median=7.5, spread=.95, low=1.5, high=60.)

# ------------------------------------------------ the three kinds of street
# main: conductors from the transformer to the end of the street (thick to thin); loop_in: share of connections whose
# meter sits on the street node itself; customers: range of the feeder's customers (real feeders: 10 to 170 at every
# number of connections); stretch: factor on the lengths
STYLES = {
    "urban": dict(main=("cable185", "cable150"), lateral=("cable150", "cable95"), service="cable25",
                  loop_in=.40, customers=(60, 160), stretch=1.0),
    "suburban": dict(main=("cable150", "cable95"), lateral=("cable95", "cable95"), service="cable25",
                     loop_in=.20, customers=(40, 120), stretch=1.3),
    "rural": dict(main=("bundle95", "bundle50"), lateral=("bundle50", "bundle25"), service="bundle25",
                  loop_in=.0, customers=(25, 70), stretch=2.0),
}
SIZES = {"small": (10, 15), "medium": (16, 25), "large": (26, 40), "huge": (41, 60)}
CUSTOMERS_PER_KW = 1.2        # old cases: customers of a terminal per kW of its nominal load


def draw_length(rng, law, stretch=1.):
    value = stretch*law["median"]*np.exp(law["spread"]*rng.normal())
    return float(np.clip(value, law["low"], law["high"]))


class Plan:
    def __init__(self, name, style, meta=None):
        self.name, self.style, self.meta = name, style, dict(meta or {})
        self.nodes = {ROOT: dict(parent=None, customers=0, r=0., x=0., length=0., conductor="", role="root")}

    def add(self, parent, length=0., conductor="", customers=0, r=None, x=None, role="street", node=None):
        if node is None:
            node = max(self.nodes)+1
        if r is None:
            r, x = impedance(conductor, length)
        self.nodes[node] = dict(parent=parent, customers=int(customers), r=float(r), x=float(x), length=float(length),
                                conductor=conductor, role=role)
        return node

    def children(self, node):
        return [k for k, v in self.nodes.items() if v["parent"] == node]

    def below(self, node):
        """Meters at or below a node."""
        out = [node] if self.nodes[node]["customers"] > 0 else []
        for child in self.children(node):
            out += self.below(child)
        return out

    def merge_series(self):
        """Remove unmetered nodes with one child (no measurement can place them) and unmetered dead ends."""
        changed = True
        while changed:
            changed = False
            for node in [k for k in self.nodes if k != ROOT]:
                info, kids = self.nodes[node], self.children(node)
                if info["customers"] > 0 or len(kids) > 1:
                    continue
                if kids:
                    child = self.nodes[kids[0]]
                    for key in ("r", "x", "length"):
                        child[key] += info[key]
                    child["parent"] = info["parent"]
                    names = [c for c in (info["conductor"], child["conductor"]) if c]
                    child["conductor"] = "+".join(dict.fromkeys(names))
                del self.nodes[node]
                changed = True
                break
        return self


# ------------------------------------------------ old cases
def star_plan(case):
    """The old case, unchanged lines; customers per terminal from its nominal load."""
    old = backend.load_case(case)
    order, parent, incoming = backend._radial_order(old)
    bus = {b["bus_id"]: b for b in old["buses"]}
    plan = Plan(f"{case}_star", "star", dict(old_case=case, shape="star"))
    for b in order[1:]:
        line = incoming[b]
        customers = max(1, round(CUSTOMERS_PER_KW*bus[b]["pd_kw"])) if bus[b]["is_terminal"] else 0
        plan.add(parent[b], float(line["length_m"]), "", customers, line["r_ohm"], line["x_ohm"],
                 "service" if customers else "lateral", node=b)
    return plan


def street_plan(case):
    """The old case rearranged along one street: same meters, customers, service lines and laterals.

    1. The lines that left the transformer side by side become consecutive sections of one street, in the old order
       and with their old lengths, now in a main cable. The first one is the trunk that carries every meter.
    2. Where more than two lines left one point, they leave one by one along the street, nearest first: services,
       then laterals from the smallest to the largest, and the street itself last. The new short sections have the
       lengths of real street sections.
    """
    rng = np.random.default_rng(np.random.SeedSequence([20261125, OLD_CASES.index(case)]))
    plan = star_plan(case)
    plan.name, plan.style, plan.meta = f"{case}_street", "street", dict(old_case=case, shape="street")
    thick, thin = STYLES["suburban"]["main"]
    total = len(plan.below(ROOT))
    head_lines = plan.children(ROOT)

    def main_cable(meters_below):
        return thick if meters_below > total/3 else thin

    last = ROOT
    for k, node in enumerate(head_lines):                     # step 1, from the transformer outwards
        rest = sum(len(plan.below(b)) for b in head_lines[k:])
        conductor = main_cable(rest)
        r, x = impedance(conductor, plan.nodes[node]["length"])
        plan.nodes[node].update(parent=last, conductor=conductor, r=r, x=x, role="trunk" if k == 0 else "street")
        last = node
    for node in [k for k in plan.nodes if k != ROOT]:         # step 2
        kids = sorted(plan.children(node), key=lambda k: (plan.nodes[k]["role"] in ("street", "trunk"),
                                                          len(plan.below(k)), plan.nodes[k]["r"], k))
        at = node
        while len(kids) > 2:
            kids.pop(0)
            downstream = sum(len(plan.below(k)) for k in kids)
            on_main = any(plan.nodes[k]["role"] == "street" for k in kids) or node in head_lines
            conductor = main_cable(downstream) if on_main else thin
            nxt = plan.add(at, round(draw_length(rng, SECTION), 1), conductor, role="street")
            for k in kids:
                plan.nodes[k]["parent"] = nxt
            at = nxt
    return plan.merge_series()


# ------------------------------------------------ random street feeders
def random_plan(seed, size, style=None, stream="dev"):
    """One street feeder. `stream` separates development ("dev") and held-out ("test") random numbers."""
    rng = np.random.default_rng(np.random.SeedSequence([20261125, {"dev": 1, "test": 2}[stream], int(seed),
                                                        list(SIZES).index(size)]))
    style = style or list(STYLES)[int(seed) % len(STYLES)]
    s = STYLES[style]
    low, high = SIZES[size]
    n = int(rng.integers(low, high+1))
    plan = Plan(f"{stream}_{size}_{int(seed):03d}", style,
                dict(shape="random", size=size, seed=int(seed), stream=stream))

    total = max(int(rng.integers(s["customers"][0], s["customers"][1]+1)), n)
    customers = list(1+rng.multinomial(total-n, rng.dirichlet(np.ones(n))))     # uneven, as on real feeders

    def connect(node):
        """One connection at a street node: looped in (the meter is the node) or over a service line."""
        if plan.nodes[node]["customers"] == 0 and rng.random() < s["loop_in"]:
            plan.nodes[node]["customers"] = int(customers.pop())
        else:
            plan.add(node, draw_length(rng, SERVICE, s["stretch"]), s["service"], customers.pop(), role="service")

    def street(node, count, level):
        """Place `count` connections along a street that starts at `node` (level 0: the main street)."""
        left = count
        while left > 0:
            if left >= 6 and level < 2 and rng.random() < .15:       # a side street leaves here
                side = int(rng.integers(2, left//2+1))
                first = plan.add(node, draw_length(rng, SECTION, s["stretch"]), s["lateral"][level], role="lateral")
                street(first, side, level+1)
                left -= side
            else:                                                    # one to three connections at this pole
                here = min(left, 1+int(rng.choice(3, p=(.75, .20, .05))))
                for _ in range(here):
                    connect(node)
                left -= here
            if left > 0:                                             # the street goes on to the next pole
                if level == 0:
                    conductor = s["main"][0] if left > count/3 else s["main"][1]
                else:
                    conductor = s["lateral"][level-1]
                node = plan.add(node, draw_length(rng, SECTION, s["stretch"]), conductor, role="street")

    head = plan.add(ROOT, draw_length(rng, TRUNK, s["stretch"]), s["main"][0], role="trunk")
    street(head, n, 0)
    return plan.merge_series()


# ------------------------------------------------ network format of the other studies
def to_network(plan, base_kw=200., base_kv=.4):
    """Buses, branches and metadata as in physical.backend."""
    buses, branches = [], []
    for node, v in sorted(plan.nodes.items()):
        metered = v["customers"] > 0
        kind = "root" if node == ROOT else "terminal" if metered else "hidden_internal"
        buses.append(dict(bus_id=node, bus_type=kind, pd_kw=0., qd_kvar=0., is_observed=metered or node == ROOT,
                          has_load=metered, is_terminal=metered, customers=v["customers"]))
        if v["parent"] is not None:
            branches.append(dict(from_bus=v["parent"], to_bus=node, r_ohm=v["r"], x_ohm=v["x"], status=True,
                                 branch_type=v["role"], length_m=v["length"], conductor=v["conductor"]))
    return dict(metadata=dict(name=plan.name, style=plan.style, **plan.meta), buses=buses, branches=branches,
                root_bus=ROOT, base_kw=base_kw, base_kv=base_kv, base_mva=base_kw/1000.)


def labels_of(network):
    return tuple(b["bus_id"] for b in network["buses"] if b["is_terminal"])
