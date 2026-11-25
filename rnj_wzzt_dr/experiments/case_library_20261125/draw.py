"""Draw the feeders: old star cases, their street versions, random feeders and real ones.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.draw

Writes outputs/case_library_20261125/feeders.png. Distance from the transformer runs to the right; every horizontal
line is one cable; orange cables carry more than half of the meters (the root-side lines), blue ones fewer.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc

from . import build, feeders

INK, MUTED, SURFACE, FEW, MANY = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"


def draw(ax, network, labels, title):
    order, parent, incoming = sc.backend._radial_order(network)
    n = len(labels)
    below = {b: set() for b in order}
    for i, l in enumerate(labels):
        below[l].add(i)
    for b in reversed(order[1:]):
        below[parent[b]] |= below[b]
    kids = {b: [k for k in order[1:] if parent[k] == b] for b in order}
    x = {order[0]: 0.}
    for b in order[1:]:
        x[b] = x[parent[b]]+float(incoming[b].get("length_m") or 0.)
    y, row = {}, [0]

    def place(b):
        ends = [place(k) for k in sorted(kids[b], key=lambda k: len(below[k]))]
        if b in labels and kids[b]:                      # a meter on the street node itself
            ends.append(row[0]); row[0] += 1
        if not ends:
            y[b] = row[0]; row[0] += 1
        else:
            y[b] = ends[-1] if not (b in labels and kids[b]) else ends[-2]
        return y[b]
    place(order[0])
    for b in order[1:]:
        p = parent[b]
        share = len(below[b])/n
        color = MANY if share > .5 else FEW
        width = 1.2+2.2*share
        ax.plot([x[p], x[p]], [y[p], y[b]], color=color, lw=width, solid_capstyle="round", zorder=2)
        ax.plot([x[p], x[b]], [y[b], y[b]], color=color, lw=width, solid_capstyle="round", zorder=2)
    for l in labels:
        ax.plot(x[l], y[l], "o", ms=4.5, color=INK, mec=SURFACE, mew=1, zorder=3)
    ax.plot(0, y[order[0]], "s", ms=9, color=INK, mec=SURFACE, mew=1, zorder=4)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_yticks([]); ax.set_xlabel("m", fontsize=7, color=MUTED); ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0cfca"); ax.invert_yaxis(); ax.set_facecolor(SURFACE)


def main():
    fig, axes = plt.subplots(5, 3, figsize=(13, 16), facecolor=SURFACE)
    for r, case in enumerate(("soumalas11", "paper15", "flynn16", "pengwah18")):
        for c, (plan, what) in enumerate(((feeders.star_plan(case), "old case (star)"), (feeders.street_plan(case), "new street version"))):
            net = feeders.to_network(plan)
            draw(axes[r, c], net, feeders.labels_of(net), f"{case}: {what}")
    real = ("eulv_td008690_feeder19", "eulv_td400067_feeder147", "eulv_td401426_feeder155", "eulv_td004454_feeder160")
    for r, case in enumerate(real):
        net = sc.load_network(case)
        labels = sc.network_info(case, net)[1]
        draw(axes[r, 2], net, labels, f"real feeder {case.replace('eulv_', '')}, {len(labels)} meters")
    for c, name in enumerate(("dev_medium_001_none", "dev_medium_002_none", "dev_large_003_none")):
        rec = build.load("dev", name)
        draw(axes[4, c], rec["network"], rec["labels"], f"random {name[4:-5]} ({rec['network']['metadata']['style']}), {rec['n']} meters")
    fig.legend(handles=[Line2D([], [], color=MANY, lw=3, label="carries more than half of the meters"),
                        Line2D([], [], color=FEW, lw=2, label="carries at most half"),
                        Line2D([], [], color=INK, marker="o", ls="", ms=5, label="meter"),
                        Line2D([], [], color=INK, marker="s", ls="", ms=8, label="transformer (PCC)")],
               loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .975))
    path = build.OUT/"feeders.png"
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    print(path)


if __name__ == "__main__":
    main()
