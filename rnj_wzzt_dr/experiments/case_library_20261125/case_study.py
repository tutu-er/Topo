"""Worked case studies (DEVELOPMENT cases): truth, our tree with the located source, and RNJ, side by side.

    python -m rnj_wzzt_dr.experiments.case_library_20261125.case_study

Writes outputs/case_library_20261125/case_study.png and case_study.json (the numbers behind the figure).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import build, compare_dev as cd

CASES = ("paper15_street_l1_end", "flynn16_street_l2_none", "pengwah18_street_l3_end", "dev_small_002_end",
         "dev_medium_005_end", "dev_medium_003_none")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
RIGHT, WRONG, MISSED, PATH = "#2a78d6", "#eb6834", "#9a9993", "#1baf7a"


def analyse(name):
    from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
    from rnj_wzzt_dr.identification import final, trees
    rec = build.load("dev", name)
    inp = build.to_input(rec); n = inp.n
    tr, va = p2.series(inp)
    inner = lambda f: {tuple(sorted(C)) for C in f if 1 < len(C) < n}
    r = final.fit(tr, va, inp.hist["Ppcc"], inp.hist["Qpcc"])
    shared, depth, _ = trees.geometry(r["S"]["R"], r["S"]["R"], 1.)
    truth = inner(rec["truth"])
    return dict(name=name, n=n, truth=truth, ours=inner(r["located"]), rnj=inner(trees.rnj(shared, depth, .08)),
                path=[tuple(C) for C in r["path"]], certified=r["record"]["certified"], evidence=float(r["source_evidence"]),
                places=[(float(c), D) for c, D in r["places"]][:4], source=rec["source"], network=rec["network"], labels=rec["labels"])


def layout(lines, order, n):
    """Cladogram: y of the meters as given by `order`; every line at the depth of the number of lines that contain it."""
    nodes = sorted(lines, key=len)
    depth = {C: 1+sum(set(C) < set(D) for D in nodes) for C in nodes}
    return depth


def panel(ax, title, lines, truth, order, n, path=(), source_at=None):
    """Nested brackets: one horizontal bar per line from its first to its last meter."""
    pos = {m: k for k, m in enumerate(order)}
    depth = layout(lines, order, n)
    top = max(depth.values(), default=0)+1
    for m in range(n):
        ax.plot(pos[m], 0, "o", ms=4, color=INK, zorder=4)
    for C in lines:
        xs = sorted(pos[m] for m in C)
        ok = C in truth
        on_path = C in path
        color = PATH if on_path else RIGHT if ok else WRONG
        ax.plot([xs[0]-.25, xs[-1]+.25], [depth[C], depth[C]], color=color, lw=2.4 if ok else 2.0,
                ls="-" if ok else (0, (4, 2)), solid_capstyle="butt", zorder=2)
        ax.plot([xs[0], xs[0]], [0, depth[C]], color="#d0cfca", lw=.7, zorder=1)
        ax.plot([xs[-1], xs[-1]], [0, depth[C]], color="#d0cfca", lw=.7, zorder=1)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_xticks(range(n)); ax.set_xticklabels([str(m+1) for m in order], fontsize=5)
    ax.set_yticks([]); ax.set_ylim(-.8, top+.5); ax.set_xlim(-.9, n-.1)
    for s in ("top", "left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURFACE)


def order_from(truth, n):
    """Meter order that keeps the true groups together (depth-first, largest group first)."""
    out = []

    def walk(node):
        kids = [C for C in truth if set(C) < set(node) and not any(set(C) < set(D) < set(node) for D in truth)]
        covered = set().union(*map(set, kids)) if kids else set()
        for item in sorted([("g", C) for C in kids]+[("m", (m,)) for m in node if m not in covered],
                           key=lambda t: (-len(t[1]), t[1])):
            walk(item[1]) if item[0] == "g" else out.extend(item[1])
    walk(tuple(range(n)))
    return out


def main():
    rows, table = [], []
    for name in CASES:
        a = analyse(name)
        rows.append(a)
        T = a["truth"]
        for label, got in (("ours", a["ours"]), ("RNJ 0.08", a["rnj"])):
            table.append(dict(case=name, method=label, meters=a["n"], truth=len(T), found=len(got & T), false=len(got-T),
                              root_found=sum(len(C) > a["n"]/2 for C in got & T), root_truth=sum(len(C) > a["n"]/2 for C in T)))
        print(name, {k: v for k, v in table[-2].items() if k != "case"}, "|", table[-1], "| certified", a["certified"],
              "evidence", round(a["evidence"], 1), "source", a["source"] and (a["source"]["where"], a["source"]["share"]), flush=True)
    (build.OUT/"case_study.json").write_text(json.dumps(dict(table=table), default=str, indent=1))
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.6*len(rows)), facecolor=SURFACE)
    for row, a in zip(axes, rows):
        n, order = a["n"], order_from(a["truth"], a["n"])
        src = a["source"]
        where = f"source at {src['where']}, {int(100*src['share'])}%" if src else "no source"
        path_truth = {tuple(C) for C in (src["path"] if src else [])}
        panel(row[0], f"{a['name']}: truth ({n} meters, {where})", a["truth"], a["truth"], order, n, path=path_truth)
        panel(row[1], f"ours: {len(a['ours'] & a['truth'])} of {len(a['truth'])} true lines found, {len(a['ours']-a['truth'])} false", a["ours"], a["truth"], order, n,
              path={C for C in a["path"] if len(C) < n})
        panel(row[2], f"RNJ (tolerance 0.08): {len(a['rnj'] & a['truth'])} found, {len(a['rnj']-a['truth'])} false", a["rnj"], a["truth"], order, n)
    fig.legend(handles=[Line2D([], [], color=RIGHT, lw=2.5, label="true line, found (left: true line)"),
                        Line2D([], [], color=WRONG, lw=2, ls=(0, (4, 2)), label="false line"),
                        Line2D([], [], color=PATH, lw=2.5, label="line on the source's path (truth: true path; ours: returned path)"),
                        Line2D([], [], color=INK, marker="o", ls="", ms=4, label="meter")], loc="upper center", ncol=2,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.savefig(build.OUT/"case_study.png", dpi=110, facecolor=SURFACE)
    print("saved", build.OUT/"case_study.png")


if __name__ == "__main__":
    main()
