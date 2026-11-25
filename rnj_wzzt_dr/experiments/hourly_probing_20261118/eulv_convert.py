"""Convert feeders of the Non-Synthetic European LV Test System into this study's case format.

    python -m rnj_wzzt_dr.experiments.hourly_probing_20261118.eulv_convert <RunDss folder> [min_terms max_terms]

(run from the repository root; the downloaded folder is only read, never put on the import path)

Source: Koirala, Suarez-Ramon, Mohamed, Arboleya, "Non-synthetic European low voltage test system",
Int. J. Electr. Power Energy Syst. 118 (2020) 105712; data DOI 10.17632/685vgp64sm.1, CC BY 4.0
(REAL_NETWORK_CASE.md). The OpenDSS text tables are read as data; nothing in the package is executed.

Per feeder (the subtree behind one line leaving a transformer's LV bus, with normally-open fuses open):
- terminals are the building service connections (buses with customers); each meters its customers' sum;
- branches without customers downstream are dropped, and chains through hidden degree-2 nodes are merged
  (series R and X added), so hidden nodes are branching points;
- R and X are the phase values of the line codes (they carry no mutual terms), i.e. the balanced
  single-phase equivalent decided for this study;
- loads: the customers' hourly smart-meter readings (20 days, 481 points), summed per terminal; Q follows
  from the dataset's power factor 0.95.
Network tables go to cases_eulv/<case>/ (study folder); loads go to data/nonsynthetic_eu_lv_20261123/loads/
(outside Git).
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import scenario as sc

PF = .95
DATASET = dict(name="Non-Synthetic European Low Voltage Test System", doi="10.17632/685vgp64sm.1",
               paper="Koirala et al., Int. J. Electr. Power Energy Syst. 118 (2020) 105712",
               licence="CC BY 4.0", operator="EDP (Spain)")
LINE = re.compile(r"New Line\.(\S+)\s+Bus1=([^.\s]+)\S*\s+Bus2=([^.\s]+)\S*.*?Linecode=(\S+)\s+Length=([\d.]+)", re.I)


def _lines(path, keep_open=False):
    out = []
    for raw in path.read_text(errors="replace").splitlines():
        s = raw.strip()
        if not s or (s.startswith("!") and not keep_open):
            continue
        m = LINE.search(s.lstrip("!"))
        if m:
            out.append(dict(name=m.group(1), a=m.group(2), b=m.group(3), code=m.group(4), length=float(m.group(5)),
                            open=s.startswith("!")))
    return out


def read_tables(R):
    R = Path(R)
    txt = (R/"linecode.txt").read_text()
    codes = {}
    for m in re.finditer(r"New LineCode\.(\S+)(.*?)(?=New LineCode|\Z)", txt, re.S | re.I):
        r = re.search(r"rmatrix\s*=\s*\[\s*([\d.]+)", m.group(2)); x = re.search(r"xmatrix\s*=\s*\[\s*([\d.]+)", m.group(2))
        codes[m.group(1)] = (float(r.group(1)), float(x.group(1)))            # ohm/km, phase conductor
    lines = _lines(R/"Line_indexed_check.txt")+[f for f in _lines(R/"fuse_indexed.txt")]
    tx = {}
    for raw in (R/"Transformer_ind.txt").read_text().splitlines():
        m = re.search(r"Transformer\.(\S+).*?Buses=\[\S+\s+([^.\]\s]+).*?kVAs=\[(\d+)", raw)
        if m:
            tx[m.group(2)] = (m.group(1), float(m.group(3)))
    shapes = {}
    for raw in (R/"Loadshape.txt").read_text().splitlines():
        m = re.search(r"Loadshape\.(\S+).*?file=([^)]+)\)", raw, re.I)
        if m:
            shapes[m.group(1).lower()] = re.split(r"[\\/]", m.group(2).strip())[-1]
    loads = collections.defaultdict(list)
    for raw in (R/"Load_indexed.txt").read_text().splitlines():
        m = re.search(r"Load\.(\S+).*?Bus1=([^.\s]+)(\S*).*?daily=(\S+)", raw, re.I)
        if m:
            loads[m.group(2)].append(dict(name=m.group(1), three_phase=m.group(3).count(".") >= 4,
                                          shape=shapes[m.group(4).lower()]))
    return codes, lines, tx, loads


def feeder_tree(lv, first, adj):
    """Subtree behind line `first` leaving LV bus `lv`: parent and incoming edge of every bus."""
    nb = first["b"] if first["a"] == lv else first["a"]
    parent, edge, order = {nb: lv}, {nb: first}, [nb]
    for u in order:
        for e in adj[u]:
            v = e["b"] if e["a"] == u else e["a"]
            if v == parent.get(u) or v == lv or v in parent:
                continue
            parent[v] = u; edge[v] = e; order.append(v)
    return parent, edge, order


def reduce_tree(lv, parent, edge, order, loads, codes):
    """Drop branches without customers; merge hidden degree-2 chains. Returns children map and edges."""
    has = {u: u in loads for u in order}
    for u in reversed(order):
        if has[u]:
            has[parent[u]] = has.get(parent[u], False) or True
    keep = [u for u in order if has[u]]
    z = {u: (codes[edge[u]["code"]][0]*edge[u]["length"]/1000., codes[edge[u]["code"]][1]*edge[u]["length"]/1000.,
             edge[u]["length"]) for u in keep}
    kids = collections.defaultdict(list)
    for u in keep:
        kids[parent[u]].append(u)
    # merge: a hidden node with one child is removed, its edge added to the child's
    par = {u: parent[u] for u in keep}
    changed = True
    while changed:
        changed = False
        for u in list(par):
            if u not in loads and len(kids[u]) == 1:
                c = kids[u][0]; p = par[u]
                z[c] = tuple(a+b for a, b in zip(z[c], z[u]))
                kids[p][kids[p].index(u)] = c; par[c] = p
                del par[u], z[u], kids[u]
                changed = True
    return kids, par, z


def _units(lv, adj, whole_transformer):
    """One unit per feeder, or (whole_transformer) one unit with every feeder of the transformer under its LV bus."""
    trees = [(first["name"], *feeder_tree(lv, first, adj)) for first in adj[lv]]
    if not whole_transformer:
        yield from ((dict(name=nm), p, e, o) for nm, p, e, o in trees)
        return
    parent, edge, order = {}, {}, []
    for _, p, e, o in trees:
        if set(p) & set(parent):
            raise ValueError(f"feeders of LV bus {lv} overlap")
        parent.update(p); edge.update(e); order += o
    yield dict(name="all"), parent, edge, order


def convert(R, out_cases, out_loads, lo=8, hi=25, whole_transformer=False, index_name="index.json"):
    """Write every feeder (or, with whole_transformer, every transformer's whole LV network) with lo..hi terminals.

    The study's 33 feeders use the defaults. index_name keeps other selections from overwriting their index."""
    codes, lines, tx, loads = read_tables(R)
    adj = collections.defaultdict(list)
    for e in lines:
        adj[e["a"]].append(e); adj[e["b"]].append(e)
    profile_dir = Path(R)/"day_20_profile"
    index = []
    for lv, (tname, kva) in sorted(tx.items(), key=lambda kv: int(kv[0])):
        for first, parent, edge, order in _units(lv, adj, whole_transformer):
            kids, par, z = reduce_tree(lv, parent, edge, order, loads, codes)
            bfs, i = [lv], 0
            while i < len(bfs):
                bfs += sorted(kids.get(bfs[i], []), key=lambda b: (b not in loads, b)); i += 1
            terms = [b for b in bfs if b in loads]
            if not lo <= len(terms) <= hi:
                continue
            case = f"eulv_{tname.lower()}_{first['name'].lower()}"
            ids = {b: k+1 for k, b in enumerate(bfs)}
            p_kw = np.column_stack([sum(np.loadtxt(profile_dir/c["shape"]) for c in loads[b]) for b in terms])
            mean_kw = p_kw.mean(axis=0)
            d = out_cases/case
            d.mkdir(parents=True, exist_ok=True)
            with (d/"buses.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["bus_id", "original_bus_id", "bus_type", "pd_kw", "qd_kvar", "is_observed", "has_load",
                            "is_terminal", "is_original_case33_bus", "hidden_degree", "note", "customers"])
                for b in bfs:
                    deg = len(kids.get(b, []))+(b != lv)
                    if b == lv:
                        w.writerow([1, b, "root", 0., 0., True, False, False, False, deg, f"LV bus of {tname}", 0])
                    elif b in loads:
                        k = terms.index(b); n3 = sum(c["three_phase"] for c in loads[b])
                        w.writerow([ids[b], b, "terminal", round(float(mean_kw[k]), 4),
                                    round(float(mean_kw[k]*np.tan(np.arccos(PF))), 4), True, True, True, False, deg,
                                    f"service connection, {len(loads[b])} customers ({n3} three-phase)"
                                    + ("; metered internal node" if kids.get(b) else ""), len(loads[b])])
                    else:
                        w.writerow([ids[b], b, "hidden_internal", 0., 0., False, False, False, False, deg,
                                    "hidden branching node", 0])
            with (d/"branches.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["from_bus", "to_bus", "r_ohm", "x_ohm", "status", "branch_type", "original_branch_id",
                            "length_m", "is_candidate", "is_true_closed"])
                for b in bfs[1:]:
                    r, x, L = z[b]
                    w.writerow([ids[par[b]], ids[b], round(r, 6), round(x, 6), True,
                                "service" if b in loads and not kids.get(b) else "backbone", f"{par[b]}-{b}",
                                round(L, 2), True, True])
            internal = sum(1 for b in terms if kids.get(b))
            meta = dict(root_bus=1, base_kv=.4, base_mva=.2, transformer_kva=kva,
                        source=dict(DATASET, transformer=tname, feeder_line=first["name"], lv_bus=lv,
                                    terminals=len(terms), hidden_nodes=len(bfs)-1-len(terms),
                                    metered_internal_nodes=internal, customers=int(sum(len(loads[b]) for b in terms)),
                                    farthest_m=round(max(_path(b, par, z, 2) for b in terms), 1),
                                    farthest_r_ohm=round(max(_path(b, par, z, 0) for b in terms), 5),
                                    load_resolution="hourly, 20 days (481 points), kW", power_factor=PF,
                                    note="balanced single-phase equivalent; phase R and X of the line codes"))
            (d/"metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            out_loads.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_loads/f"{case}.npz", p_kw=p_kw, power_factor=PF,
                                customers=np.array([len(loads[b]) for b in terms]),
                                gis_bus=np.array(terms))
            index.append(dict(case=case, **meta["source"], transformer_kva=kva))
            print(f"{case}: {len(terms)} terminals, {meta['source']['hidden_nodes']} hidden, "
                  f"{meta['source']['customers']} customers, internal metered {internal}")
    (out_cases/index_name).write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def _path(b, par, z, k):
    s = 0.
    while b in par:
        s += z[b][k]; b = par[b]
    return s


if __name__ == "__main__":
    lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (8, 25)
    idx = convert(sys.argv[1], sc.EULV_CASES, sc.EULV_LOADS, lo, hi)
    print(len(idx), "feeders converted")
