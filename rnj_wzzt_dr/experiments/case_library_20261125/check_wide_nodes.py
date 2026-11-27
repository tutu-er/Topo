"""Development check (draft v14). Pricing at the wide nodes of found trees: enumeration (old rule, up to 12 children) against the node MIQP (new rule,
beyond K = 9 children). Same best line and gain expected when the MIQP proves optimality. Development twins only.
Run: python -m rnj_wzzt_dr.experiments.case_library_20261125.check_wide_nodes"""
import json
import time

from rnj_wzzt_dr.experiments.case_library_20261125.dev_cases import load_dev_case
from rnj_wzzt_dr.identification_v8.identify import place_load
from rnj_wzzt_dr.identification_v8.model import Model, balance, load_share, GATE
from rnj_wzzt_dr.identification_v8.search import best_addition, node_miqp
from rnj_wzzt_dr.identification_v8.sets import basis, canonical, children, lines_of

rows = list({json.loads(l)["name"]: json.loads(l)
             for l in open("C:/0-workspace/Topo/repo/outputs/case_library_20261125/dev_v8.jsonl", encoding="utf-8")}.values())
for r in rows:
    rec, inp = load_dev_case(r["name"]) if r["name"].startswith("dev_huge") else (None, None)
    if inp is None:
        continue
    h, n = inp.hist, inp.n
    u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"])
    u = u if load_share(h["P"], u) >= GATE else None
    model = Model(h["P"], h["Q"], h["Y"], h["V0sq"], u)
    meter_tree = [tuple(C) for C in r["tree"]]
    tree = list(place_load(model, meter_tree)[0]) if u is not None else lines_of(meter_tree, n)
    family = canonical([*basis(model.leaves), *tree])
    widths = sorted((len(k) for k in children(family, model.leaves).values()), reverse=True)
    if widths[0] <= 9:
        print(f"{r['name']}: widest node {widths[0]}, enumerated under both rules")
        continue
    t0 = time.perf_counter(); old, oi = best_addition(model, tree, max_children=12); t_old = time.perf_counter()-t0
    t0 = time.perf_counter(); new, ni = best_addition(model, tree, max_children=9, node_seconds=60.); t_new = time.perf_counter()-t0
    print(f"{r['name']}: widths {widths[:4]}, load {u is not None}")
    print(f"   enumeration : line {old}, gain {oi['gain']:.3e}, certified {oi['certified']}, {t_old:.1f} s")
    print(f"   node MIQP   : line {new}, gain {ni['gain']:.3e}, certified {ni['certified']}, by_miqp {ni['by_miqp']},"
          f" unproved {ni['unproved']}, {t_new:.1f} s")
    for node, kids in children(family, model.leaves).items():      # each wide node on its own
        if len(kids) > 9:
            t0 = time.perf_counter(); line, value, status = node_miqp(model, tree, kids, 60.)
            print(f"   node of {len(kids)} children: MIQP status {status}, line {line}, {time.perf_counter()-t0:.1f} s")
