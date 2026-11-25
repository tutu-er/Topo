"""Checks of the converted real feeders: AC on the real loads, voltage range, internal segment sizes."""
import json, sys
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc, model

idx = json.loads((sc.EULV_CASES/"index.json").read_text())
rows = []
for c in idx:
    case = c["case"]
    net = sc.load_network(case)
    net_, labels, pn, qn = sc.network_info(case, net)
    n = len(labels)
    P, Q, root = sc.eulv_load(case, 0, 20, net, 1)
    v, info = backend._ac_voltages(net, labels, P, Q, root)
    fam = model.canonical_family(sc.truth_family(net, labels), n)
    order, parent, incoming = backend._radial_order(net)
    desc = {b: set() for b in order}
    for i, l in enumerate(labels):
        desc[l].add(i)
    for ch in reversed(order[1:]):
        desc[parent[ch]].update(desc[ch])
    internal_r = [incoming[ch]["r_ohm"] for ch in order[1:] if len(desc[ch]) > 1]
    peak_kw = float((P.sum(axis=1)*net["base_kw"]).max())
    rows.append(dict(case=case, n=n, nonsingleton=len(model.nonsingleton(fam)), vmin=float(v.min()), vmax=float(v.max()),
                     spread=float((v.max(axis=1)-v.min(axis=1)).max()), peak_kw=peak_kw, kva=net["metadata"]["transformer_kva"],
                     r_internal_min=min(internal_r) if internal_r else None,
                     below_floor=sum(r < .003 for r in internal_r), below_001=sum(r < .01 for r in internal_r),
                     internal=len(internal_r)))
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
a = np.array([[r["vmin"], r["vmax"], r["spread"], r["peak_kw"]] for r in rows])
print(f"\nvoltage min over all feeders and hours {a[:,0].min():.4f}, max {a[:,1].max():.4f}; largest spread within an hour "
      f"{a[:,2].max():.4f} (median {np.median(a[:,2]):.4f}); peak feeder load {a[:,3].min():.0f}-{a[:,3].max():.0f} kW")
tot_int = sum(r["internal"] for r in rows)
print(f"internal segments {tot_int}: below 0.003 ohm (the floor) {sum(r['below_floor'] for r in rows)}, "
      f"below 0.01 ohm {sum(r['below_001'] for r in rows)}")
print(f"feeders with vmin < 0.95: {sum(r['vmin'] < .95 for r in rows)}; with vmin < 0.96 (planner band): {sum(r['vmin'] < .96 for r in rows)}")
