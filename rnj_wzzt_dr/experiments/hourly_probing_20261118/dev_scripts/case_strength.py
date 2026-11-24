"""RX75 strength of every true non-singleton branch on the four cases, with true R/X.

s_C = (a r_C + b x_C)/m, a = .75/mu_R, b = .25/mu_X, m = median diag(aR+bX).
Also: the shortest-line resistance that still passes 0.08, and the voltage
signature of each internal line (max |dV| it causes at peak load).
"""
import json
import numpy as np
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc

out = {}
for case in ("soumalas11", "pengwah18", "flynn16", "paper15"):
    net = backend.load_case(case)
    net, labels, p_nom, q_nom = sc.network_info(case, net)
    n = len(labels)
    zb = net["base_kv"]**2/net["base_mva"]
    order, parent, incoming = backend._radial_order(net)
    desc = {b: set() for b in order}
    for i, l in enumerate(labels):
        desc[l].add(i)
    for c in reversed(order[1:]):
        desc[parent[c]].update(desc[c])
    rows = []
    R = np.zeros((n, n)); X = np.zeros((n, n))
    for c in order[1:]:
        e = incoming[c]; C = sorted(desc[c])
        z = np.zeros(n); z[C] = 1
        r, x = 2*float(e["r_ohm"])/zb, 2*float(e["x_ohm"])/zb
        R += r*np.outer(z, z); X += x*np.outer(z, z)
        rows.append(dict(edge=[e["from_bus"], e["to_bus"]], type=e.get("branch_type"), size=len(C),
                         r_ohm=float(e["r_ohm"]), x_ohm=float(e["x_ohm"]), length_m=float(e.get("length_m") or 0), r=r, x=x, C=C))
    means = []
    for M in (R, X):
        d = np.diag(M)[:, None]+np.diag(M)[None, :]-2*M
        means.append(d[d > 0].mean())
    a, b = .75/means[0], .25/means[1]
    m = np.median(np.diag(a*R+b*X))
    root_path_ohm = np.diag(R)*zb/2
    for row in rows:
        row["s"] = (a*row["r"]+b*row["x"])/m
        # resistance at which this branch would sit exactly at 0.08 (same X/R)
        row["r_ohm_at_008"] = float(row["r_ohm"])*.08/row["s"]
    internal = [r for r in rows if 1 < r["size"] < n]
    common = [r for r in rows if r["size"] == n]
    out[case] = dict(n=n, zbase=zb, median_root_path_ohm=float(np.median(root_path_ohm)),
                     max_root_path_ohm=float(root_path_ohm.max()), a=a, b=b, m=m,
                     peak_kw=float(p_nom.sum()*net["base_kw"]),
                     internal=[{k: v for k, v in r.items() if k not in ("r", "x", "C")} for r in internal],
                     common=[{k: v for k, v in r.items() if k not in ("r", "x", "C")} for r in common],
                     service_xr=[r["x_ohm"]/r["r_ohm"] for r in rows if r["size"] == 1][:3],
                     backbone_ohm_per_km=[1000*r["r_ohm"]/r["length_m"] for r in rows if r["length_m"]][:2])
    print(f"\n=== {case}: n={n}, median root path R={out[case]['median_root_path_ohm']:.4f} ohm, max {out[case]['max_root_path_ohm']:.4f}")
    for r in internal+common:
        print(f"  {str(r['edge']):12s} {r['type']:9s} |C|={r['size']:2d} L={r['length_m']:5.0f} m R={r['r_ohm']:.4f} s={r['s']:.3f}  R@0.08={r['r_ohm_at_008']:.4f} ohm")
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
