import sys; sys.path.insert(0, r"C:\0-workspace\Topo\repo")
import numpy as np
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc, model, unbalanced
for case in ("soumalas11", "pengwah18"):
    net, labels, pn, qn = sc.network_info(case)
    P, Q, root = sc.generate_load(case, 1, 1)
    P *= 1.4; Q *= 1.4
    v, _ = backend._ac_voltages(net, labels, P, Q, root)
    y = root[:, None]**2 - v**2
    fam = sc.truth_family(net, labels)
    cand = model.Candidate(fam, len(labels))
    order, parent, incoming = backend._radial_order(net)
    zb = net["base_kv"]**2/net["base_mva"]
    desc = {b: set() for b in order}
    for i, l in enumerate(labels): desc[l].add(i)
    for c in reversed(order[1:]): desc[parent[c]].update(desc[c])
    theta = np.zeros(2*cand.S)
    for c in order[1:]:
        k = cand.family.index(tuple(sorted(desc[c])))
        theta[k] += 2*incoming[c]["r_ohm"]/zb; theta[cand.S+k] += 2*incoming[c]["x_ohm"]/zb
    lin = cand.linear(theta, P, Q); full = cand.predict(theta, P, Q, root**2)
    g0 = np.abs(y-lin).max(); g1 = np.abs(y-full).max()
    print(case, "max gap linear %.2e  with loss %.2e  reduction %.1f%%" % (g0, g1, 100*(1-g1/g0)),
          "rms", np.sqrt(np.mean((y-lin)**2)), np.sqrt(np.mean((y-full)**2)))
    tp = unbalanced.ThreePhase(net, labels, equal=True)
    v3 = tp.terminal_voltages(P, Q, root)
    print("   three-phase equal split vs backend max diff %.2e" % np.abs(v3-v).max())
    M = cand.sensitivity(theta, P[:2], Q[:2], root[:2]**2)
    R = (cand.Z.T*theta[:cand.S])@cand.Z
    print("   J/R head fraction", np.round((M[0]-R).max()/R.max(), 4))
