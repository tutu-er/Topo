import numpy as np
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
for case in ("soumalas11", "pengwah18", "flynn16", "paper15"):
    net = backend.load_case(case)
    net, labels, p, q = sc.network_info(case, net)
    P, Q, root = sc.generate_load(case, 1, 3, network=net)
    v, _ = backend._ac_voltages(net, labels, P, Q, root)
    drop = root[:, None]-v
    nominal = backend._ac_voltages(net, labels, p[None], q[None], np.array([1.02]))[0]
    lengths = [float(b["length_m"]) for b in net["branches"]]
    print(f"{case:10s} n={len(labels):2d} peak sum P={p.sum()*net['base_kw']:.0f} kW  sim max drop={100*drop.max():.2f}%  "
          f"drop at nominal peak={100*(1.02-nominal.min()):.2f}%  total line length={sum(lengths):.0f} m  base_kw={net['base_kw']}")
