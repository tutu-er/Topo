"""Check the residual covariance of the true model: is the root-meter error shared across terminals?"""
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, config, scenario as sc
s = config.Settings(meter_bias_sigma=.0005, meter_offsets=True)
for name in ("soumalas11_s1", "pengwah18_s2", "flynn16_s3"):
    inp = bank.load(rf"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120\v2_bias_typical\inputs\{name}.pkl")
    n = inp.n; zb = .8
    truth = model.canonical_family(sc.truth_family(inp.network, inp.labels), n)
    f = bank.fit_one(truth, n, inp.hist, 2*.003/zb, 2., 3, True, True)
    c = model.Candidate(truth, n)
    lam = c.loss_term(f.theta, inp.hist["P"], inp.hist["Q"], inp.hist["V0sq"])
    res = inp.hist["Y"]-lam-c.linear(f.theta, inp.hist["P"], inp.hist["Q"])-f.offset
    C = np.cov(res.T)
    off = C[~np.eye(n, dtype=bool)].mean(); diag = np.diag(C).mean()
    print(name, "n", n, "mean var %.3g" % diag, "mean cov %.3g" % off, "shared share %.2f" % (off/diag),
          "expected terminal var %.3g" % (4*s.voltage_sigma**2*np.mean(inp.hist["Y"]*0+1)))
