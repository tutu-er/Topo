from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, scenario as sc
for case in ("soumalas11", "pengwah18", "flynn16", "paper15"):
    net, labels, _, _ = sc.network_info(case); n = len(labels)
    truth = model.canonical_family(sc.truth_family(net, labels), n); key = model.nonsingleton(truth)
    res = {}
    for mc in (5, 8):
        miss = 0; tot = 0
        for C in key:
            if len(C) == n: continue
            tot += 1
            merged = model.canonical_family([S for S in truth if S != C], n)
            keys = {model.nonsingleton(model.canonical_family(f, n)) for f in bank.neighbours(merged, n, max_children=mc)}
            miss += key not in keys
        res[mc] = f"{miss}/{tot} unreachable"
    print(case, res)
