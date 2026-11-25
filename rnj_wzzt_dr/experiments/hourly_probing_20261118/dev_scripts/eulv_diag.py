"""Why is the truth missing from the real-feeder banks? Compare the truth, its contractions (tiny segments
merged) and the bank leader on the passive history."""
import sys
sys.path.insert(0, r"C:/0-workspace/Topo/repo")
import numpy as np
from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, model, scores
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for

s = settings_for("eulv")
for nm in ("eulv_td009116_feeder106_s0", "eulv_td009116_feeder106_s9", "eulv_td008690_feeder21_s0",
           "eulv_td008690_feeder21_s9"):
    inp = bank.load(ROOT/"eulv"/"inputs"/f"{nm}.pkl")
    n, net, labels = inp.n, inp.network, inp.labels
    zb = net["base_kv"]**2/net["base_mva"]; r_min = 2*s.r_min_ohm/zb
    order, parent, incoming = backend._radial_order(net)
    desc = {b: set() for b in order}
    for i, l in enumerate(labels):
        desc[l].add(i)
    for ch in reversed(order[1:]):
        desc[parent[ch]].update(desc[ch])
    r_of = {tuple(sorted(desc[ch])): incoming[ch]["r_ohm"] for ch in order[1:] if len(desc[ch]) > 1}
    truth = model.canonical_family(sc.truth_family(net, labels), n)
    fams = {"truth": truth}
    for thr in (.003, .01):
        fams[f"truth, segments < {thr} merged"] = model.canonical_family(
            [C for C in truth if len(C) == 1 or r_of.get(C, 1.) >= thr], n)
    bank_keys = {model.nonsingleton(f) for f in inp.families}
    fits, J, _ = bank.passive_fit(list(fams.values()), n, inp.hist, r_min, s.upper_bound, offsets=True, whiten=True)
    bf, bJ, _ = bank.passive_fit(inp.families, n, inp.hist, r_min, s.upper_bound, offsets=True, whiten=True)
    L = int(np.argmin(bJ))
    V0 = inp.day["root"]**2
    v_true = backend._ac_voltages(net, labels, inp.day["P"], inp.day["Q"], inp.day["root"])[0]
    vL = scores.voltages(model.Candidate(inp.families[L], n), bf[L].theta, inp.day["P"], inp.day["Q"], V0, True)
    print(f"\n{nm}: n {n}, bank {len(inp.families)}, closure {inp.meta.get('closure', {}).get('capped')}, "
          f"internal segments {len(r_of)}, below 0.003 ohm {sum(r < .003 for r in r_of.values())}, "
          f"below 0.01 {sum(r < .01 for r in r_of.values())}")
    print(f"  bank leader: J {bJ[L]:.1f}, supports {len(model.nonsingleton(inp.families[L]))}, "
          f"max |V - V_true| over the new day {np.max(np.abs(vL-v_true)):.5f} pu")
    for (name, f), j, ft in zip(fams.items(), J, fits):
        vk = scores.voltages(model.Candidate(f, n), ft.theta, inp.day["P"], inp.day["Q"], V0, True)
        print(f"  {name:32s}: supports {len(model.nonsingleton(f)):2d}, in bank {model.nonsingleton(f) in bank_keys}, "
              f"J - J_leader {j-bJ[L]:+7.1f}, max |V - V_true| {np.max(np.abs(vk-v_true)):.5f} pu")
