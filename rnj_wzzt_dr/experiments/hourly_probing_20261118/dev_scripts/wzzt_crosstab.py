import json
from collections import Counter
R = json.load(open(r"C:/0-workspace/Topo/repo/outputs/hourly_probing_dev_20261120/eulv/wzzt_criterion.json"))
c = Counter()
for r in R:
    for b, wl, wd, fl, fd in zip(r["bic_8"], r["wzzt_lv"], r["wzzt_dm"], r["floor_lv"], r["floor_dm"]):
        c[("BIC supports" if b else "BIC rejects", "WZZT-dm supports" if wd else "WZZT-dm rejects",
           "survives floor" if fd else "floor removes")] += 1
for k, v in sorted(c.items()):
    print(k, v)
b = sum(sum(r["bic_8"]) for r in R)
lost = sum(bb and not fd for r in R for bb, fd in zip(r["bic_8"], r["floor_dm"]))
lost2 = sum(bb and not (wd and fd) for r in R for bb, wd, fd in zip(r["bic_8"], r["wzzt_dm"], r["floor_dm"]))
print(f"BIC-supported segments {b}: removed by the 0.08 floor {lost}; lost by WZZT-dm criterion or floor {lost2}")
gain18 = sum((not a) and bb for r in R for a, bb in zip(r["bic_8"], r["bic_18"]))
loss18 = sum(a and not bb for r in R for a, bb in zip(r["bic_8"], r["bic_18"]))
print(f"8 -> 18 days: newly supported {gain18}, no longer supported {loss18}")
