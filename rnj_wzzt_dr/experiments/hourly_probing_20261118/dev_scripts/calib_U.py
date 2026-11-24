"""How loose is the strong greedy's risk bound U? Compare U after the chosen block with the
realised error of the MAP class one hour later (DEVELOPMENT_RUNS section 16 logs)."""
import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import load

C = ('hard_short01', 'hard_short005', 'hard_noisy')
rows = []
for arm in ('bgreedy_P', 'bgreedy_lowV_P'):
    for c in C:
        for r in load(c):
            if r['arm'] != arm or r['stratum'] != 'S1':
                continue
            gates = {g['g']: g for g in r['gates']}
            for b in r['blocks']:
                if b.get('cert_B') is None:
                    continue
                nxt = gates.get(b['hour']+1)
                if nxt is None or 'truth_eq' not in nxt:
                    continue
                wrong = not (nxt['leader_correct'] or nxt['truth_eq'])
                bought = any(abs(x) > 0 for x in b['u_kw'])
                rows.append((arm, b['cert_B0'], b['cert_B'], wrong, bought, b['cost']))
a = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in rows], float)
U0, U, wrong, bought, cost = a.T
print(f"{len(a)} planned gates; blocks bought at {int(bought.sum())}")
edges = [0, .02, .05, .1, .2, .4, .8, 1.6, np.inf]
print("| U after the chosen block | Gates | Mean U | Realised wrong-class rate one hour later | Ratio U / rate |")
print("| --- | ---: | ---: | ---: | ---: |")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (U >= lo) & (U < hi)
    if m.sum() < 20:
        continue
    rate = wrong[m].mean()
    print(f"| {lo:g}–{hi:g} | {int(m.sum())} | {U[m].mean():.3f} | {rate:.3f} | {U[m].mean()/rate if rate else float('inf'):.1f} |")
print(f"overall: mean U {U.mean():.3f}, realised {wrong.mean():.3f}")
# risk removed per CNY on bought gates
m = bought > 0
dU = U0[m]-U[m]
print(f"bought: median U0 {np.median(U0[m]):.3f}, median U {np.median(U[m]):.3f}, median dU {np.median(dU):.3f}, "
      f"median cost {np.median(cost[m]):.1f} CNY, median dU per CNY {np.median(dU/np.maximum(cost[m], 1e-9)):.4f}")
for arm in ('bgreedy_P', 'bgreedy_lowV_P'):
    sel = np.array([r[0] == arm for r in rows]) & (bought > 0)
    print(arm, 'blocks', int(sel.sum()), 'median cost', round(float(np.median(cost[sel])), 1),
          'median dU/cost', round(float(np.median((U0[sel]-U[sel])/cost[sel])), 4))
