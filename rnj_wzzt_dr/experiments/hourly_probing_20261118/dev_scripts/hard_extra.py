import numpy as np
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import load, _mcnemar_p
from rnj_wzzt_dr.experiments.hourly_probing_20261118.confirm import _paired

C = ('hard_short01', 'hard_short005', 'hard_noisy')
ARMS = ('passive_bic', 'bgreedy_P', 'bgreedy_lowV_P', 'qbc_P', 'cert_P', 'random_matched_P')
allr = {c: [r for r in load(c) if r['stratum'] == 'S1' and not r['input'].endswith('_missing')] for c in C}
ok = lambda r: r['outcome'] == 'correct'
for a, b in [('qbc_P', 'cert_P'), ('bgreedy_lowV_P', 'cert_P'), ('bgreedy_lowV_P', 'random_matched_P'),
             ('bgreedy_lowV_P', 'qbc_P'), ('random_matched_P', 'passive_bic')]:
    g = l = n = 0
    for c in C:
        t = _paired([r for r in allr[c] if r['arm'] == a], [r for r in allr[c] if r['arm'] == b], ok)
        g += t['gain']; l += t['loss']; n += t['pairs']
    print(f'{a} vs {b}: {g}/{l} of {n}, p={_mcnemar_p(g, l):.3g}')
print()
for c in C + ('pooled',):
    rs = sum(allr.values(), []) if c == 'pooled' else allr[c]
    base = sum(ok(r) for r in rs if r['arm'] == 'passive_bic')
    for a in ARMS:
        v = [r for r in rs if r['arm'] == a]
        wv = [r['wrong_vdev'] for r in v if r['ever_confirmed_wrong'] and r.get('wrong_vdev') is not None]
        fw = [r for r in v if r['outcome'] == 'wrong']
        pd = np.mean([r['probed_hours'] > 0 for r in v])
        down = sum(sum(-min(x, 0) for x in b['u_kw']) for r in v for b in r['blocks'])
        tot = sum(sum(abs(x) for x in b['u_kw']) for r in v for b in r['blocks'])
        spend = sum(r['cost_cny'] for r in v)
        net = sum(ok(r) for r in v) - base
        per = spend / net if net > 0 else float('nan')
        print(f"{c:14s} {a:17s} final wrong {len(fw):3d}  wrong>0.2% {sum(x > 2e-3 for x in wv):3d}  wrong>0.1% {sum(x > 1e-3 for x in wv):3d}  "
              f"probed days {100 * pd:3.0f}%  reductions {100 * down / tot if tot else 0:3.0f}%  net extra correct {net:3d}  CNY per extra correct {per:6.0f}")
    print()
r0 = allr['hard_short01'][0]
print(sorted(r0.keys()))
blk = next((b for r in allr['hard_short01'] for b in r['blocks']), None)
print(blk)
