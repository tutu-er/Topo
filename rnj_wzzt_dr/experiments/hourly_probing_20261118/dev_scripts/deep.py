import json, sys, numpy as np
from collections import Counter, defaultdict
recs = [json.loads(l) for l in open(sys.argv[1])]
def arm(a, st=None): return [r for r in recs if r['arm']==a and (st is None or r['stratum'] in st)]
# 1 planner reasons at gates where t_opt planned
for st in ('S1','S3'):
    rs = arm('t_opt_P', (st,))
    c = Counter(b['reason'] for r in rs for b in r['blocks'])
    print(st, 'plan reasons', dict(c))
    parts = Counter()
    for r in rs:
        for b in r['blocks']:
            if b['parts']:
                parts[min(b['parts'], key=b['parts'].get)] += 1
    print(st, 'binding window part', dict(parts))
    ratio = [min(q/b['eta'][j] for j,q in b['q_planned'].items()) for r in rs for b in r['blocks'] if b['q_planned'] and all(b['eta'][j]>0 for j in b['q_planned'])]
    if ratio: print(st, 'planned q/eta at ceiling: median %.3f p90 %.3f max %.3f' % tuple(np.percentile(ratio,[50,90,100])))
    act = Counter(t for r in rs for b in r['blocks'] for t in b['active_types'])
    bey = Counter(t for r in rs for b in r['blocks'] for t in b['beyond_types'])
    print(st, 'active types', dict(act), 'beyond types', dict(bey))
    # realised change in E vs planned q
    dE = [(b['q_planned'][j], b['E_after'][j]-b['E_seal'][j]) for r in rs for b in r['blocks'] if b.get('E_after') for j in b['q_planned'] if j in b['E_after'] and j in b['E_seal']]
    if dE:
        q, d = np.array(dE).T
        print(st, 'planned q (2h) median %.2f; realised dE over 1h median %.2f mean %.2f; frac dE<0 %.2f' % (np.median(q), np.median(d), d.mean(), (d<0).mean()))
    bias = defaultdict(list)
    for r in rs:
        for b in r['blocks']:
            for k,v in (b.get('bias') or {}).items():
                bias[(v['type'], k.split('_')[1])].append((v['whole'], v['probe']))
    for k,v in sorted(bias.items()):
        v=np.array(v); print(st, 'b/J', k, 'n', len(v), 'whole |>0.5| %.2f median %.2f' % ((abs(v[:,0])>.5).mean(), np.median(v[:,0])), 'probe |>0.5| %.2f median %.2f' % ((abs(v[:,1])>.5).mean(), np.median(v[:,1])))
# 2 S1 ever wrong inputs
for a in ('passive_bic','t_opt_P'):
    print(a, 'S1 ever wrong', Counter((r['input'], tuple(r['ever_wrong_types'])) for r in arm(a,('S1',)) if r['ever_confirmed_wrong']))
    print(a, 'S1 not correct', Counter((r['input'], r['outcome']) for r in arm(a,('S1',)) if r['outcome']!='correct'))
# 3 S3 per input outcome t_opt vs passive
for a in ('passive_bic','t_opt_P','check_then_t_opt'):
    print(a, 'S3 by input', sorted(Counter((r['input'], r['outcome'][:9]) for r in arm(a,('S3',))).items()))
# 4 gate timing and violations
secs = [g['seconds'] for r in arm('t_opt_P') for g in r['gates']]
print('t_opt gate seconds mean %.2f p95 %.2f max %.2f' % (np.mean(secs), np.percentile(secs,95), max(secs)))
print('AC band hours', sum(r['violations']['ac_band_hours'] for r in recs), 'deadband hours', sum(r['violations']['deadband_hours'] for r in recs), 'vmin %.4f vmax %.4f' % (min(r['violations']['min_v'] for r in recs), max(r['violations']['max_v'] for r in recs)))
# 5 time of first trigger and kappa
kap = [b['kappa'] for r in arm('t_opt_P') for b in r['blocks'] if b['kappa']>0]
print('executed kappa median %.2f max %.2f' % (np.median(kap), max(kap)))
kw = [max(abs(x) for x in b['u_kw']) for r in arm('t_opt_P') for b in r['blocks'] if b['kappa']>0]
print('executed block max kW median %.2f max %.2f' % (np.median(kw), max(kw)))
