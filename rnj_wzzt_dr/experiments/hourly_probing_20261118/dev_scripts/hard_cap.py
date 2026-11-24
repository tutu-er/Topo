import collections
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import load

C = ('hard_short01', 'hard_short005', 'hard_noisy')
over, hourly_over = [], 0
types = collections.Counter(); final = collections.Counter()
for c in C:
    for r in load(c):
        if r['stratum'] != 'S1':
            continue
        for b in r['blocks']:
            if b['cost'] > 20.0 + 1e-6:
                hourly_over += 1
        if r['cost_cny'] > 100.0 + 1e-6:
            over.append((c, r['input'], r['path'], r['arm'], round(r['cost_cny'], 2), len(r['blocks']),
                         [round(b['cost'], 2) for b in r['blocks']], round(sum(b['cost'] for b in r['blocks']), 2)))
        if r['arm'] == 'bgreedy_lowV_P':
            for t in r.get('ever_wrong_types') or []:
                types[(c, t)] += 1
            final[(c, r['outcome'], r.get('final_type'))] += 1
print('blocks over 20 CNY:', hourly_over)
print('runs over 100 CNY:', len(over))
for o in over:
    print(o)
print()
for k, v in sorted(types.items()):
    print('ever wrong type', k, v)
for k, v in sorted(final.items(), key=str):
    print('final', k, v)
