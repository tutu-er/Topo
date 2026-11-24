import json, sys
from collections import Counter
rows = json.load(open(sys.argv[1]))
ok = [r for r in rows if 'error' not in r]
print('errors', [(r['name'], r['error'][:200]) for r in rows if 'error' in r])
for key in ('stratum', 'identifier_bank_has_truth', 'validation_mse_correct', 'leader_correct'):
    print(key, sorted(Counter((r['case'], r[key]) for r in ok).items()))
print('bank sizes', sorted(r['bank'] for r in ok))
for r in ok:
    if r['stratum'] != 'S2':
        print(' ', r['name'], r['history'], r['stratum'], 'leader', r['leader_type'], 'near', r['near_types'], 'Emin', None if r['Emin'] is None else round(r['Emin'], 2), 'neq', r['n_equivalent'])
