from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import load

rs = [r for c in ('hard_short01', 'hard_short005', 'hard_noisy') for r in load(c)
      if r['stratum'] == 'S1' and r['arm'] != 'passive_bic']
for lim in (60, 80, 90, 100):
    v = [r for r in rs if r['cost_cny'] > lim]
    print(f"> {lim} CNY: {len(v)} of {len(rs)} runs; correct {sum(r['outcome'] == 'correct' for r in v)}")
