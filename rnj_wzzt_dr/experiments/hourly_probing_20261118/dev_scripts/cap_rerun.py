"""Rerun the runs that broke the daily cap with the committed-block fix (DEVELOPMENT_RUNS section 16)."""
from concurrent.futures import ProcessPoolExecutor

from rnj_wzzt_dr.experiments.hourly_probing_20261118 import bank, run
from rnj_wzzt_dr.experiments.hourly_probing_20261118.experiments import ROOT, settings_for
from rnj_wzzt_dr.experiments.hourly_probing_20261118.report import load

CASES = [('hard_short005', 'flynn16_s53', 1, 'cert_P'), ('hard_short005', 'paper15_s59', 0, 'qbc_P'),
         ('hard_short005', 'pengwah18_s36', 1, 'cert_P'), ('hard_short005', 'pengwah18_s39', 1, 'cert_P'),
         ('hard_short005', 'pengwah18_s45', 0, 'cert_P'), ('hard_short005', 'pengwah18_s45', 0, 'bgreedy_P'),
         ('hard_short005', 'pengwah18_s46', 0, 'cert_P'), ('hard_noisy', 'paper15_s36', 1, 'bgreedy_P'),
         ('hard_noisy', 'soumalas11_s37', 0, 'cert_P')]


def one(case):
    cond, name, path, arm = case
    inp = bank.load(ROOT/cond/"inputs"/f"{name}.pkl")
    r = run.run(inp, arm, path, settings_for(cond))
    return case, r["outcome"], r["cost_cny"], max((b["cost"] for b in r["blocks"]), default=0.)


if __name__ == "__main__":
    old = {}
    for cond in ('hard_short005', 'hard_noisy'):
        for r in load(cond):
            old[(cond, r['input'], r['path'], r['arm'])] = (r['outcome'], r['cost_cny'])
    with ProcessPoolExecutor(9) as pool:
        for case, outcome, cost, mx in pool.map(one, CASES):
            o = old[case]
            print(f"{case}: before {o[0]} {o[1]:.1f} CNY -> after {outcome} {cost:.1f} CNY (largest block {mx:.1f})")
