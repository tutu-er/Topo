"""A declared feasible arithmetic problem exercises the previously unused cost stage.

This is a solver test, not an extra physical network or topology-effect case.
"""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import refine as r


class CostBranch(unittest.TestCase):
    def test_known_feasible_min_cost_stage(self):
        lib=r.REPO/'research_experiments/wzzt_dr_selection/dr_heterogeneous_resources/library.json'
        cfg=r.read(lib);allocation=r.read(lib.with_name('case_allocations.json'))['cases']
        c=next(v for v in allocation if v['case']=='paper15')
        d=r.Devices({'nodes':[c['nodes'][0]]},cfg,r.profiles);A=d.cap[1]
        packet=dict(mu_C=2*np.ones(96),mu_F=.1*np.ones(96),L_C=.005*A*np.ones((96,1)),L_F=np.zeros((96,1)),
            effect_C=np.array([[0.,-1.,0.,0.]]),effect_F=np.zeros((1,4)),threshold=1.897+.05*A)
        prob=r.Probability(packet,d,4,2026110807,2026110808)
        old=np.zeros((4,24));old[1]=-.9;q=prob.evaluate(old)
        self.assertGreater(q['probability'],.95)
        limits={name:dict(maxiter=1,seconds=5) for name in ['mean','tail','prob_old','prob_tail']}
        limits['cost']=dict(maxiter=8,seconds=20)
        out=r.solve(prob,old,limits)
        self.assertTrue(out['qualified'])
        self.assertIn('cost',[v['stage'] for v in out['traces']])
        self.assertGreaterEqual(out['best']['probability'],.95-1e-7)
        self.assertLess(out['best']['cost'],q['cost'])
        self.assertTrue(d.audit(out['best']['mean_u'])['passed'])


if __name__=='__main__':unittest.main()
