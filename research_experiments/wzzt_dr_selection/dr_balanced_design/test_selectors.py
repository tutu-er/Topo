"""Small score-only checks; no project imports or solver."""
import unittest
from copy import deepcopy
import numpy as np
import selection_rules as s

def point(supports):return dict(supports=supports,r=[1.]*len(supports),x=[0.]*len(supports))
def entry(raw,val,contract=None,cval=None):
    return dict(raw=raw,raw_val=val,contract=raw if contract is None else contract,contract_val=val if cval is None else cval)

class SelectorTests(unittest.TestCase):
    def test_original_band_can_prefer_higher_loss_simple_family(self):
        pts=[entry(point([[0],[1],[2],[0,1]]),1.),entry(point([[0],[1],[2]]),1.02)]
        self.assertEqual(s.choose(pts,'original_3pct_raw')['source_index'],1)
        self.assertEqual(s.choose(pts,'strict_raw')['source_index'],0)
    def test_contract_is_scored_after_actual_weight_deletion(self):
        p=dict(supports=[[0],[1],[2],[0,1,2]],r=[1.,1.,1.,.01],x=[0.,0.,0.,0.])
        q=s.contract(p,dict(a=1.,b=1.,median_root_depth=1.))
        P=np.eye(3);Q=np.zeros_like(P);R,X=s.matrices(p);Y=P@R.T
        self.assertEqual(s.score(p,(P,Q,Y)),0.)
        self.assertGreater(s.score(q,(P,Q,Y)),0.)
        self.assertEqual(len(q['supports']),3)
        self.assertEqual(len(p['supports']),4)
    def test_combined_selects_raw_when_contract_worsens(self):
        raw=point([[0],[1],[2],[0,1,2]]);contract=point([[0],[1],[2]])
        selected=s.choose([entry(raw,.001,contract,.1)],'strict_raw_or_contract')
        self.assertEqual(selected['variant'],'raw')
    def test_combined_can_select_contract_when_recomputed_score_better(self):
        raw=point([[0],[1],[2],[0,1,2]]);contract=point([[0],[1],[2]])
        selected=s.choose([entry(raw,.1,contract,.001)],'strict_raw_or_contract')
        self.assertEqual(selected['variant'],'contract')
    def test_strict_numerical_tie_support_then_original_index(self):
        pts=[entry(point([[0],[1],[2],[0,1]]),1.),entry(point([[0],[1],[2]]),1.+5e-13),entry(point([[0],[1],[2]]),1.)]
        self.assertEqual(s.choose(pts,'strict_raw')['source_index'],1)
        pts[1]['raw_val']=1.+2e-12
        self.assertEqual(s.choose(pts,'strict_raw')['source_index'],2)
    def test_zero_coefficients_retained_in_candidate_and_effective_separate(self):
        p=dict(supports=[[0],[1],[2],[0,1]],r=[1.,0.,1.,0.],x=[0.,0.,0.,0.])
        q=s.choose([entry(p,1.)],'strict_raw')['point']
        self.assertEqual(q,p)
        self.assertEqual(s.effective(q),{(0,),(2,)})
    def test_no_truth_field_needed_and_input_not_mutated(self):
        pts=[entry(point([[0],[1],[2]]),1.)];old=deepcopy(pts)
        for rule in s.RULES:s.choose(pts,rule)
        self.assertEqual(pts,old)

if __name__=='__main__':unittest.main(verbosity=2)
