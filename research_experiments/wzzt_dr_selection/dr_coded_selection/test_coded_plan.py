"""Pure template/selection/input-design checks; no real solver or future data."""
import unittest
import numpy as np
import pandas as pd
from coded_model import TEMPLATES,apply_response,contract_spec,directions
from coded_planner import select_historical_pair,ordinary_tables,_choice


class CodedPlanTests(unittest.TestCase):
    def test_equal_power_energy_and_actual_template_signs(self):
        p=np.full((96,3),.02);q=np.full((96,3),.007)
        for kind in TEMPLATES:
            pp,qq,receipt=apply_response(p,q,np.array([1,1,0]),np.array([1,1,0]),
                template_kind=kind,active_positions=[0,1])
            dp=(pp-p)*200;dq=(qq-q)*200
            np.testing.assert_allclose(dq,.35*dp,atol=1e-14)
            np.testing.assert_allclose(np.maximum(-dp,0).sum(axis=0)*.25,[.2,.2,0],atol=1e-14)
            np.testing.assert_allclose(np.maximum(dp,0).sum(axis=0)*.25,[.2,.2,0],atol=1e-14)
            np.testing.assert_allclose(dp.sum(axis=0),0,atol=1e-14)
            self.assertAlmostEqual(abs(dp).max(),.2)
            self.assertTrue(receipt['capacity_pass'])
            if kind=='same':np.testing.assert_allclose(dp[:,0],dp[:,1])
            else:np.testing.assert_allclose(dp[:,0],-dp[:,1])
            if kind=='alternating_opposed':
                np.testing.assert_allclose(dp[76:84,0],[-.2,.2]*4)
            else:np.testing.assert_allclose(dp[76:84,0],[-.2]*4+[.2]*4)
            contract_spec(kind)
            d=directions(kind,[0,1],3)
            np.testing.assert_array_equal(d[0],d[1])

    def test_partial_response_does_not_fake_ancestor_balance(self):
        p=np.full((96,3),.02);q=np.full((96,3),.007)
        pp,_,_=apply_response(p,q,np.array([1,1,0]),np.array([1,0,0]),
            template_kind='opposed',active_positions=[0,1])
        self.assertGreater(abs((pp-p).sum(axis=1)).max(),0)
        with self.assertRaises(ValueError):
            apply_response(p,q,np.zeros(3),np.array([1,0,0]),template_kind='same',active_positions=[0,1])

    def test_history_pair_uses_only_eligible_crossings(self):
        leaves=[[0],[1],[2],[3]]
        a=dict(supports=leaves+[[0,1],[2,3]],r=[.1]*6,x=[0.]*6,validation_mae=1.)
        b=dict(supports=leaves+[[1,2],[0,3]],r=[.1]*6,x=[0.]*6,validation_mae=1.01)
        old=dict(labels=[101,102,103,104],base_points=[a,b],calibration=dict(a=1.,b=1.,median_root_depth=.5))
        first=select_historical_pair(old,53);second=select_historical_pair(old,53)
        self.assertEqual(first,second)
        self.assertEqual(first[0],[0,1]);self.assertTrue(first[2]['both_candidate_top_groups'])
        c,d=set(first[2]['C']),set(first[2]['D'])
        self.assertIn(old['labels'].index(first[1][0]),c-d)
        self.assertIn(old['labels'].index(first[1][1]),d-c)

    def test_ordinary_input_has_all_same_three_template_permissions(self):
        rng=np.random.default_rng(14)
        h=dict(labels=[101,102,103],base_kw=200.,training=[],validation=[])
        for day in range(6):
            s=dict(P_terminal=pd.DataFrame(.02+.004*rng.random((96,3))),
                   Q_terminal=pd.DataFrame(.007+.001*rng.random((96,3))))
            h['training' if day<3 else 'validation'].append(s)
        for law in ('local','common'):
            rows=ordinary_tables(h,[101,102],law)
            self.assertEqual(len(rows),150)
            self.assertEqual({r['template'] for r in rows},set(TEMPLATES))
            zeros=[r['value'] for r in rows if np.asarray(r['small_menu']).sum()==0]
            self.assertLess(max(zeros)-min(zeros),1e-12)
            selected=_choice(rows,141,True)
            self.assertGreaterEqual(selected['value'],max(zeros)-1e-12)


if __name__=='__main__':unittest.main()
