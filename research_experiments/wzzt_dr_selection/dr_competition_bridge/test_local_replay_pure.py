"""Pure-array checks. Never construct PoolLP, Gurobi, or a real data history."""
import unittest
import numpy as np

from physical_bridge import scene
import local_replay as r


class LocalReplayPureTests(unittest.TestCase):
    def point(self):
        return dict(supports=[[0],[1],[0,1]],r=[.1,.1,.02],x=[0.,0.,0.])

    def test_strict_positive_effective_support_retains_weak_and_excludes_zero(self):
        point=dict(supports=[[0],[1],[0,1]],r=[.1,0.,1e-30],x=[0.,0.,0.])
        self.assertEqual(r.effective_family(point),{(0,),(0,1)})
        self.assertEqual(r.structural_loss(point,{(0,),(1,)}),2)
        with self.assertRaises(ValueError):
            r.effective_family(dict(supports=[[0]],r=[-1e-20],x=[0.]))

    def test_future_mask_price_independent_energy_returns(self):
        forecast=dict(P_pu=np.full((96,2),.02),Q_pu=np.full((96,2),.007),labels=[101,102],base_kw=200.)
        inputs=r.future_inputs(forecast,[0,1])
        self.assertEqual(set(inputs),{0,1,2,3})
        for key,(p,q) in inputs.items():
            for j in range(2):
                delta=.001 if key&(1<<j) else 0.
                self.assertAlmostEqual(p[76,j],.02-delta)
                self.assertAlmostEqual(p[80,j],.02+delta)
            np.testing.assert_allclose((p-forecast['P_pu']).sum(axis=0),0,atol=1e-15)
            np.testing.assert_allclose((q-forecast['Q_pu']).sum(axis=0),0,atol=1e-15)

    def test_validation_scorer_equals_direct_all_rows(self):
        point=self.point();rh,xh=r.point_matrices(point,2)
        p=np.full((96,2),.02);q=np.full((96,2),.007)
        y=p@rh.T+q@xh.T
        s=scene(p,q,np.sqrt(1-y),np.ones(96),[101,102],'pure',0)
        old=r.history_arrays([s,s,s])
        future=r.future_inputs(dict(P_pu=p,Q_pu=q,labels=[101,102],base_kw=200.),[0,1])
        scorer=r.ValidationScorer(old,future,point)
        wrong=dict(supports=[[0],[1]],r=[.14,.14],x=[0.,0.])
        for key in ((a,b) for a in range(4) for b in range(4)):
            candidate_r,candidate_x=r.point_matrices(wrong,2)
            yp=np.concatenate([old[0],future[key[0]][0],future[key[1]][0]])
            yq=np.concatenate([old[1],future[key[0]][1],future[key[1]][1]])
            target=np.concatenate([old[2],future[key[0]][0]@rh.T+future[key[0]][1]@xh.T,
                                   future[key[1]][0]@rh.T+future[key[1]][1]@xh.T])
            expected=float(abs(yp@candidate_r.T+yq@candidate_x.T-target).mean())
            self.assertAlmostEqual(scorer.mae(wrong,key),expected,places=16)
            self.assertLess(scorer.mae(point,key),1e-15)

    def test_start_dedup_and_crossing_acceptance_rule(self):
        a=[[0],[1],[2],[0,1]]; b=[[0],[1],[2],[1,2]]; c=[[0],[1],[2],[0,2]]
        old=dict(rnj_bank=[dict(config='raw_RX75_0.16',native_family=a)],
                 base_points=[dict(supports=b,validation_mae=1),dict(supports=c,validation_mae=2)])
        starts=r._initial_families(old,[0,1])
        self.assertEqual(len(starts),3)
        trajectory=dict(decisions=[dict(gain=.01,current=1.,full_sweep=True,candidates=[
            dict(candidate=[0,1],removed=[],objective=.995),
            dict(candidate=[1,2],removed=[[0,1]],objective=.99)])])
        accepted=r._crossing_accepts(trajectory)
        self.assertEqual(len(accepted),1)
        self.assertTrue(accepted[0]['crosses'])
        trajectory['decisions'][0]['gain']=.0001
        self.assertEqual(r._crossing_accepts(trajectory),[])

    def test_proposal_view_preserves_full_family_fitting(self):
        full=[[0],[1],[2],[0,1],[0,2],[1,2]]
        hypotheses=[dict(supports=[[0],[1],[2],[0,1]]),
                    dict(supports=[[0],[1],[2],[1,2]])]
        restricted=r.proposal_pool(full,hypotheses,'historical_representative_union')
        self.assertEqual(set(restricted),{(0,),(1,),(2,),(0,1),(1,2)})
        self.assertNotIn((0,2),restricted)
        self.assertIn((0,2),r.proposal_pool(full,hypotheses,'full_old_pool'))
        class FullFitStub:
            n=3
            def __init__(self):self.last=None
            def fit(self,family):self.last=family;return {'supports':family,'forwarded':True}
        timed=FullFitStub()
        view=r._ProposalPoolView(timed,restricted)
        self.assertEqual(view.n,3)
        native_family=[[0],[1],[2],[0,2]]
        # An old native block outside new proposals is still jointly fitted.
        fitted=view.fit(native_family)
        self.assertIs(timed.last,native_family)
        self.assertTrue(fitted['forwarded'])
        self.assertIn([0,2],fitted['supports'])
        with self.assertRaises(ValueError):r.proposal_pool(full,hypotheses,'unknown')
        with self.assertRaises(ValueError):r.proposal_pool([[0]],hypotheses,'historical_representative_union')


if __name__=='__main__':unittest.main()
