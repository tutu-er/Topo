"""Pure array/unit-contract tests; no optimization model or AC is instantiated."""
import hashlib,pickle,unittest
import numpy as np
import study

def cfg():
    return dict(L=2.6,b=.4,kappa=.35,base_kw=200.,labels=[101,102,103],
        seeds=dict(natural=884001,measurement=884002,acceptance_and_independent_code=884003))

class StudyTests(unittest.TestCase):
    def test_discovery_json_identity_handles_numpy_support_indices(self):
        import json
        value=[{'candidate_supports':[(np.int64(0),np.int64(1))],'weight':np.float64(.25)}]
        self.assertEqual(json.loads(study.encode(value)),[{'candidate_supports':[[0,1]],'weight':.25}])

    def test_matched_history_all_pairs_zero_weak_strong_and_zero_root(self):
        conf=cfg();reg=dict(pq_relative=.005,voltage_relative=.001);p,q,_=study.natural_day(0,conf)
        expected=None
        for b in (.4,0.):
            for c in (0.,.02,.6):
                for pair in ([0,1],[0,2],[1,2]):
                    R,X,t=study.truth(b,c,pair,conf)
                    self.assertTrue(np.allclose((R+.35*X)@np.ones(3),2.6))
                    self.assertLessEqual(max(np.max(R),np.max(X)),2.)
                    scene=study.observed_scene(p,q,R,X,0,reg,conf,2.6*p[:,[0]]*np.ones((1,3)))
                    digest=hashlib.sha256(pickle.dumps(scene,protocol=4)).hexdigest()
                    if expected is None:expected=digest
                    self.assertEqual(digest,expected)

    def test_fixed_prices_same_budget_and_task_all_policies(self):
        conf=cfg()
        for name in ('zero_sham','pair_random','single_random','rotation_random','pair_all_on_conditional'):
            draw=study.control(name,conf);delta=draw['delta_p_kw']
            self.assertLessEqual(draw['worst_payment'],1.6+1e-12)
            np.testing.assert_allclose(delta.sum(axis=1),0,atol=1e-14)
            np.testing.assert_allclose(np.maximum(-delta,0).sum(axis=1)*.25,.2*draw['accepted'],atol=1e-14)
            device=.2+delta
            self.assertTrue(((device>=0)&(device<=.4+1e-12)).all())
            np.testing.assert_allclose(np.cumsum(device,axis=1)[:,-1]*.25,.4)
            if name=='pair_all_on_conditional':
                self.assertTrue(draw['conditional_state_not_randomized_trial'])
                self.assertEqual(draw['pattern_probability_under_local_pair_price1'],1/256)

    def test_hidden_block_can_be_unobservable_for_balanced_endpoints(self):
        conf=cfg();R,X,t=study.truth(.4,.6,[0,2],conf);K=R+.35*X
        alternative=.2*np.eye(3)+.8*np.ones((3,3))
        probe=np.array([1.,0.,-1.])
        np.testing.assert_allclose(K@np.ones(3),alternative@np.ones(3),atol=1e-14)
        np.testing.assert_allclose(K@probe,alternative@probe,atol=1e-14)
        self.assertIn([0,2],t['effective_supports'])

if __name__=='__main__':unittest.main()
