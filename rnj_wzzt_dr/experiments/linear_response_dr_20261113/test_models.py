"""Checks of domain distinctions, valid upper bounds and original fee semantics."""
import unittest
import numpy as np
from ...dr.pipeline import load_resources
from ..expected_logodds_admm_20261113.evidence import build
from .models import build_resource,point,power_box_upper


class Tests(unittest.TestCase):
    def test_continuous_models_and_net_fees(self):
        import gurobipy as gp
        original,problem,_=load_resources('paper15',labels=tuple(range(101,116)))
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            for mode in ('linear_response','storage_relaxation'):
                with gp.Model(env=env) as model:
                    model.Params.OutputFlag=0;model.Params.Threads=1
                    resource=build_resource(model,problem,original,3,mode)
                    model.setObjective(resource['fee']-.01*resource['x'].sum())
                    model.optimize();self.assertEqual(model.Status,2);self.assertEqual(model.NumIntVars,0)
                    p=point(resource);u=np.zeros_like(original.scale);u[3]=p['delivery']
                    self.assertAlmostEqual(p['model_fee'],original.fees(u),places=6)
                    if mode=='linear_response':self.assertIsNone(resource['energy'])
                    else:self.assertAlmostEqual(float(resource['energy'].X[-1]),0.,places=7)

    def test_power_box_is_an_upper_bound(self):
        original,problem,base=load_resources('paper15',labels=tuple(range(101,116)))
        rng=np.random.default_rng(23);H=rng.normal(0,.01,(3,15,30));forecast=rng.normal(.01,.002,(96,30))
        evidence=build(H,0,[1,2],forecast,1e-6,original,base,.7)
        upper=power_box_upper(dict(original=original,evidence=evidence))
        for _ in range(20):
            u=rng.uniform(original.delta_lower_kw,original.delta_upper_kw)
            self.assertTrue(np.all(evidence.gains(u)<=upper+1e-10))


if __name__=='__main__':unittest.main()
