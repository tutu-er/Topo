import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import solver


class ToyCase:
    def __init__(self,shape=(1,1)):
        self.H,self.R=shape;self.shape=shape;self.lower=np.zeros(shape);self.upper=np.ones(shape)
    def array(self,u):
        u=np.asarray(u,float)
        if u.shape!=self.shape or not np.isfinite(u).all():raise ValueError('array')
        return u
    def audit(self,u,tolerance=1e-7):
        u=self.array(u);return dict(passed=bool(np.all(u>=self.lower-tolerance) and np.all(u<=self.upper+tolerance)))
    def fee(self,u):return float(abs(self.array(u)).sum())
    def attach(self,model):
        u=model.addMVar(self.shape,lb=self.lower,ub=self.upper,name='unchanged_action');return u,u.sum()


def geometry(mode='protected'):
    # J=(1+u)^2, e=(1-u)^2, J0=e0=1. Protected optimum z=1 at u=1.
    return dict(mode=mode,J_q=np.ones((1,1,1,1)),J_a=np.full((1,1,1),2.),J_c=np.ones(1),
        e_q=np.ones((1,1,1)),e_a=np.full((1,1),-2.),e_c=1.,j_scales=np.ones(1))


class AlgebraTests(unittest.TestCase):
    def test_full_linear_coefficient_and_common_error_progress(self):
        g=solver.Geometry(ToyCase(),geometry());z,a=g.best_value_at(ToyCase(),np.array([[.5]]))
        self.assertAlmostEqual(z,.75,places=12);self.assertAlmostEqual(a['J'][0],2.25);self.assertAlmostEqual(a['error'],.25)
        self.assertTrue(a['passed']);self.assertFalse(g.check(ToyCase(),np.array([[.5]]),.76)['passed'])
    def test_all_competitors_not_only_best_pair(self):
        d=geometry('distance_only');d['J_q']=np.concatenate([d['J_q'],d['J_q']*.1]);d['J_a']=np.concatenate([d['J_a'],d['J_a']*.1]);d['J_c']=np.array([1.,10.]);d['j_scales']=np.array([1.,10.])
        g=solver.Geometry(ToyCase(),d);z,a=g.best_value_at(ToyCase(),np.ones((1,1)))
        self.assertAlmostEqual(z,.03,places=12);self.assertTrue(a['passed'])
    def test_zero_error_remains_nonincrease_without_relative_ratio(self):
        d=geometry();d['e_c']=0.;d['e_a'][:]=0.;g=solver.Geometry(ToyCase(),d)
        self.assertTrue(g.best_value_at(ToyCase(),np.zeros((1,1)))[1]['passed'])
        self.assertFalse(g.best_value_at(ToyCase(),np.ones((1,1)))[1]['passed'])
    def test_negative_progress_rejected_not_promoted_to_positive(self):
        d=geometry();d['e_a'][:]=2.;g=solver.Geometry(ToyCase(),d)
        z,a=g.best_value_at(ToyCase(),np.array([[.1]]));self.assertEqual(z,0.);self.assertFalse(a['passed'])
    def test_hour_resource_shape_preserved_and_bad_shape_rejected(self):
        case=ToyCase((2,2));Q=np.array([[[2.,1.],[1.,3.]],[[1.,0.],[0.,4.]]]);a=np.array([[2.,3.],[4.,5.]]);u=np.array([[.1,.2],[.3,.4]])
        value,_=solver.polynomial_change(Q,a,u);expected=sum(u[h]@Q[h]@u[h]+a[h]@u[h] for h in range(2));self.assertAlmostEqual(value,expected)
        with self.assertRaises(ValueError):solver.Geometry(case,geometry())
    def test_maximization_bound_conflict_withdraws_native_upper(self):
        r=solver.certify_bounds(2.,10.,2.,[{'upper_bound':1.,'native_runtime_seconds':1.}])
        self.assertTrue(r['native_upper_bound_withdrawn']);self.assertEqual(r['upper_bound'],10.);self.assertEqual(r['lower_bound'],2.)
        self.assertTrue(r['positive_existence_demonstrated'])


class TinySolveTests(unittest.TestCase):
    def test_two_modes_keep_zero_and_find_expected_positive_point(self):
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            for mode,target in [('distance_only',3.),('protected',1.)]:
                with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                    case=ToyCase();r,u,z=solver.plan(case,geometry(mode),{'bad_domain':np.array([[2.]]),'seed':np.array([[.25]])},2.,env,Path(folder))
                    self.assertAlmostEqual(z,target,places=5);self.assertTrue(r['positive_existence_demonstrated']);self.assertTrue(r['final']['passed'])
                    self.assertFalse(r['cost_is_objective']);self.assertTrue((Path(folder)/'diagnostics.json').is_file())
                    self.assertTrue(any(p.get('name')=='bad_domain' and not p['feasible'] for p in r['initial']['candidates']))
    def test_no_positive_is_not_called_infeasible(self):
        import gurobipy as gp
        d=geometry();d['e_a'][:]=2.
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start();r,u,z=solver.plan(ToyCase(),d,{},2.,env)
        self.assertEqual(z,0.);self.assertEqual(r['status'],'no_positive_witness_found');self.assertFalse(r['infeasibility_proved']);self.assertTrue(r['feasible'])


if __name__=='__main__':unittest.main()
