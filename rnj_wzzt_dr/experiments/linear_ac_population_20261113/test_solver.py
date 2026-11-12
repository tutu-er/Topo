import unittest
import numpy as np
from .solver import Model,single,solve,finite_attribute
from .study import forecast


class Tests(unittest.TestCase):
    def test_missing_native_bound_is_unavailable_not_a_plan_exception(self):
        class EarlyStop:
            @property
            def ObjBound(self):raise AttributeError('Unable to retrieve attribute ObjBound')
            MIPGap=float('inf')
            finite=3.
        self.assertIsNone(finite_attribute(EarlyStop(),'ObjBound'))
        self.assertIsNone(finite_attribute(EarlyStop(),'MIPGap'))
        self.assertEqual(finite_attribute(EarlyStop(),'finite'),3.)

    def test_zero_background_hard_case_known_global_cost(self):
        H=np.array([np.column_stack((np.eye(2),np.zeros((2,2)))),np.zeros((2,4))])
        model=Model(H,0,[1],np.ones(2),.5).prepare(np.zeros((96,4)))
        result=solve(model,2.)
        self.assertTrue(result['qualified']);self.assertTrue(result['global_optimality_verified'])
        self.assertAlmostEqual(result['cost'],2/4/48,places=9)
        self.assertAlmostEqual(result['lower_bound'],2/4/48,places=9)

    def test_nonzero_forecast_exact_dual_and_full_H(self):
        rng=np.random.default_rng(51);H=rng.normal(size=(3,3,6));prediction=rng.normal(size=(96,6))
        model=Model(H,0,[1,2],np.array([.01,.02,.03]),1e-4).prepare(prediction)
        for j in range(2):
            result=single(model,j,.7)
            self.assertTrue(result['feasible']);self.assertLess(result['gap'],1e-5)
            self.assertGreaterEqual(model.direct_gains(result['x'])[j],.7-1e-8)

    def test_joint_orthogonal_constraints_and_valid_lower_bound(self):
        H=np.zeros((3,2,4));H[1,0,0]=1.;H[2,1,1]=1.
        model=Model(H,0,[1,2],np.ones(2),.5).prepare(np.zeros((96,4)))
        result=solve(model,.7,seconds=.4)
        self.assertTrue(result['qualified']);self.assertGreaterEqual(min(result['gains']),.7-1e-8)
        exact=2*.7/4/48
        self.assertGreaterEqual(result['cost'],exact-1e-8)
        self.assertLessEqual(result['lower_bound'],exact+1e-8)

    def test_uncontrollable_comparison_preserved(self):
        model=Model(np.zeros((2,2,4)),0,[1],np.ones(2),1.).prepare(np.ones((96,4)))
        result=solve(model,1.)
        self.assertFalse(result['qualified']);self.assertEqual(result['status'],'uncontrollable_competitor')

    def test_prediction_noise_separate_and_relative(self):
        p=np.ones((96,2));q=.35*p
        pred,z=forecast(p,q,np.random.default_rng(7),.05)
        np.testing.assert_allclose(pred,np.column_stack((p,q))*(1+.05*z),rtol=0,atol=0)
        self.assertGreater(float(np.std((pred[:,:2]-p)/p)),.005)
        self.assertEqual(pred.shape,(96,4))


if __name__=='__main__':unittest.main()
