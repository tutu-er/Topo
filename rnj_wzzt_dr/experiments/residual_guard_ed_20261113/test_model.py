import unittest
import numpy as np
from ..linear_ac_population_20261113.solver import Model
from .model import Evidence,along
from .solver import solve


class Tests(unittest.TestCase):
    def model(self):
        rng=np.random.default_rng(20261113);H=rng.normal(size=(3,2,4));F=rng.normal(size=(96,4))
        return Model(H,0,[1,2],np.array([.7,1.3]),.2).prepare(F)

    def test_quadratics_match_independent_expected_SSE(self):
        m=self.model();means=m.H*np.array([.98,1.03])[None,:,None];e=Evidence(m,means)
        x=np.random.default_rng(9).normal(size=m.shape)*.03;U=m.forecast+m.shift(x);answer=[]
        for j in m.competitors:
            def nominal(Z):return np.sum((Z@(m.H[0]-m.H[j]).T)**2)/(2*m.sigma2)
            answer.append(nominal(U)-nominal(m.forecast))
        for i,j,K in e.pairs:
            def odds(Z):
                Y=Z@K.T
                return (np.sum((Y-Z@m.H[j].T)**2)-np.sum((Y-Z@m.H[i].T)**2))/(2*m.sigma2)
            answer.append(odds(U)-odds(m.forecast))
        np.testing.assert_allclose(e.values(x),answer,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(e.direct(x),answer,rtol=1e-9,atol=1e-10)

    def test_interval_handles_negative_quadratic_and_zero_guard(self):
        class E:
            shape=(1,1);Q=np.array([[[-1.]],[[1.]]]);a=np.array([[[1.]],[[.5]]])
            def values(self,x):return np.array([-float(x[0,0])**2+2*x[0,0],x[0,0]**2+x[0,0]])
        e=E();x=along(e,np.ones((1,1)),np.array([.5,0.]),2.)
        self.assertIsNotNone(x);self.assertGreaterEqual(e.values(x)[0],.5-1e-8)
        self.assertIsNone(along(e,-np.ones((1,1)),np.array([.5,0.]),2.))

    def test_capacity_infeasibility_remains_valid_with_guards(self):
        H=np.array([np.c_[np.eye(2),np.zeros((2,2))],np.zeros((2,4))]);m=Model(H,0,[1],np.ones(2),.5).prepare(np.zeros((96,4)))
        r=solve(m,2.,.1,means=H)
        self.assertFalse(r['qualified']);self.assertTrue(r['infeasibility_proved'])

    def test_native_return_satisfies_original_and_corrected_evidence(self):
        A=np.c_[np.eye(2),np.zeros((2,2))];H=np.array([A,.8*A])
        m=Model(H,0,[1],np.ones(2),.5).prepare(np.ones((96,4)))
        means=H.copy();means[0]*=np.array([.5,1.1])[:,None]
        r=solve(m,1.,.3,means=means,seconds=.3)
        self.assertTrue(r['qualified']);e=Evidence(m,means)
        self.assertIsNotNone(r['native'])
        self.assertGreaterEqual(min(e.direct(r['x'])-e.thresholds(1.)),-1e-8)
        self.assertLessEqual(r['lower_bound'],r['cost']+1e-12)


if __name__=='__main__':unittest.main()
