import unittest,numpy as np
from scipy.integrate import quad
from scipy.stats import norm,ncx2
from quadratic import *
class QuadraticTests(unittest.TestCase):
    def test_normal(self):
        q=quadratic_sf(np.zeros(2),np.array([1.,2.]),.3,1.2)
        self.assertAlmostEqual(q['p'],norm.sf(.9/np.sqrt(5)),places=12)
    def test_chi(self):
        q=quadratic_sf(np.array([1.]),np.array([2.]),.5,3.)
        self.assertAlmostEqual(q['p'],ncx2.sf(3.5,1,1),places=12)
    def test_negative_chi(self):
        q=quadratic_sf(np.array([-1.]),np.array([0.]),0.,-.8)
        self.assertAlmostEqual(q['p'],ncx2.cdf(.8,1,0),places=12)
    def test_indefinite(self):
        q=quadratic_sf(np.array([1.,-1.]),np.zeros(2),0.,0.)
        self.assertAlmostEqual(q['p'],.5,places=12)
    def test_quadratic_plus_normal(self):
        q=quadratic_sf(np.array([.5,0.]),np.array([.3,1.]),.2,1.1)
        exact=quad(lambda z:norm.sf(1.1-.2-.5*z*z-.3*z)*norm.pdf(z),-12,12,epsabs=1e-11)[0]
        self.assertLess(abs(q['p']-exact),2e-7)
        self.assertLess(q['error'],1e-6)
    def test_projection_variance(self):
        q=quadratic_projection(np.array([.3,-.7]),np.array([[2,.4],[.4,1.]]),np.array([1.,-1.03]))
        self.assertAlmostEqual(q['variance_full'],q['variance_factored'],places=10)
if __name__=='__main__':unittest.main()
