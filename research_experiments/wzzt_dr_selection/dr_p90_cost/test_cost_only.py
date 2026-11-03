import unittest
import numpy as np
from scipy.stats import norm
from cost_only import solve

class Device:
    R=1;kind=np.array([0]);scale=np.ones((1,96))
class Toy:
    devices=Device()
    def __init__(self,unattainable=False):self.evaluations=0;self.unattainable=unattainable
    def evaluate(self,x):
        self.evaluations+=1;x=np.asarray(x).ravel();z=3*x.mean()
        return dict(mean_u=np.repeat(x,4).reshape(1,96),cost=float(np.mean(x*x)),cost_gradient=2*x/24,
            probability=.2 if self.unattainable else float(norm.cdf(z)),invalid=[],
            gradient=np.zeros(24) if self.unattainable else np.full(24,3*norm.pdf(z)/24))
class Tests(unittest.TestCase):
    def test_cost_at_90(self):
        q=solve(Toy(),[('warm',np.full((1,24),.8))],maxiter=30,seconds=10)
        self.assertTrue(q['qualified']);self.assertAlmostEqual(q['best']['cost'],(norm.ppf(.9)/3)**2,places=6)
        self.assertGreaterEqual(q['best']['probability'],.9-1e-7)
    def test_no_feasible_start_is_not_infeasibility_proof(self):
        q=solve(Toy(True),[('zero',np.zeros((1,24))),('warm',np.ones((1,24)))])
        self.assertFalse(q['qualified']);self.assertFalse(q['infeasibility_proven']);self.assertEqual(q['status'],'no_feasible_start')
    def test_target_is_explicit(self):
        a=solve(Toy(),[('warm',np.ones((1,24)))],target=.9)
        b=solve(Toy(),[('warm',np.ones((1,24)))],target=.95)
        self.assertLess(a['best']['cost'],b['best']['cost'])
if __name__=='__main__':unittest.main()
