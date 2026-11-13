import unittest
import numpy as np
from ..linear_ac_population_20261113.solver import Model
from .solver import solve,box_upper


class Tests(unittest.TestCase):
    def model(self,zero=True):
        H=np.array([np.column_stack((np.eye(2),np.zeros((2,2)))),np.zeros((2,4))])
        return Model(H,0,[1],np.ones(2),.5).prepare(np.zeros((96,4)) if zero else np.ones((96,4)))

    def test_bound_can_prove_infeasibility(self):
        m=self.model();r=solve(m,2.,.1)
        self.assertFalse(r['qualified']);self.assertTrue(r['infeasibility_proved'])
        self.assertAlmostEqual(float(box_upper(m,.1)[0]),1.92,places=10)

    def test_bounded_hard_case_matches_independent_cost(self):
        m=self.model();r=solve(m,.7,.1,seconds=.5)
        self.assertTrue(r['qualified']);self.assertLessEqual(np.max(abs(r['x'])),.1+1e-8)
        self.assertGreaterEqual(r['gains'][0],.7-1e-8)
        self.assertAlmostEqual(r['cost'],.7/4/48,places=8)
        self.assertTrue(r['global_optimality_verified'])

    def test_box_upper_dominates_sampled_controls(self):
        m=self.model(False);rng=np.random.default_rng(19);upper=box_upper(m,.3)
        for _ in range(20):self.assertTrue(np.all(m.gains(rng.uniform(-.3,.3,m.shape))<=upper+1e-10))

    def test_unbounded_optimum_within_capacity_remains_global(self):
        m=self.model(False);r=solve(m,.7,.3)
        self.assertTrue(r['qualified']);self.assertTrue(r['global_optimality_verified'])
        self.assertLessEqual(np.max(abs(r['x'])),.3+1e-10)


if __name__=='__main__':unittest.main()
