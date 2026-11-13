import unittest
import numpy as np
from ..linear_ac_population_20261113.solver import Model
from .polish import improve


class Tests(unittest.TestCase):
    def test_original_feasibility_and_independent_known_optimum(self):
        H=np.array([np.c_[np.eye(2),np.zeros((2,2))],np.zeros((2,4))])
        m=Model(H,0,[1],np.ones(2),.5).prepare(np.zeros((96,4)))
        initial=np.full(m.shape,.08);x,r=improve(m,initial,.7,.1)
        self.assertGreaterEqual(min(m.direct_gains(x)),.7-1e-8)
        self.assertLessEqual(r['cost'],r['initial_cost'])
        self.assertAlmostEqual(r['cost'],.7/(4*48),places=7)
        self.assertFalse(r['lower_bound_from_QP_used'])


if __name__=='__main__':unittest.main()
