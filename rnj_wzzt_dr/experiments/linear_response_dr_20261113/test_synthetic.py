import unittest
import numpy as np
from .synthetic import make


class Tests(unittest.TestCase):
    def test_balanced_natural_input_has_no_discrimination(self):
        H,profile,sigma,effect,k=make()
        np.testing.assert_allclose((H[0]-H[1])@np.ones(2),0,atol=1e-16)
        self.assertTrue(np.all(np.linalg.eigvalsh(H)>0))
        self.assertAlmostEqual(float(profile.sum()),0.,places=12)

    def test_gain_and_analytic_fee_bound_from_full_predictions(self):
        H,profile,sigma,effect,k=make()
        for gamma in (np.log(2),2.,4.,8.):
            amplitude=np.sqrt(gamma/k)
            self.assertLess(amplitude,.1)
            power=np.ones((96,2))+amplitude*np.column_stack((profile,-profile))
            distance=power@(H[0]-H[1]).T
            self.assertAlmostEqual(float(np.sum(distance**2)/(2*sigma**2)),gamma,places=11)
            self.assertAlmostEqual(amplitude**2,gamma/k,places=14)


if __name__=='__main__':unittest.main()
