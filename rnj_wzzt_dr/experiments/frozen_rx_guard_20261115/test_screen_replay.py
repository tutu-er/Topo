import unittest
import numpy as np

from .screen_replay import budget_value,screen_decision


class ScreenTests(unittest.TestCase):
    def test_budget_and_probability_both_required(self):
        yes=[{'passed':True}];no=[{'passed':False}]
        self.assertTrue(screen_decision(True,yes)['retain_original'])
        self.assertFalse(screen_decision(False,yes)['retain_original'])
        self.assertFalse(screen_decision(True,no)['retain_original'])
        with self.assertRaises(ValueError):screen_decision(True,[])

    def test_arithmetic_tolerance_does_not_relax_budget(self):
        self.assertTrue(budget_value(np.array([1.]),np.eye(1),1.)['passed'])
        self.assertFalse(budget_value(np.array([1.+1e-7]),np.eye(1),1.)['passed'])
        x=np.array([[1.,-2.],[3.,4.]]);A=np.arange(16,dtype=float).reshape(4,4);Q=A.T@A
        self.assertAlmostEqual(budget_value(x,Q,1.)['value'],float(x.ravel()@Q@x.ravel()))


if __name__=='__main__':unittest.main()
