import unittest
import numpy as np
from .line_polish import polish_resources
from .test_line_seed import Case,PolynomialEvidence
from .solver import BUFFER


class PolishTests(unittest.TestCase):
    def test_removes_irrelevant_high_fee_column_then_reduces_needed_column(self):
        case=Case(2);e=PolynomialEvidence(q=0.,a=.5,c=-.5+BUFFER,resources=2)
        receipt,u=polish_resources(case,e,np.array([[.6,1.]]))
        self.assertEqual(receipt['resource_order'],[1,0])
        self.assertEqual(receipt['accepted_steps'],2)
        self.assertAlmostEqual(float(u[0,1]),0.,places=8)
        self.assertAlmostEqual(float(u[0,0]),.5,places=7)
        self.assertAlmostEqual(receipt['cost_yuan'],.5,places=7)
        self.assertFalse(receipt['global_qcqp_optimality_claimed'])

    def test_coupled_physical_endpoint_is_skipped(self):
        class Coupled(Case):
            def audit(self,u):
                a=super().audit(u)
                a['passed']=bool(a['passed'] and abs(self.array(u).sum()-1)<1e-10)
                return a
        case=Coupled(2);e=PolynomialEvidence(q=0.,c=1.,resources=2)
        original=np.array([[.6,.4]])
        receipt,u=polish_resources(case,e,original)
        np.testing.assert_array_equal(u,original)
        self.assertEqual(receipt['accepted_steps'],0)
        self.assertTrue(all(s['status']=='zeroed_endpoint_resource_infeasible' for s in receipt['steps']))

    def test_invalid_initial_point_is_rejected(self):
        case=Case();e=PolynomialEvidence(q=0.,c=-1.)
        with self.assertRaises(ValueError):polish_resources(case,e,np.zeros(case.shape))


if __name__=='__main__':unittest.main()
