import unittest
from .guard import choose_output
from .study import mechanism

class OutputChecks(unittest.TestCase):
    def test_given_positive_weak_point_survives(self):
        p,tr,va,c=mechanism('positive_weak');r=choose_output(p,tr,va,c)
        self.assertEqual(r['chosen_index'],0)
        self.assertEqual(len(r['point']['supports']),3)
        self.assertAlmostEqual(r['candidate_points'][0]['validation_mae'],0.,places=14)
        self.assertAlmostEqual(r['candidate_points'][1]['validation_mae'],.00065,places=12)
    def test_given_zero_common_block_stays_contracted(self):
        p,tr,va,c=mechanism('zero_common');r=choose_output(p,tr,va,c)
        self.assertEqual(r['chosen_index'],1)
        self.assertEqual(len(r['point']['supports']),2)
        self.assertAlmostEqual(r['point']['validation_mae'],0.,places=14)

if __name__=='__main__':unittest.main()
