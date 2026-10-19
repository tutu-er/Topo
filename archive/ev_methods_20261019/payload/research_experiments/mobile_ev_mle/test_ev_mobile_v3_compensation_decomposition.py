import unittest
from . import ev_mobile_v3_compensation_decomposition as audit


class CompensationTests(unittest.TestCase):
    def test_change_identity(self):
        r=audit.decomposition(dict(a=20.,b=30.),dict(a=10.,b=15.),dict(a=50.,b=100.),dict(a=12.,b=20.),'a','b')
        self.assertEqual(r['delta']['fixed_gap'],40.)
        self.assertEqual(r['delta']['relative_compensation'],37.)
        self.assertEqual(r['delta']['joint_gap'],3.)
        self.assertEqual(r['delta']['identity_error'],0.)

    def test_worse_joint_fit_is_failure(self):
        with self.assertRaises(ValueError):
            audit.decomposition(dict(a=10.,b=10.),dict(a=11.,b=9.),dict(a=10.,b=10.),dict(a=9.,b=9.),'a','b')


if __name__=='__main__':unittest.main()
