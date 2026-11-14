"""Small independent mathematical checks; no AC, fitting, or optimization."""
import unittest

import numpy as np

from .placement import choose_nodes, exact_pair_q_box


class PlacementChecks(unittest.TestCase):
    def test_scalar_hourly_sign_closed_form(self):
        # For one scalar output, a common hourly sign is chosen from the sum of
        # its four residuals, not independently for each quarter hour.
        F = np.zeros((96, 2))
        F[:, 1] = np.resize([1., -2., .4, -.1, 2., .2, -1., -.2], 96)
        D = np.array([[0., 2.]])
        sigma2, capacity, base = 3., 5., 10.
        result = exact_pair_q_box(D, F, [0], capacity, sigma2, base)
        d = 2*F[:, 1]
        a = 2*capacity/base
        expected = (np.dot(d, d)+96*a*a+2*abs(d.reshape(24, 4).sum(1)*a).sum())/(2*sigma2)
        self.assertAlmostEqual(result['J_max'], expected, places=11)
        unconstrained_quarter = (np.abs(d)+abs(a))@(np.abs(d)+abs(a))/(2*sigma2)
        self.assertLess(result['J_max'], unconstrained_quarter)

    def test_joint_columns_keep_cross_terms(self):
        # Q columns (1,0) and (-1,2): the largest squared norm is 8,
        # while summing their squared norms gives 6 and is not a maximum.
        D = np.array([[0., 0., 1., -1.], [0., 0., 0., 2.]])
        result = exact_pair_q_box(D, np.zeros((96, 4)), [0, 1], 1., 1., 1.)
        self.assertAlmostEqual(result['J_max'], 96*8/2)
        self.assertNotEqual(result['J_max'], 96*6/2)

    def test_weakest_pair_cannot_be_ignored(self):
        score = np.array([[10., 0., 0.], [0., 6., 6.]])
        selected = choose_nodes(score, [0, 1, 2], [301, 302, 303], count=2)
        self.assertEqual(selected['node_labels'], [301, 302])
        self.assertEqual(selected['minimum_sensitivity'], 6.)
        self.assertEqual(selected['combinations_evaluated'], 3)


if __name__ == '__main__':
    unittest.main()
