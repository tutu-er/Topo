import unittest
import numpy as np
from .budget_bound import ellipsoid_upper


class BoundTests(unittest.TestCase):
    def test_scalar_exact_and_negative_curvature(self):
        # u²<=1 and box |u|<=1; the regularization leaves the same interval.
        result = ellipsoid_upper(np.array([[[2.]]]), np.array([[3.]]), -4.,
            np.array([[1.]]), 1., 1., .01)
        self.assertGreaterEqual(result['upper_bound'], 4.)
        self.assertAlmostEqual(result['raw_upper_bound'], 4., places=7)
        result = ellipsoid_upper(np.array([[[-1.]]]), np.array([[.2]]), 1.,
            np.array([[1.]]), 1., 1., .01)
        self.assertGreaterEqual(result['upper_bound'], 1.04)
        self.assertAlmostEqual(result['raw_upper_bound'], 1.04, places=7)

    def test_nullspace_and_random_feasible_points(self):
        rng = np.random.default_rng(2026111521)
        for _ in range(10):
            A = rng.normal(size=(2, 3, 3)); A = (A+A.swapaxes(1, 2))/2
            a = rng.normal(size=(2, 3)); W = np.diag([.2, 1., 0.])
            result = ellipsoid_upper(A, a, -.3, W, .5, 6., .01)
            u = rng.uniform(-1, 1, size=(1000, 2, 3))
            energy = np.einsum('thi,ij,thj->t', u, W, u)
            u = u[energy <= .5]
            scores = np.einsum('thi,hij,thj->t', u, A, u)+2*np.einsum('hi,thi->t', a, u)-.3
            self.assertTrue(np.all(scores <= result['upper_bound']))
            self.assertGreater(result['lambda_dual'], max(0., result['maximum_whitened_eigenvalue']))


if __name__ == '__main__':
    unittest.main()
