"""Independent post-freeze small checks; no AC, topology fit or full planning."""
import unittest

import numpy as np

from spatial_optimizer import cost, feasible, cone_step, SpatialProbability
from response_law import MAIN_LAW, MISSPEC_LAW, draw, linear_factor, analytic_covariance


class IndependentSpatialOptimizerTests(unittest.TestCase):
    def test_old_contract_embeds_exactly(self):
        u = np.random.default_rng(3401).uniform(-1, 1, (2, 96))
        self.assertAlmostEqual(cost(u, np.ones(2)), .25*np.sum(u*u/.5), places=12)
        self.assertTrue(feasible(u, np.ones(2), np.ones(2)))

    def test_dividing_one_resource_does_not_reduce_cost(self):
        u = np.array([[.7, -.2, .5, -.6]])
        allocation = np.array([.2, .3, .5])
        split_u = allocation[:, None]*u
        self.assertAlmostEqual(cost(u, np.ones(1)), cost(split_u, allocation), places=14)

    def test_capacity_cauchy_bound(self):
        rng = np.random.default_rng(3402)
        a = rng.uniform(.1, 1, 7)
        a *= 2/a.sum()
        u = rng.uniform(-1, 1, (7, 12))*a[:, None]
        bound = .5*np.sum(np.sum(np.abs(u), axis=0)**2)/a.sum()
        self.assertGreaterEqual(cost(u, a)+1e-13, bound)

    def test_zero_capacity_cost_domain(self):
        self.assertEqual(cost(np.zeros((3, 4)), np.zeros(3)), 0)
        with self.assertRaises(ValueError):
            cost(np.ones((1, 1)), np.zeros(1))

    def test_joint_resource_constraints(self):
        caps = np.array([1., .4, .8])
        a = np.array([.8, .4, .8])
        self.assertTrue(feasible(np.tile(a[:, None], (1, 3)), a, caps))
        self.assertFalse(feasible(np.zeros((3, 3)), [1., .4, .8], caps))
        self.assertFalse(feasible(np.array([[.9], [0], [0]]), a, caps))

    def test_rotation_fee_and_quote_match(self):
        a = np.array([.5, .25, .75, .5])
        desired = 43.7
        multiplier = np.sqrt(desired/(48*a.sum()))
        signs = np.where((np.arange(96)[None, :]//24+np.arange(4)[:, None])%2 == 0, 1., -1.)
        u = multiplier*a[:, None]*signs
        self.assertAlmostEqual(cost(u, a), desired, places=12)
        self.assertLessEqual(float(np.max(np.abs(u)/(.5*a[:, None]))), 2)

    def test_response_factor_exact_covariance(self):
        for law in [MAIN_LAW, MISSPEC_LAW]:
            factor = linear_factor(3, 5, law)
            actual = factor@factor.T
            expected = analytic_covariance(3, 5, law)
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
            np.testing.assert_allclose(np.diag(actual), 1, atol=2e-15, rtol=0)
        self.assertEqual(MISSPEC_LAW.mean, .9)
        self.assertEqual(MISSPEC_LAW.relative_sd, MAIN_LAW.relative_sd)

    def test_fixed_conic_step_has_analytic_answer(self):
        step, info = cone_step(np.zeros((1, 1)), np.ones(1), .5, np.ones((1, 1)),
                               np.ones(1), target=.7, radius=1, total=1,
                               fixed_capacity=np.ones(1), seconds=10)
        self.assertIsNotNone(step)
        u, a = step
        self.assertTrue(feasible(u, a, np.ones(1), total=1))
        np.testing.assert_allclose(u, [[.2]], atol=3e-5, rtol=0)
        self.assertAlmostEqual(cost(u, a), .02, delta=1e-5)
        self.assertLess(info["slack"], 1e-6)

    def test_free_capacity_conic_step(self):
        # Two equal sites, two equal times: sum(u)=.4, a_i=1, u_it=.1.
        step, info = cone_step(np.zeros((2, 2)), np.ones(2), .5, np.ones((2, 2)),
                               np.ones(2), target=.9, radius=1, total=2, seconds=10)
        self.assertIsNotNone(step)
        u, a = step
        self.assertTrue(feasible(u, a, np.ones(2)))
        np.testing.assert_allclose(u, .1, atol=5e-5, rtol=0)
        np.testing.assert_allclose(a, 1, atol=5e-4, rtol=0)
        self.assertAlmostEqual(cost(u, a), .02, delta=1e-5)
        self.assertLess(info["projection_max"], 1e-6)

    def test_zero_capacity_in_conic_step(self):
        step, _ = cone_step(np.zeros((2, 2)), np.array([1., 0]), .5, np.ones((2, 2)),
                            np.array([1., 0]), target=.7, radius=1,
                            fixed_capacity=np.array([1., 0]), seconds=10)
        self.assertIsNotNone(step)
        u, a = step
        self.assertEqual(a[1], 0)
        np.testing.assert_array_equal(u[1], 0)
        self.assertTrue(np.isfinite(cost(u, a)))

    def test_correlated_probability_uses_multiplier_once(self):
        rng = np.random.default_rng(3403)
        packet = dict(shape=(2, 2), all_node_count=3, active_positions=[0, 2],
                      cv=np.full((2, 2), MAIN_LAW.relative_sd), elasticities=np.full((2, 2), .5),
                      dt=.25, kappa=1.03, mu_C=rng.normal(size=4), mu_F=rng.normal(size=4),
                      L_C=rng.normal(size=(4, 1)), L_F=rng.normal(size=(4, 1)),
                      effect_C=rng.normal(size=(2, 2)), effect_F=rng.normal(size=(2, 2)),
                      direction=np.ones(1), complement=np.zeros((1, 0)), threshold=.1,
                      response_factor=None)
        problem = SpatialProbability(packet, 8, 3404, 3405)
        expected = draw(3405, 8, 3, 2, MAIN_LAW)[:, [0, 2], :]
        np.testing.assert_array_equal(problem.core.multiplier, expected)
        u = np.array([[.12, -.2], [.08, .1]])
        answer = problem.evaluate(u)
        self.assertIsNotNone(answer["gradient"])
        direction = np.array([[.3, .4], [-.5, .2]])
        eps = 1e-6
        difference = (problem.evaluate(u+eps*direction)["probability"]-
                      problem.evaluate(u-eps*direction)["probability"])/(2*eps)
        analytic = float(answer["gradient"]@direction.ravel())
        self.assertAlmostEqual(difference, analytic, delta=2e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
