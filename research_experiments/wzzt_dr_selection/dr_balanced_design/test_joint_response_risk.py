"""Analytical counterexamples; these decision tables are NOT WZZT results."""
import unittest
import numpy as np
from research_experiments.wzzt_dr_selection.dr_balanced_design import joint_response_risk as j


class JointRiskTests(unittest.TestCase):
    def domain(self, moments=()):
        return j.binary_domain(('a:p1', 'b:p1'), ((.5, .5), (.5, .5)), moments)

    def test_frechet_extremes_and_joint_constraint(self):
        for low, high in ((0., .5), (.1, .3), (.25, .25)):
            domain = self.domain((j.Moment((0, 1), (1, 1), low, high),))
            lp = j.ExpectationLP(domain)
            both = domain.atoms.prod(axis=1).astype(float)
            self.assertAlmostEqual(lp.maximize(both), high)
            self.assertAlmostEqual(-lp.maximize(-both), low)
            for receipt in lp.receipts:
                self.assertLess(max(receipt['residuals'].values()), 1e-8)

    def test_same_marginals_reverse_menu_ranking(self):
        # Atom order 00,10,01,11. Independent .25, common .5, exclusive 0.
        a, b = np.array([1., 0, 0, 0]), np.array([0., 1, 1, 0])
        for t, winner in ((.25, 0), (.5, 1), (0., 0)):
            w = np.array([t, .5-t, .5-t, t])
            self.assertEqual(int(np.argmin([a@w, b@w])), winner)
        lp = j.ExpectationLP(self.domain())
        result = j.evaluate_regret(lp, [[a, b]], [[a, b]])
        np.testing.assert_allclose(result['regret_lower'], [.5, 1.])
        np.testing.assert_allclose(result['regret_upper'], [.5, 1.])
        self.assertAlmostEqual(lp.maximize(a)-lp.maximize(b), -.5)
        self.assertAlmostEqual(lp.maximize(a-b), .5)

    def test_shared_uncertain_cell_cancels(self):
        lp = j.ExpectationLP(self.domain())
        lo = np.zeros((1, 2, 4)); hi = np.ones((1, 2, 4))*5
        ids = np.array([[['same:'+str(i) for i in range(4)]]*2])
        result = j.evaluate_regret(lp, lo, hi, ids)
        self.assertEqual(result['regret_upper'], [0., 0.])
        self.assertFalse(result['complete_table'])
        choice = j.select_menu(result, [0., .8])
        self.assertEqual(choice['candidate_index'], 0)
        self.assertTrue(choice['selection_resolved'])

    def test_missing_mass_stays_unresolved(self):
        lp = j.ExpectationLP(self.domain())
        lo = np.zeros((1, 2, 4)); hi = np.ones((1, 2, 4))*5
        hi[0, 0] = 0
        result = j.evaluate_regret(lp, lo, hi)
        self.assertEqual(result['regret_lower'], [0., 0.])
        self.assertEqual(result['regret_upper'], [0., 5.])
        # If BOTH menus have missing cells, a free quote does not justify fill0.
        hi[0, 0] = 5
        result = j.evaluate_regret(j.ExpectationLP(self.domain()), lo, hi)
        self.assertFalse(j.select_menu(result, [0., .8])['selection_resolved'])

    def test_positive_weak_block_can_be_missed_by_all(self):
        losses = np.ones((1, 3, 4))
        result = j.evaluate_regret(j.ExpectationLP(self.domain()), losses, losses)
        self.assertEqual(result['regret_upper'], [0., 0., 0.])
        self.assertEqual(result['risk'][0][0], dict(lower=1., upper=1.))
        self.assertEqual(j.select_menu(result, [0., .8, 1.2])['candidate_index'], 0)

    def test_shared_zero_model_does_not_flatten_other_explanations(self):
        a, b = [1., 0, 0, 0], [0., 1, 1, 0]
        losses = np.array([[a, b], [[0.]*4, [0.]*4]])
        result = j.evaluate_regret(j.ExpectationLP(self.domain()), losses, losses)
        np.testing.assert_allclose(result['regret_upper'], [.5, 1.])

    def test_cross_price_mechanism_is_common_and_nonmonotone_allowed(self):
        domain = j.binary_domain(('node0:e0:p1', 'node0:e0:p2'), ((.5, .5), (.75, .75)))
        lp = j.ExpectationLP(domain)
        nonmonotone = ((domain.atoms[:, 0] == 1) & (domain.atoms[:, 1] == 0)).astype(float)
        self.assertAlmostEqual(lp.maximize(nonmonotone), .25)
        np.testing.assert_array_equal(j.projected_keys(domain, (-1,)), [0, 0, 0, 0])
        self.assertEqual(set(j.projected_keys(domain, (0,))), {0, 1})

    def test_linear_delivery_fee_is_dependence_invariant(self):
        domain = j.binary_domain(('p1', 'p2'), ((.5, .5), (.75, .75)))
        fee = .2*(domain.atoms[:, 0]+2*domain.atoms[:, 1])
        lp = j.ExpectationLP(domain)
        self.assertAlmostEqual(lp.maximize(fee), .4)
        self.assertAlmostEqual(-lp.maximize(-fee), .4)

    def test_a_narrow_joint_interval_can_resolve_an_unstable_ranking(self):
        domain = self.domain((j.Moment((0, 1), (1, 1), .1, .2),))
        losses = np.array([[[1., 0, 0, 0], [0., 1, 1, 0]]])
        result = j.evaluate_regret(j.ExpectationLP(domain), losses, losses)
        np.testing.assert_allclose(result['regret_upper'], [0., .7])

    def test_frozen_domain_prevents_stale_probability_cache(self):
        domain = self.domain()
        with self.assertRaises(ValueError):
            domain.b_eq[0] = .5

    def test_pair_projection_preserves_exact_extrema_without_full_atom_table(self):
        bounds = {'a': (.5, .5), 'b': (.75, .75), 'omitted': (.3, .3)}
        full = j.binary_domain(tuple(bounds), list(bounds.values()))
        small = j.marginal_projection(bounds, ('a', 'b'))
        def objective(domain):
            return ((domain.atoms[:, 0] == 1) & (domain.atoms[:, 1] == 0)).astype(float)
        self.assertAlmostEqual(j.ExpectationLP(full).maximize(objective(full)), .25)
        self.assertAlmostEqual(j.ExpectationLP(small).maximize(objective(small)), .25)
        # 24 global queries would need 2^24 types; pair projection needs only 4.
        larger = {**bounds, **{str(i): (.5, .5) for i in range(21)}}
        self.assertEqual(len(j.marginal_projection(larger, ('a', 'b')).atoms), 4)

    def test_projection_does_not_drop_extra_global_moments(self):
        with self.assertRaises(ValueError):
            j.marginal_projection({'a': (.5, .5), 'b': (.5, .5)}, ('a',),
                                  extra_moments=(j.Moment((0, 1), (1, 1), .25, .25),))

    def test_unfeasible_joint_probabilities_raise_without_zero_result(self):
        domain = self.domain((j.Moment((0, 1), (1, 1), .6, .6),))
        lp = j.ExpectationLP(domain)
        with self.assertRaises(RuntimeError):
            lp.maximize(np.zeros(4))
        self.assertEqual(lp.receipts[-1]['status'], 2)
        self.assertIsNone(lp.receipts[-1]['value'])

    def test_names_union_size_and_inconsistent_cache_are_rejected(self):
        with self.assertRaises(ValueError):
            j.binary_domain(('a', 'a'), ((.5, .5), (.5, .5)))
        with self.assertRaises(ValueError):
            j.binary_domain(tuple(map(str, range(17))), [(.5, .5)]*17)
        lp = j.ExpectationLP(self.domain())
        lo = np.zeros((1, 2, 4)); hi = np.ones((1, 2, 4))
        hi[0, 1, 0] = 2
        with self.assertRaises(ValueError):
            j.evaluate_regret(lp, lo, hi, np.full(lo.shape, 'same'))


if __name__ == '__main__':
    unittest.main()
