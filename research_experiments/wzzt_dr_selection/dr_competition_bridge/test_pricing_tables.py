"""Small independent arithmetic tests for the price distribution interface."""
from fractions import Fraction as F
import itertools
import math
import unittest
import numpy as np

import pricing_tables as p


class PricingTableTests(unittest.TestCase):
    def test_counts_and_full_price_domains(self):
        for n, count in ((1, 9), (2, 50), (3, 168), (4, 423), (15, 45446)):
            menus = p.enumerate_menus(n)
            self.assertEqual(len(menus), count)
            self.assertEqual(len({tuple(a.flat) for a in menus}), count)
            self.assertTrue((menus.sum(axis=(1, 2)) <= 4).all())
            self.assertTrue((menus[0] == 0).all())
        for n in (1, 2, 3, 4):
            # Independent brute-force enumeration, not the module recursion.
            brute = {row for row in itertools.product(range(3), repeat=2*n) if sum(row) <= 4}
            self.assertEqual({tuple(a.flat) for a in p.enumerate_menus(n)}, brute)
        self.assertEqual(len(p.single_node_menus(15)), 121)

    def test_reachable_training_and_four_event_masks(self):
        for n, count in ((2, 16), (3, 57), (4, 163)):
            menus = p.enumerate_menus(n)
            self.assertEqual(len(p.unique_training_masks(menus)), count)
        menu = np.ones((2, 2), dtype=np.uint8)
        outcomes = p.enumerate_event_outcomes(menu, 'local')
        self.assertEqual(len(outcomes), 256)
        self.assertEqual(sum(q for _, q in outcomes), 1)
        self.assertEqual({q for _, q in outcomes}, {F(1, 256)})
        for mask, _ in outcomes:
            self.assertEqual(mask.shape, (4, 2))
            np.testing.assert_array_equal(p.masks_from_key(p.outcome_key(mask), 2), mask)

    def test_probability_mass_marginals_and_validation_independence(self):
        for menu in p.enumerate_menus(2):
            for law in ('local', 'common'):
                outcomes = p.enumerate_outcome_keys(menu, law)
                self.assertEqual(sum(q for _, q in outcomes), 1)
                pairs = dict(p.enumerate_training_outcomes(menu, law))
                train_marginal = {k:F(0) for k in pairs}
                valid_marginal = {k:F(0) for k in pairs}
                for key, q in outcomes:
                    train_marginal[key[:2]] += q
                    valid_marginal[key[2:]] += q
                    self.assertEqual(q, pairs[key[:2]]*pairs[key[2:]])
                self.assertEqual(train_marginal, pairs)
                self.assertEqual(valid_marginal, pairs)
                expected = F(0)
                for key, q in outcomes:
                    amount = p.actual_payment(menu, key)
                    self.assertLessEqual(amount, F(8, 5))
                    expected += q*amount
                self.assertEqual(expected, p.payment(menu)['expected'])

    def test_common_law_matches_uniform_threshold_intervals(self):
        menu = np.array([[1, 2], [2, 1]], dtype=np.uint8)
        rows = p.enumerate_training_outcomes(menu, 'common')
        first = {0:F(1,4), 2:F(1,4), 3:F(1,2)}
        second = {0:F(1,4), 1:F(1,4), 3:F(1,2)}
        self.assertEqual(dict(rows), {(a,b):qa*qb for a,qa in first.items() for b,qb in second.items()})
        same = p.enumerate_event_outcomes(np.ones((2,2), dtype=np.uint8), 'common')
        self.assertEqual(len(same), 16)
        for mask, _ in same:
            np.testing.assert_array_equal(mask[:,0], mask[:,1])

    def test_input_difference_is_not_monotone_in_number_of_sites(self):
        single = np.array([[2,0],[2,0]], dtype=np.uint8)
        sync = np.ones((2,2), dtype=np.uint8)
        probability = lambda menu:sum(q for mask,q in p.enumerate_event_outcomes(menu)
                                      if (mask[:,0] != mask[:,1]).any())
        self.assertEqual(probability(single), F(255,256))
        self.assertEqual(probability(sync), F(15,16))
        self.assertEqual(p.payment(single)['worst'], F(8,5))
        self.assertEqual(p.payment(sync)['worst'], F(8,5))
        self.assertEqual(p.payment(single)['expected'], F(6,5))
        self.assertEqual(p.payment(sync)['expected'], F(4,5))

    def test_probability_integration_and_cost_tie_keep_zero(self):
        menus = p.enumerate_menus(2)
        constant = lambda key:7
        result = p.score_menus(menus, constant, tie_seed=972)
        self.assertEqual(result['selected_index'], 0)
        self.assertEqual(result['records'][0]['expected_fee'], 0)
        # Expected count of participant events has direct sum-of-marginals form.
        loss = lambda key:sum(value.bit_count() for value in key)
        for law in ('local', 'common'):
            for menu in menus:
                direct = 2*sum((F(0),F(1,2),F(3,4))[int(value)] for value in menu.flat)
                self.assertAlmostEqual(p.expected_loss(loss, menu, law), float(direct))

    def test_missing_nonfinite_invalid_and_seed_ties(self):
        with self.assertRaises(KeyError):
            p.expected_loss({}, np.ones((2,2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            p.expected_loss(lambda key:math.inf, np.zeros((2,2), dtype=np.uint8))
        for menu in (np.zeros((3,2)), [[.5,0],[0,0]], [[-1,0],[0,0]]):
            with self.assertRaises(ValueError):
                p.enumerate_outcome_keys(menu)
        for value in (0, True, 1.5):
            with self.assertRaises(ValueError):
                p.enumerate_menus(value)
        a = np.array([[2,0],[2,0]], dtype=np.uint8)
        b = np.array([[0,2],[0,2]], dtype=np.uint8)
        selected = p.score_menus(np.stack((a,b)), lambda key:0, tie_seed=415)
        expected = min(range(2), key=lambda i:(p.menu_tie_key((a,b)[i],415),i))
        self.assertEqual(selected['selected_index'], expected)


if __name__ == '__main__':
    unittest.main()
