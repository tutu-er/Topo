"""Independent contract, probability, and physical checks; no loads/solver."""
from collections import Counter
from itertools import permutations
import unittest
import numpy as np
import menus as c


class MenuTests(unittest.TestCase):
    def test_exact_catalog_extension_and_original_identity(self):
        full = c.menus()
        self.assertEqual(full[:16], c._old.menus())
        self.assertEqual(len(full), 22)
        self.assertEqual(len({m['id'] for m in full}), 22)
        self.assertEqual(len({(m['price_rows'], m['pair_rows']) for m in full}), 22)
        self.assertEqual(Counter(m['kind'] for m in full),
                         dict(zero=1, single1=3, single2=3, rotation=6, pair1=3, pair_rotation=6))
        for m in full[16:]:
            self.assertNotEqual(m['pair_rows'][0], m['pair_rows'][1])
            self.assertTrue(all(sorted(row) == [0, 1, 1] for row in m['price_rows']))

    def test_budget_and_expected_cost(self):
        expected = dict(zero=0., single1=.4, single2=1.2, rotation=1.2, pair1=.8, pair_rotation=.8)
        for m in c.menus():
            self.assertLessEqual(m['worst_payment'], 1.6)
            self.assertAlmostEqual(m['worst_payment'], .4 * sum(map(sum, m['price_rows'])))
            self.assertAlmostEqual(c.expected_payment(m), expected[m['kind']])
            p = sum(prob * .4 * sum(m['price_rows'][e][i] * abs(s[e][i]) for e in range(2) for i in range(3))
                    for s, prob in c.enumerate_states(m))
            self.assertAlmostEqual(p, c.expected_payment(m))

    def test_normalized_unique_laws(self):
        for m in c.menus():
            for law in ('local', 'common'):
                states = c.enumerate_states(m, law)
                self.assertEqual(sum(p for _, p in states), 1.)
                self.assertEqual(len({s for s, _ in states}), len(states))
                self.assertTrue(all(p > 0 for _, p in states))
                for s, p in states:
                    self.assertEqual(c.probability_of_training_state(m['id'], s, law), p)

    def test_pair_rotation_local_has_seven_by_seven_states(self):
        for m in c.menus()[16:]:
            states = c.enumerate_states(m)
            self.assertEqual(len(states), 49)
            marginals = [Counter(), Counter()]
            for s, p in states:
                for e in range(2): marginals[e][s[e]] += p
            self.assertTrue(all(len(v) == 7 for v in marginals))
            for s, p in states:
                self.assertEqual(p, marginals[0][s[0]] * marginals[1][s[1]])

    def test_partial_zero_and_balanced_modes_are_not_removed(self):
        for m in c.menus()[16:]:
            states = c.enumerate_states(m)
            for e in range(2):
                both = sum(p for s, p in states if sum(abs(v) for v in s[e]) == 2)
                partial = sum(p for s, p in states if sum(abs(v) for v in s[e]) == 1)
                none = sum(p for s, p in states if sum(abs(v) for v in s[e]) == 0)
                self.assertEqual((both, partial, none), (.25, .5, .25))
                for s, _ in states:
                    if sum(abs(v) for v in s[e]) == 2: self.assertEqual(sum(s[e]), 0)
            both_rounds = sum(p for s, p in states if all(sum(abs(v) for v in row) == 2 for row in s))
            self.assertEqual(both_rounds, 1 / 16)

    def test_common_response_is_correlated_but_not_forced(self):
        for m in c.menus()[16:]:
            states = c.enumerate_states(m, 'common')
            self.assertEqual(len(states), 9)
            for e in range(2):
                self.assertEqual(sum(p for s, p in states if sum(abs(v) for v in s[e]) == 2), .5)
                self.assertEqual(sum(p for s, p in states if sum(abs(v) for v in s[e]) == 1), 0.)
            self.assertEqual(sum(p for s, p in states if all(sum(abs(v) for v in row) == 2 for row in s)), .25)

    def test_zero_mean_does_not_mean_zero_response(self):
        for m in c.menus():
            for law in ('local', 'common'):
                states = c.enumerate_states(m, law)
                for e in range(2):
                    for i in range(3):
                        self.assertEqual(sum(p * s[e][i] for s, p in states), 0.)
                        self.assertEqual(sum(p * abs(s[e][i]) for s, p in states), float(c.PROBABILITY[m['price_rows'][e][i]]))

    def test_validation_is_independent_repeated_price_not_outcome(self):
        for m in c.menus():
            states = c.enumerate_states(m)
            total = 0.
            for st, pt in states:
                for sv, pv in states:
                    p = c.probability_of_state(m, st, sv)
                    self.assertEqual(p, pt * pv)
                    total += p
            self.assertEqual(total, 1.)

    def test_label_permutations_preserve_full_library_and_laws(self):
        def perm(row, p):
            out = [0] * 3
            for i, j in enumerate(p): out[j] = row[i]
            return tuple(out)
        # Pair order changes global event orientation only, already symmetric.
        by_prices = {m['price_rows']: m for m in c.menus()}
        for p in permutations(range(3)):
            for m in c.menus():
                mapped = by_prices[tuple(perm(row, p) for row in m['price_rows'])]
                for law in ('local', 'common'):
                    transformed = {tuple(perm(row, p) for row in s): q for s, q in c.enumerate_states(m, law)}
                    self.assertEqual(transformed, dict(c.enumerate_states(mapped, law)))

    def test_sampler_keeps_original_bounds_energy_q_and_partial_cost(self):
        for m in c.menus():
            for law in ('local', 'common'):
                for seed in (0, 1, 7, 93871):
                    d = c.sample_menu(m['id'], seed, law)
                    st, sv = c.states_from_draw(d)
                    self.assertGreater(c.probability_of_state(m, st, sv, law), 0.)
                    np.testing.assert_array_equal(d['prices'][:2], m['price_rows'])
                    np.testing.assert_array_equal(d['prices'][2:], m['price_rows'])
                    dp = d['delta_p_kw']; power = .2 + dp
                    self.assertTrue((power >= -1e-12).all() and (power <= .4 + 1e-12).all())
                    np.testing.assert_allclose(power.sum(axis=1) * .25, .4, atol=1e-12)
                    np.testing.assert_allclose(dp.sum(axis=1), 0., atol=1e-12)
                    np.testing.assert_allclose(d['delta_q_kvar'], .35 * dp, atol=1e-15)
                    np.testing.assert_allclose(d['committed_payment'], .2 * np.sum(d['prices'] * d['accepted'], axis=1))
                    self.assertLessEqual(float(sum(d['committed_payment'])), 1.6 + 1e-12)
                    self.assertTrue(d['partial_response_cost_retained'] and d['acceptance_before_code'])
                    self.assertFalse(d['device_declaration']['empirical_calibration'])

    def test_original_samplers_preserved_for_old_menus(self):
        for m in c.menus()[:16]:
            actual = c.sample_menu(m, 718)
            expected = c._old.sample_menu(m, 718)
            for key in ('prices', 'accepted', 'orientation', 'pairs', 'delta_p_kw', 'delta_q_kvar', 'committed_payment'):
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_named_baselines_reproducible_without_outcomes(self):
        for kind in ('single2', 'rotation', 'pair1', 'pair_rotation'):
            for seed in range(5):
                self.assertEqual(c.baseline_menu(kind, seed), c.baseline_menu(kind, seed))
                self.assertEqual(c.baseline_menu(kind, seed)['kind'], kind)
        for kind in ('single2', 'rotation', 'pair1'):
            self.assertEqual(c.baseline_menu(kind, 71), c._old.baseline_menu(kind, 71))

    def test_inputs_reject_unknown_or_over_budget_contract(self):
        with self.assertRaises(ValueError): c.resolve('missing')
        with self.assertRaises(ValueError): c.baseline_menu('missing', 1)
        with self.assertRaises(ValueError): c.sample_menu('zero', True)
        with self.assertRaises(ValueError): c.enumerate_states('zero', 'invalid')
        bad = dict(c.menus()[-1], price_rows=((2, 2, 0), (2, 0, 2)))
        with self.assertRaises(ValueError): c.resolve(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
