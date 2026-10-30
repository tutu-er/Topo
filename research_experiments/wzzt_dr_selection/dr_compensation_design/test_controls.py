"""Pure contract/distribution checks; no solver, fit, loads or voltage data."""
from collections import Counter
from itertools import permutations
import unittest

import controls as c


class ControlsTests(unittest.TestCase):
    def test_sixteen_menus_and_budget(self):
        menus = c.menus()
        self.assertEqual(len(menus), 16)
        self.assertEqual(len({m['id'] for m in menus}), 16)
        self.assertEqual(Counter(m['kind'] for m in menus),
                         {'zero': 1, 'single1': 3, 'single2': 3, 'rotation': 6, 'pair1': 3})
        for m in menus:
            self.assertLessEqual(m['worst_payment'], 1.6)
            self.assertAlmostEqual(m['worst_payment'], .4 * sum(map(sum, m['price_rows'])))

    def test_law_mass_and_unique_states(self):
        for m in c.menus():
            for dependence in ('local', 'common'):
                states = c.enumerate_states(m, dependence)
                self.assertEqual(sum(p for _, p in states), 1.)
                self.assertEqual(len({s for s, _ in states}), len(states))
                self.assertTrue(all(p > 0 for _, p in states))

    def test_zero_menu_merges_all_orientation_codes(self):
        zero = ((0, 0, 0), (0, 0, 0))
        for dependence in ('local', 'common'):
            self.assertEqual(c.enumerate_states('zero', dependence), [(zero, 1.)])
            self.assertEqual(c.probability_of_state('zero', zero, zero, dependence), 1.)

    def test_exact_marginals_and_zero_orientation_mean(self):
        for m in c.menus():
            for dependence in ('local', 'common'):
                states = c.enumerate_states(m, dependence)
                for e in range(2):
                    for i in range(3):
                        accepted = sum(p * abs(s[e][i]) for s, p in states)
                        self.assertEqual(accepted, float(c.PROBABILITY[m['price_rows'][e][i]]))
                        self.assertEqual(sum(p * s[e][i] for s, p in states), 0.)

    def test_validation_is_independent_not_copied(self):
        for m in c.menus():
            for dependence in ('local', 'common'):
                states = c.enumerate_states(m, dependence)
                for st, pt in states:
                    for sv, pv in states:
                        self.assertEqual(c.probability_of_state(m, st, sv, dependence), pt * pv)
                self.assertEqual(sum(c.probability_of_state(m, st, sv, dependence)
                                     for st, _ in states for sv, _ in states), 1.)

    def test_two_events_independent(self):
        for m in c.menus():
            for dependence in ('local', 'common'):
                states = c.enumerate_states(m, dependence)
                marginal0 = Counter(); marginal1 = Counter()
                for s, p in states:
                    marginal0[s[0]] += p; marginal1[s[1]] += p
                for s, p in states:
                    self.assertEqual(p, marginal0[s[0]] * marginal1[s[1]])

    def test_common_pair_no_partial_participation(self):
        m = next(m for m in c.menus() if m['kind'] == 'pair1')
        a, b = m['pair_rows'][0]
        for s, _ in c.enumerate_states(m, 'common'):
            for row in s:
                self.assertEqual(abs(row[a]), abs(row[b]))
                self.assertEqual(row[a], -row[b])

    def test_local_pair_partial_participation_preserved(self):
        m = next(m for m in c.menus() if m['kind'] == 'pair1')
        a, b = m['pair_rows'][0]
        states = c.enumerate_states(m, 'local')
        partial_probability = sum(p for s, p in states if abs(s[0][a]) != abs(s[0][b]))
        self.assertEqual(partial_probability, .5)
        self.assertEqual(len(states), 49)

    def test_label_permutation_preserves_law(self):
        def perm_row(row, permutation):
            out = [None] * 3
            for i, j in enumerate(permutation): out[j] = row[i]
            return tuple(out)
        menus = c.menus()
        by_prices = {m['price_rows']: m for m in menus}
        for permutation in permutations(range(3)):
            for m in menus:
                mapped = by_prices[tuple(perm_row(row, permutation) for row in m['price_rows'])]
                for dependence in ('local', 'common'):
                    transformed = {tuple(perm_row(row, permutation) for row in state): p
                                   for state, p in c.enumerate_states(m, dependence)}
                    self.assertEqual(transformed, dict(c.enumerate_states(mapped, dependence)))

    def test_impossible_state_has_zero_probability(self):
        m = next(m for m in c.menus() if m['kind'] == 'pair1')
        zero = ((0, 0, 0),) * 2
        impossible = ((1, 1, 1),) * 2
        self.assertEqual(c.probability_of_state(m, impossible, zero), 0.)

    def test_original_sampler_outputs_supported_states_and_device_bounds(self):
        for m in c.menus():
            for dependence in ('local', 'common'):
                for seed in range(4):
                    draw = c.sample_menu(m, seed=seed, dependence=dependence, base_kw=200.)
                    st, sv = c.states_from_draw(draw)
                    self.assertGreater(c.probability_of_state(m, st, sv, dependence), 0.)
                    self.assertEqual(draw['worst_payment'], m['worst_payment'])
                    self.assertLessEqual(float(sum(draw['committed_payment'])), 1.6 + 1e-12)
                    self.assertTrue(draw['partial_response_cost_retained'])
                    self.assertTrue(draw['acceptance_before_code'])
                    self.assertTrue((draw['delta_p_kw'] >= -.2 - 1e-12).all())
                    self.assertTrue((draw['delta_p_kw'] <= .2 + 1e-12).all())
                    self.assertTrue((abs(draw['delta_p_kw'].sum(axis=1)) < 1e-12).all())

    def test_sampler_reproducible(self):
        m = c.menus()[-1]
        first = c.sample_menu(m, 93871)
        second = c.sample_menu(m, 93871)
        for key in ('prices', 'accepted', 'orientation', 'pairs', 'delta_p_kw', 'delta_q_kvar'):
            self.assertTrue((first[key] == second[key]).all())

    def test_baselines_named_pre_outcome_selection(self):
        for kind in ('single2', 'rotation', 'pair1'):
            for seed in range(8):
                m = c.baseline_menu(kind, seed)
                self.assertEqual(m['kind'], kind)
                self.assertEqual(m['id'], c.baseline_menu(kind, seed)['id'])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): c.enumerate_states('missing')
        with self.assertRaises(ValueError): c.enumerate_states('zero', 'unknown')
        with self.assertRaises(ValueError): c.probability_of_state('zero', ((0, 0, 0),), ((0, 0, 0),) * 2)
        with self.assertRaises(ValueError): c.sample_menu('zero', True)
        with self.assertRaises(ValueError): c.baseline_menu('unrecognized', 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
