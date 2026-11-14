"""Small independent checks of panel simulation and event bookkeeping."""
import unittest

import numpy as np

from . import probability_panel as panel


def toy(single=False):
    H = np.asarray([[[1., .2]], [[.7, -.1]]])
    if single:
        H = H[:1]
    n = len(H)
    model = dict(H=H, history=np.zeros(n), representatives=[0], targets=[0],
        means=H[:1].copy(), families=['target']+(['other'] if n == 2 else []),
        n=1, sigma2=.5, root_sd=.03, independent_sd=.1,
        covariance=np.asarray([[.03**2+.1**2]]), internal=np.zeros(n, int),
        validation_cells=3, score=dict(cells=3, reference_mse=1., validation_tie=1e-12))
    forecast = np.asarray([[1., .2], [1.3, -.1], [.7, .1]])
    powers = {name:forecast.copy() for name in panel.POLICIES}
    return model, forecast, powers


class ProbabilityPanelTests(unittest.TestCase):
    def test_exact_dedup_matches_independent_per_policy_calculation(self):
        model, forecast, powers = toy()
        # Include exact duplicates, a distinct action and a difference small
        # enough that np.allclose would incorrectly merge the arrays.
        powers['mgf_fixed'] += .2
        powers['mgf_fixed_rotation'] = powers['mgf_fixed'].copy()
        powers['normal_fixed_rotation'][0, 0] += 1e-8
        self.assertTrue(np.allclose(powers['normal_fixed_rotation'], powers['no_dfs']))
        self.assertFalse(np.array_equal(powers['normal_fixed_rotation'], powers['no_dfs']))
        draws, batch_size, seed, index, forecast_index = 19, 6, 17, 3, 2
        expected = np.empty((2, 8, draws, len(model['H'])))
        rng = [np.random.default_rng(np.random.SeedSequence([seed, index, forecast_index, c]))
               for c in range(3)]
        T, n = len(forecast), model['n']
        # Reference deliberately evaluates every policy, with no shared
        # computation or call into the optimized sampling function.
        for start in range(0, draws, batch_size):
            count = min(batch_size, draws-start)
            xi = rng[0].normal(size=(count, T, 2*n))*.05*abs(forecast)[None]
            error = (rng[1].normal(size=(count, T, n))*model['independent_sd']
                     + rng[2].normal(size=(count, T, 1))*model['root_sd'])
            for li, law in enumerate(panel.LAWS):
                for pi, policy in enumerate(panel.POLICIES):
                    X = powers[policy][None] + (xi if law == 'forecast' else 0.)
                    if len(X) == 1:
                        X = np.broadcast_to(X, (count, T, 2*n))
                    Y = X@model['means'][0].T + error
                    for j, h in enumerate(model['H']):
                        expected[li, pi, start:start+count, j] = (
                            np.square(Y-X@h.T).sum(axis=(1, 2))+model['history'][j])
        actual = panel.simulate_target(model, powers, forecast, 0, draws=draws,
            seed=seed, entry_index=index, forecast_index=forecast_index, batch_size=batch_size)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual[:, 0], actual[:, 5]))

    def test_stream_identity_batch_and_common_policy(self):
        model, forecast, powers = toy()
        args = dict(draws=17, seed=101, entry_index=2, forecast_index=1)
        a = panel.simulate_target(model, powers, forecast, 0, batch_size=3, **args)
        b = panel.simulate_target(model, powers, forecast, 0, batch_size=7, **args)
        np.testing.assert_array_equal(a, b)
        for pi in range(len(panel.POLICIES)):
            np.testing.assert_array_equal(a[:, 0], a[:, pi])
        args['entry_index'] = 3
        c = panel.simulate_target(model, powers, forecast, 0, batch_size=3, **args)
        self.assertFalse(np.array_equal(a, c))

    def test_single_native_has_vacuous_foreign_event(self):
        model, forecast, powers = toy(single=True)
        total = panel.simulate_target(model, powers, forecast, 0,
            draws=9, seed=1, entry_index=0, forecast_index=0, batch_size=4)
        rows, comparisons, events = panel.summarize_target(
            model, forecast, powers, total, 0, {}, {})
        self.assertEqual(len(rows), 16)
        self.assertTrue(events['family_success'].all())
        for row in rows:
            self.assertEqual(row['comparison_count'], 0)
            self.assertEqual(row['theoretical_representative_success_lower'], 1.)
            self.assertEqual(row['target_family_winner_count'], 9)
            self.assertTrue(row['single_family_vacuous'])
        self.assertTrue(all(r['gained'] == r['lost'] == 0 for r in comparisons))

    def test_multiple_points_one_family_and_original_tie(self):
        model, forecast, powers = toy()
        model['families'] = ['target', 'target']
        total = np.ones((2, 8, 6, 2))
        model['internal'] = np.asarray([1, 0])
        rows, _, events = panel.summarize_target(model, forecast, powers, total, 0, {}, {})
        self.assertTrue(np.all(events['winner_native'] == 1))
        self.assertTrue(all(r['comparison_count'] == 0 for r in rows))

    def test_direct_sse_moments_match_joint_gaussian_formula(self):
        model, forecast, powers = toy()
        total = panel.simulate_target(model, powers, forecast, 0,
            draws=4096, seed=9, entry_index=1, forecast_index=0, batch_size=128)
        rows, _, _ = panel.summarize_target(model, forecast, powers, total, 0, {}, {})
        for row in rows:
            pair = row['pairs'][0]
            self.assertLess(abs(pair['mean_error_in_standard_errors']), 5.)
            ratio = pair['observed_logodds_variance']/pair['variance']
            self.assertGreater(ratio, .90)
            self.assertLess(ratio, 1.10)

    def test_rotation_does_not_inherit_quote_feasibility_claim(self):
        plans = {'mgf_fixed':dict(new_gate_feasible=False, anchor_fallback=True,
                                 zero_fallback=False, executed_source='validated_anchor_fallback')}
        main = panel.quote_provenance('mgf_fixed', plans)
        rotation = panel.quote_provenance('mgf_fixed_rotation', plans)
        self.assertFalse(main['new_gate_feasible'])
        self.assertTrue(main['anchor_fallback'])
        self.assertIsNone(rotation['new_gate_feasible'])
        self.assertTrue(rotation['parent_anchor_fallback'])


if __name__ == '__main__':
    unittest.main()
